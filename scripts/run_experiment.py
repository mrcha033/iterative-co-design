"""
Main experiment runner for iterative co-design methods.

This script orchestrates experiments comparing different optimization strategies:
- Dense baseline (no optimization)
- Sparsity-only (HDS applied alone)
- Permutation-only (IASP applied alone)
- Linear pipeline (IASP then HDS sequentially)
- Iterative co-design (HDS and IASP in feedback loop)

The script uses Hydra for configuration management and supports various models
(Mamba, BERT) and datasets (WikiText-103, SST-2). Results are automatically
saved and can be logged to Weights & Biases.

Usage:
    python scripts/run_experiment.py model=mamba_370m dataset=wikitext103 method=iterative
    python scripts/run_experiment.py model=bert_base dataset=sst2 method=dense dry_run=true
"""

from pathlib import Path

# ------------------------------------------------------------------
# Ensure the project's local 'src' package directory has highest import
# precedence and force reload of local 'utils' and 'co_design' packages to
# avoid conflicts with similarly named third-party packages installed in the
# environment.
# ------------------------------------------------------------------
import sys
import importlib

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Remove already-imported third-party variants, then invalidate caches so that
# subsequent imports resolve to the project's versions.
for _pkg in ("utils", "co_design"):
    if _pkg in sys.modules:
        del sys.modules[_pkg]
importlib.invalidate_caches()

import json
import random
import numpy as np
import torch
import fnmatch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)
from datasets import load_dataset
import hydra
from omegaconf import DictConfig
from utils.logging import initialize_wandb
import wandb

from utils.evaluation import calculate_task_metric
from utils.profiler import LatencyProfiler
from utils.cleanup import cleanup_old_runs
from utils.input import make_dummy_input
from co_design.iasp import run_iasp_on_mamba, run_iasp_on_bert
from models.wrapper import ModelWrapper
from co_design.hds import apply_hds
from co_design.layout_aware import apply_layout_aware_hds_finetuning
import logging
from inspect import getmro

logger = logging.getLogger(__name__)

def set_random_seeds(seed: int):
    """Set random seeds for reproducible experiments."""
    print(f"Setting random seeds to {seed} for reproducible results")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_and_data(cfg: DictConfig):
    """Loads model, tokenizer, and dataset based on the config."""
    print(f"Loading model: {cfg.model.name}")
    if cfg.model.task == "language_modeling":
        model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    elif cfg.model.task == "sequence_classification":
        model = AutoModelForSequenceClassification.from_pretrained(cfg.model.name)
    else:
        raise ValueError(f"Unknown task: {cfg.model.task}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {cfg.dataset.name}")
    try:
        dataset = load_dataset(cfg.dataset.path, cfg.dataset.get("subset"))
    except Exception:
        logger.warning(f"Could not find dataset at path '{cfg.dataset.path}'. Attempting to download from Hub...")
        dataset = load_dataset(cfg.dataset.name, cfg.dataset.get("subset"))

    eval_split = "validation"
    if eval_split not in dataset:
        eval_split = "test"
    
    eval_dataset = dataset[eval_split]

    def tokenize_function(examples):
        text_column = "text" if "text" in examples else "sentence"
        return tokenizer(
            examples[text_column], 
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    tokenized_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    data_loader = DataLoader(tokenized_dataset, batch_size=cfg.dataset.batch_size, collate_fn=default_data_collator)

    return model, tokenizer, data_loader, eval_dataset


def save_results(cfg: DictConfig, method: str, metrics: dict):
    output_dir = Path.cwd()
    results_file = output_dir / f"{method}_metrics.json"
    print(f"Saving results to {results_file}")
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=4)
    
    if wandb.run:
        wandb.log(metrics)


def _expand_target_layers(model, target_spec):
    """Expands a wildcard target layer specification into a list of layer names."""
    if isinstance(target_spec, str) and "*" in target_spec:
        all_layers = [name for name, _ in model.named_modules()]
        expanded_layers = fnmatch.filter(all_layers, target_spec)
        if not expanded_layers:
            logger.warning(f"Wildcard '{target_spec}' did not match any layers.")
        else:
            logger.info(f"Expanded wildcard '{target_spec}' to {len(expanded_layers)} layers.")
        return expanded_layers
    # If it's already a list or a single layer name without wildcards
    return target_spec if isinstance(target_spec, list) else [target_spec]


def _measure_and_collect_metrics(
    wrapped_model, tokenizer, data_loader, eval_dataset, profiler, cfg, modularity: float = 0.0
):
    """Helper function to measure and collect all relevant metrics."""
    logger.info("Calculating evaluation metric...")
    
    # Use the correct metric for the task
    if cfg.model.task == "sequence_classification":
        # Note: This requires an accuracy function in utils.evaluation
        task_metric = calculate_task_metric(
            wrapped_model.model, data_loader, "accuracy"
        )
    else:
        task_metric = calculate_task_metric(
            wrapped_model.model, data_loader, "perplexity"
        )

    metric_name, metric_value = list(task_metric.items())[0]
    logger.info(f"{metric_name.title()}: {metric_value:.4f}")

    logger.info("Measuring latency...")
    device = next(wrapped_model.model.parameters()).device
    dummy_input = make_dummy_input(wrapped_model.model, tokenizer, device)
    latency = profiler.measure_latency(wrapped_model.model, dummy_input)
    logger.info(f"Latency: {latency:.2f} ms")

    logger.info("Measuring L2 cache hit rate...")
    # Move dummy_input to CPU for cache measurement if it's on CUDA
    cache_dummy_input = {k: v.cpu() for k, v in dummy_input.items()}
    cache_result = profiler.measure_cache_hits(wrapped_model.model, cache_dummy_input)
    cache_hits = cache_result.get("lts__t_sector_hit_rate.pct", 0.0) if cache_result else 0.0
    
    logger.info(f"Modularity: {modularity:.4f}")

    return {
        metric_name: metric_value,
        "latency_ms": latency,
        "l2_cache_hit_rate_pct": cache_hits,
        "modularity": modularity,
    }


def run_dense(cfg: DictConfig):
    """Runs the dense baseline experiment."""
    set_random_seeds(cfg.seed)
    model, tokenizer, data_loader, eval_dataset = get_model_and_data(cfg)
    if torch.cuda.is_available():
        model.cuda()
    
    wrapped_model = ModelWrapper(model)
    profiler = LatencyProfiler()

    logger.info("--- Dense Baseline Performance ---")
    metrics = _measure_and_collect_metrics(
        wrapped_model, tokenizer, data_loader, eval_dataset, profiler, cfg, modularity=0.0
    )
    save_results(cfg, "dense", metrics)


def run_sparsity_only(cfg: DictConfig):
    """Runs the sparsity-only experiment."""
    set_random_seeds(cfg.seed)
    model, tokenizer, data_loader, eval_dataset = get_model_and_data(cfg)
    if torch.cuda.is_available():
        model.cuda()
    
    wrapped_model = ModelWrapper(model)
    profiler = LatencyProfiler()

    logger.info("--- Applying HDS Sparsification ---")
    apply_hds(wrapped_model, data_loader, cfg.model.hds)
    
    logger.info("--- Measuring Performance for Sparsity-Only ---")
    metrics = _measure_and_collect_metrics(
        wrapped_model, tokenizer, data_loader, eval_dataset, profiler, cfg, modularity=0.0
    )
    save_results(cfg, "sparsity_only", metrics)


IASP_DISPATCH = {
    "mamba": run_iasp_on_mamba,
    "bert": run_iasp_on_bert,
}

def detect_family(model: nn.Module) -> str:
    """Robustly detects model family by inspecting class hierarchy."""
    for cls in getmro(model.__class__):
        name = cls.__name__.lower()
        if "mamba" in name:
            return "mamba"
        if "bert" in name:
            return "bert"
    return "unknown"

def _run_iasp(model, data_loader, cfg) -> tuple[list[int], float]:
    """Helper function to run the correct IASP function and return permutation and modularity."""
    model_type = detect_family(model)
    iasp_config = cfg.model.iasp

    iasp_runner = IASP_DISPATCH.get(model_type)
    if iasp_runner:
        logger.info(f"Running IASP for auto-detected '{model_type}' model...")
        # Pass the entire model-specific IASP config
        return iasp_runner(model, data_loader, iasp_config)
    else:
        raise NotImplementedError(f"IASP is not implemented for model type: {model_type}")


def run_permute_only(cfg: DictConfig):
    """Runs the permutation-only experiment."""
    set_random_seeds(cfg.seed)
    model, tokenizer, data_loader, eval_dataset = get_model_and_data(cfg)
    if torch.cuda.is_available():
        model.cuda()
    
    wrapped_model = ModelWrapper(model)
    profiler = LatencyProfiler()
    
    logger.info("--- Applying IASP Permutation ---")

    # --- Pre-IASP weight integrity check ---
    lm_head_weight_before = None
    if hasattr(wrapped_model.model, "lm_head") and wrapped_model.model.lm_head is not None:
        lm_head_weight_before = wrapped_model.model.lm_head.weight.detach().clone()
        logger.info("Saved lm_head weights for integrity check.")

    permutation, modularity_score = _run_iasp(wrapped_model.model, data_loader, cfg)

    # --- Post-IASP weight integrity check ---
    if lm_head_weight_before is not None:
        lm_head_weight_after = wrapped_model.model.lm_head.weight.detach()
        if not torch.equal(lm_head_weight_before, lm_head_weight_after):
            logger.critical("CRITICAL WARNING: The 'lm_head' weights were modified by the IASP process. This is unintended and likely the cause of downstream errors.")
        else:
            logger.info("Integrity check passed: lm_head weights were not modified.")
    
    logger.info("--- Measuring Performance for Permutation-Only ---")
    metrics = _measure_and_collect_metrics(
        wrapped_model, tokenizer, data_loader, eval_dataset, profiler, cfg, modularity=modularity_score
    )
    save_results(cfg, "permute_only", metrics)


def run_linear_pipeline(cfg: DictConfig):
    """Runs the linear pipeline (IASP -> HDS) experiment."""
    set_random_seeds(cfg.seed)
    model, tokenizer, data_loader, eval_dataset = get_model_and_data(cfg)
    if torch.cuda.is_available():
        model.cuda()

    wrapped_model = ModelWrapper(model)
    profiler = LatencyProfiler()

    logger.info("--- Step 1: IASP Permutation ---")
    permutation, modularity_score = _run_iasp(wrapped_model.model, data_loader, cfg)
    
    logger.info("--- Step 2: HDS Sparsification ---")
    apply_hds(wrapped_model, data_loader, cfg.model.hds)
    
    logger.info("--- Measuring Performance for Linear Pipeline ---")
    metrics = _measure_and_collect_metrics(
        wrapped_model, tokenizer, data_loader, eval_dataset, profiler, cfg, modularity=modularity_score
    )
    save_results(cfg, "linear_pipeline", metrics)


def run_iterative(cfg: DictConfig):
    """Runs the iterative co-design experiment."""
    set_random_seeds(cfg.seed)
    model, tokenizer, data_loader, eval_dataset = get_model_and_data(cfg)
    if torch.cuda.is_available():
        model.cuda()

    wrapped_model = ModelWrapper(model)
    profiler = LatencyProfiler()
    num_iterations = cfg.method_configs.iterative.iterations
    
    modularity_score = 0.0
    for i in range(num_iterations):
        logger.info(f"\n--- Iteration {i+1}/{num_iterations} ---")
        
        logger.info(f"--- Iteration {i+1}, Step 1: HDS Sparsification ---")
        apply_hds(wrapped_model, data_loader, cfg.model.hds)
        
        logger.info(f"--- Iteration {i+1}, Step 2: IASP Permutation ---")
        permutation, current_modularity = _run_iasp(wrapped_model.model, data_loader, cfg)
        if i == num_iterations - 1:
            modularity_score = current_modularity
        
    logger.info("--- Final Performance Measurement for Iterative ---")
    metrics = _measure_and_collect_metrics(
        wrapped_model, tokenizer, data_loader, eval_dataset, profiler, cfg, modularity=modularity_score
    )
    save_results(cfg, "iterative", metrics)


def run_bidirectional_iterative(cfg: DictConfig):
    """Runs the fully bidirectional iterative co-design experiment."""
    set_random_seeds(cfg.seed)
    model, tokenizer, data_loader, eval_dataset = get_model_and_data(cfg)
    if torch.cuda.is_available():
        model.cuda()

    wrapped_model = ModelWrapper(model)
    profiler = LatencyProfiler()
    num_iterations = cfg.method_configs.iterative.iterations
    
    # Determine the actual dimension being permuted by IASP
    model_type = detect_family(model)
    target_dim = 0
    if 'mamba' in model_type:
        try:
            # Safely get d_inner from the model's architecture
            first_mixer = model.backbone.layers[0].mixer
            target_dim = first_mixer.out_proj.in_features
        except (AttributeError, IndexError):
            # Fallback for different Mamba versions
            target_dim = getattr(model.config, 'd_inner', model.config.hidden_size * getattr(model.config, 'expand', 2))
    else: # BERT
        try:
            # Safely get d_ffn from the model's architecture
            first_ffn = model.bert.encoder.layer[0].intermediate.dense
            target_dim = first_ffn.out_features
        except (AttributeError, IndexError):
             # Fallback for different BERT versions
            target_dim = model.config.intermediate_size

    logger.info(f"Bidirectional iterative co-design targeting dimension: {target_dim}")
    
    modularity_score = 0.0
    permutation = list(range(target_dim)) # Start with identity permutation of the correct size

    for i in range(num_iterations):
        logger.info(f"\n--- Bidirectional Iteration {i+1}/{num_iterations} ---")
        
        logger.info(f"--- Iteration {i+1}, Step 1: Layout-Aware HDS Sparsification ---")
        # Ensure the permutation tensor is on the correct device for the model
        perm_tensor = torch.as_tensor(permutation, device=wrapped_model.device)
        apply_layout_aware_hds_finetuning(
            wrapped_model, data_loader, cfg.model, perm_tensor
        )
        
        logger.info(f"--- Iteration {i+1}, Step 2: IASP Permutation ---")
        # _run_iasp will return a new permutation of size target_dim
        permutation, current_modularity = _run_iasp(wrapped_model.model, data_loader, cfg)
        if i == num_iterations - 1:
            modularity_score = current_modularity
        
    logger.info("--- Final Performance Measurement for Bidirectional Iterative ---")
    metrics = _measure_and_collect_metrics(
        wrapped_model, tokenizer, data_loader, eval_dataset, profiler, cfg, modularity=modularity_score
    )
    save_results(cfg, "bidirectional_iterative", metrics)

def run_cleanup_if_configured(cfg: DictConfig, dry_run: bool = False):
    """Runs the cleanup utility if enabled in the config."""
    if not cfg.get("cleanup_old_outputs", False):
        return

    # Safeguard: Do not clean if the current working directory is the project root.
    # This can happen with `hydra.run.dir=.`
    try:
        if Path.cwd().samefile(hydra.utils.get_original_cwd()):
            logger.error("Refusing to clean output directories from the project root. "
                         "Please use Hydra's default output directory structure.")
            return
    except (FileNotFoundError, TypeError):
        # get_original_cwd can fail or return None in some contexts.
        pass

    if dry_run:
        print("\n🧹 Cleanup is enabled. Old output directories would be removed.")
    else:
        print("\n🧹 Running cleanup of old experiment outputs...")
        cleanup_old_runs(dry_run=False)
        print("✅ Cleanup completed successfully")

def print_dry_run_plan(cfg: DictConfig):
    """Prints the execution plan for a dry run."""
    print("\n" + "="*50)
    print(" " * 18 + "DRY RUN PLAN")
    print("="*50)
    print(f"Project: {cfg.project_name}")
    print(f"Experiment: {cfg.get('experiment_name', 'default_experiment')}")
    print(f"Method: {cfg.method}")
    print("-" * 50)
    print("Configuration:")
    print(f"  Model: {cfg.model.name}")
    print(f"  Dataset: {cfg.dataset.name}")
    print(f"  Seed: {cfg.seed}")
    print(f"  Log to W&B: {cfg.wandb.log}")
    print("-" * 50)
    
    method = cfg.method
    if method == "dense":
        print("📋 Dense Baseline Plan:")
        print("  1. Load model and dataset")
        print("  2. Measure performance (perplexity, latency, cache hits, modularity)")
        print("  3. Save results")
    elif method == "sparsity_only":
        print("📋 Sparsity-Only (HDS) Plan:")
        print("  1. Load model and dataset")
        print("  2. Apply HDS to induce sparsity")
        print("  3. Measure performance")
        print("  4. Save results")
    elif method == "permute_only":
        print("📋 Permutation-Only (IASP) Plan:")
        print("  1. Load model and dataset")
        print("  2. Find and apply optimal permutation (IASP)")
        print("  3. Measure performance")
        print("  4. Save results")
    elif method == "linear_pipeline":
        print("📋 Linear Pipeline (IASP -> HDS) Plan:")
        print("  1. Load model and dataset")
        print("  2. Find initial optimal permutation (IASP)")
        print("  3. Apply HDS to induce sparsity on permuted model")
        print("  4. Measure final performance")
        print("  5. Save results")
    elif method == "iterative":
        print("📋 Iterative Co-Design Plan:")
        print(f"  Number of iterations: {cfg.method_configs.iterative.iterations}")
        print("  For each iteration:")
        print("    - Apply HDS to induce sparsity")
        print("    - Find and apply optimal permutation (IASP)")
        print("  After all iterations:")
        print("    - Measure final performance")
        print("    - Save results")
    
    print("-" * 50)
    run_cleanup_if_configured(cfg, dry_run=True)
    print("="*50)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.get("dry_run", False):
        print_dry_run_plan(cfg)
        return

    # Initialize W&B
    if cfg.wandb.log:
        initialize_wandb(cfg)

    # Run cleanup if enabled
    run_cleanup_if_configured(cfg)
    
    METHOD_RUNNERS = {
        "dense": run_dense,
        "sparsity_only": run_sparsity_only,
        "permute_only": run_permute_only,
        "linear_pipeline": run_linear_pipeline,
        "iterative": run_iterative,
        "bidirectional_iterative": run_bidirectional_iterative,
    }

    runner = METHOD_RUNNERS.get(cfg.method)
    if runner:
        runner(cfg)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

if __name__ == "__main__":
    main()
