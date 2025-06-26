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
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)
from datasets import load_dataset
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.logging import initialize_wandb
import wandb

from utils.evaluation import calculate_task_metric
from utils.profiler import LatencyProfiler
from utils.cleanup import cleanup_old_runs
from co_design.iasp import find_optimal_permutation, collect_activations
from co_design.modularity import calculate_modularity
from models.wrapper import ModelWrapper
from co_design.hds import apply_hds
import logging

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
        return tokenizer(examples[text_column], truncation=True, max_length=512)

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


def _measure_and_collect_metrics(wrapped_model, tokenizer, data_loader, eval_dataset, profiler, cfg, permutation=None, partition=None):
    """Helper function to measure and collect all relevant metrics."""
    logger.info("Calculating evaluation metric...")
    task_metric = calculate_task_metric(
        wrapped_model.model, tokenizer, data_loader, cfg.model.task
    )
    metric_name, metric_value = list(task_metric.items())[0]
    logger.info(f"{metric_name.title()}: {metric_value:.4f}")

    logger.info("Measuring latency...")
    dummy_input = {"input_ids": torch.randint(0, cfg.model.vocab_size, (1, 512)).cuda()}
    latency = profiler.measure_latency(wrapped_model.model, dummy_input)
    logger.info(f"Latency: {latency:.2f} ms")

    logger.info("Measuring L2 cache hit rate...")
    cache_result = profiler.measure_cache_hits(wrapped_model.model, dummy_input)
    cache_hits = cache_result.get("lts__t_sector_hit_rate.pct", 0.0) if cache_result else 0.0
    
    logger.info("Calculating modularity...")
    iasp_cfg = cfg.model.iasp
    target_layer_spec = iasp_cfg.get("target_layer_names", iasp_cfg.get("target_layer_name"))
    
    # For dense, we don't have a permutation, so we can't calculate modularity in a meaningful way here.
    if partition is None:
        modularity = 0.0
    else:
        activations = collect_activations(wrapped_model.model, tokenizer, eval_dataset, target_layer_spec, num_samples=iasp_cfg.get("num_samples", 128))
        correlation_matrix = np.corrcoef(activations.T.numpy())
        correlation_matrix = np.nan_to_num(correlation_matrix)
        modularity = calculate_modularity(correlation_matrix, partition)
    
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
        wrapped_model, tokenizer, data_loader, eval_dataset, profiler, cfg, partition=[list(range(model.config.hidden_size))]
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
    partition, _ = apply_hds(wrapped_model, data_loader, cfg.model.hds)
    
    logger.info("--- Measuring Performance for Sparsity-Only ---")
    metrics = _measure_and_collect_metrics(
        wrapped_model, tokenizer, data_loader, eval_dataset, profiler, cfg, partition=partition
    )
    save_results(cfg, "sparsity_only", metrics)


def run_permute_only(cfg: DictConfig):
    """Runs the permutation-only experiment."""
    set_random_seeds(cfg.seed)
    model, tokenizer, data_loader, eval_dataset = get_model_and_data(cfg)
    if torch.cuda.is_available():
        model.cuda()
    
    wrapped_model = ModelWrapper(model)
    profiler = LatencyProfiler()
    iasp_cfg = cfg.model.iasp
    
    logger.info("--- Applying IASP Permutation ---")
    target_layer_spec = iasp_cfg.get("target_layer_names", iasp_cfg.get("target_layer_name"))
    permutation = find_optimal_permutation(
        model=wrapped_model.model,
        tokenizer=tokenizer,
        dataset=eval_dataset,
        target_layer_names=target_layer_spec,
        cluster_size_range=tuple(iasp_cfg.cluster_size_range),
        num_samples=iasp_cfg.get("num_samples", 128)
    )
    wrapped_model.permute_model(permutation)
    
    # For modularity calculation, we need a partition
    nodes_per_cluster = model.config.hidden_size // (model.config.hidden_size // 32)
    partition = [permutation[i:i+nodes_per_cluster] for i in range(0, len(permutation), nodes_per_cluster)]
    
    logger.info("--- Measuring Performance for Permutation-Only ---")
    metrics = _measure_and_collect_metrics(
        wrapped_model, tokenizer, data_loader, eval_dataset, profiler, cfg, permutation=permutation, partition=partition
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
    iasp_cfg = cfg.model.iasp

    logger.info("--- Step 1: IASP Permutation ---")
    target_layer_spec = iasp_cfg.get("target_layer_names", iasp_cfg.get("target_layer_name"))
    permutation = find_optimal_permutation(
        model=wrapped_model.model,
        tokenizer=tokenizer,
        dataset=eval_dataset,
        target_layer_names=target_layer_spec,
        cluster_size_range=tuple(iasp_cfg.cluster_size_range),
        num_samples=iasp_cfg.get("num_samples", 128)
    )
    wrapped_model.permute_model(permutation)
    
    logger.info("--- Step 2: HDS Sparsification ---")
    partition, _ = apply_hds(wrapped_model, data_loader, cfg.model.hds)
    
    logger.info("--- Measuring Performance for Linear Pipeline ---")
    metrics = _measure_and_collect_metrics(
        wrapped_model, tokenizer, data_loader, eval_dataset, profiler, cfg, permutation=permutation, partition=partition
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
    iasp_cfg = cfg.model.iasp
    num_iterations = cfg.method_configs.iterative.iterations
    
    # Initialize permutation as identity
    permutation = list(range(model.config.hidden_size))
    
    for i in range(num_iterations):
        logger.info(f"\n--- Iteration {i+1}/{num_iterations} ---")
        
        logger.info(f"--- Iteration {i+1}, Step 1: HDS Sparsification ---")
        partition, _ = apply_hds(wrapped_model, data_loader, cfg.model.hds)
        
        logger.info(f"--- Iteration {i+1}, Step 2: IASP Permutation ---")
        target_layer_spec = iasp_cfg.get("target_layer_names", iasp_cfg.get("target_layer_name"))
        permutation = find_optimal_permutation(
            model=wrapped_model.model,
            tokenizer=tokenizer,
            dataset=eval_dataset,
            target_layer_names=target_layer_spec,
            cluster_size_range=tuple(iasp_cfg.cluster_size_range),
            num_samples=iasp_cfg.get("num_samples", 128)
        )
        wrapped_model.permute_model(permutation)

    # For modularity calculation, we need a partition
    nodes_per_cluster = model.config.hidden_size // (model.config.hidden_size // 32)
    final_partition = [permutation[i:i+nodes_per_cluster] for i in range(0, len(permutation), nodes_per_cluster)]

    logger.info("--- Final Performance Measurement for Iterative ---")
    metrics = _measure_and_collect_metrics(
        wrapped_model, tokenizer, data_loader, eval_dataset, profiler, cfg, permutation=permutation, partition=final_partition
    )
    save_results(cfg, "iterative", metrics)

def run_cleanup_if_configured(cfg: DictConfig, dry_run: bool = False):
    """Runs the cleanup utility if enabled in the config."""
    if cfg.get("cleanup_old_outputs", False):
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
    print(f"Experiment: {cfg.experiment_name}")
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
    }

    runner = METHOD_RUNNERS.get(cfg.method)
    if runner:
        runner(cfg)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

if __name__ == "__main__":
    main()
