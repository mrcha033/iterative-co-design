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
# Environment variable setup must be the very first thing to ensure all
# imported modules see them correctly.
# ------------------------------------------------------------------
import os
# Set a default service wait time to prevent rate-limiting on fast sweeps
os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
# Extra safety for rust-based tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TOKENIZERS_FAST_CORE_PARALLELISM"] = "false"

# ------------------------------------------------------------------
# Python path setup
# ------------------------------------------------------------------
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
# Use project root for consistent 'src.' prefixed imports
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import random
import numpy as np
import torch
import torch.nn as nn
import fnmatch
from torch.utils.data import DataLoader
from collections import deque
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)
from datasets import load_dataset
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.logging import initialize_wandb
import wandb
import math
import os
from tqdm.auto import tqdm

# Removed insecure 'eval' resolver. Dynamic paths should be handled in code.
# For example, use hydra.utils.to_absolute_path() for file paths.

from src.utils.evaluation import calculate_task_metric
from src.utils.profiler import LatencyProfiler
from src.utils.cleanup import cleanup_old_runs
from src.utils.input import make_dummy_input
from src.co_design.iasp import run_iasp_on_mamba, run_iasp_on_bert
from src.models.wrapper import ModelWrapper
from src.co_design.hds import apply_hds
from src.co_design.layout_aware import apply_layout_aware_hds_finetuning
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


class StreamingSlidingWindowDataset(torch.utils.data.IterableDataset):
    """
    An IterableDataset that tokenizes and creates sliding windows from a streaming
    Hugging Face dataset on the fly, designed for memory efficiency.
    """
    def __init__(self, hf_dataset, tokenizer, seq_len, stride, buffer_size=65536, max_samples=None):
        super().__init__()
        if stride <= 0 or stride > seq_len:
            raise ValueError(f"Stride must be in (0, seq_len], but got stride={stride} and seq_len={seq_len}")
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.max_samples = max_samples
        # Buffer size should be larger than seq_len
        self.buffer_size = max(buffer_size, seq_len * 2)
    
    def __len__(self):
        """
        Return estimated dataset length for progress bars.
        For IterableDataset, this is just an estimate to help tqdm.
        """
        # If max_samples is set, use that as the length
        if self.max_samples is not None:
            return self.max_samples
        
        # Otherwise use a reasonable default that doesn't trigger extreme ETA estimates
        return min(10000, max(1000, self.buffer_size))
        
    def __iter__(self):
        # Correctly seed the random number generator for each worker for reproducibility.
        worker_info = torch.utils.data.get_worker_info()
        seed = worker_info.seed if worker_info else torch.initial_seed()
        rng = random.Random(seed & 0xFFFFFFFF)

        # Use a deque with maxlen for a memory-efficient, automatically-managed buffer.
        buffer = deque(maxlen=self.buffer_size)
        
        samples_yielded = 0
        dataset_iterator = iter(self.hf_dataset)

        while self.max_samples is None or samples_yielded < self.max_samples:
            try:
                sample = next(dataset_iterator)
            except StopIteration:
                if self.max_samples is not None:
                    # If the dataset is exhausted but we haven't met the sample count,
                    # restart the iterator to loop over the data again.
                    dataset_iterator = iter(self.hf_dataset)
                    continue
                else:
                    # If no max_samples is set, just stop.
                    break

            if "text" not in sample or not sample["text"]:
                continue

            # Truncate long documents to buffer_size and sample a random starting point
            # to avoid biasing towards the end of long texts.
            token_ids = self.tokenizer(
                sample["text"],
                truncation=True,
                max_length=self.buffer_size,
                add_special_tokens=False
            )["input_ids"]

            # Skip documents that are shorter than the desired sequence length
            # to prevent them from creating biased, repetitive samples.
            if len(token_ids) < self.seq_len:
                continue

            if len(token_ids) > self.seq_len:
                start_index = rng.randint(0, len(token_ids) - self.seq_len)
                buffer.extend(token_ids[start_index:])
            else:
                buffer.extend(token_ids)

            # Yield all complete windows from the current buffer
            while len(buffer) >= self.seq_len:
                if self.max_samples is not None and samples_yielded >= self.max_samples:
                    return # Stop iteration once max_samples is reached

                window_ids = [buffer[i] for i in range(self.seq_len)]
                yield {"input_ids": torch.tensor(window_ids, dtype=torch.long)}
                samples_yielded += 1

                # Slide the window forward by popping `stride` elements from the left
                for _ in range(self.stride):
                    if not buffer: break
                    buffer.popleft()


def lm_collate(batch):
    """Custom collate function that creates labels for language modeling."""
    # Use default collator to stack tensors and handle padding
    collated_batch = default_data_collator(batch)
    # Create a detached clone for labels. This is the safest practice, as it
    # prevents any potential in-place modifications to input_ids from affecting
    # the labels during evaluation (e.g., in model.generate).
    collated_batch["labels"] = collated_batch["input_ids"].clone().detach()
    return collated_batch


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
        # Add a pad token if one doesn't exist, which is common for some base models.
        # Use the EOS token as the pad token.
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        # The model's embedding matrix needs to be resized to reflect the new token.
        model.resize_token_embeddings(len(tokenizer))
        # Ensure the model config uses the correct pad_token_id for loss calculation
        model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Loading dataset: {cfg.dataset.name}")

    if cfg.model.task == "language_modeling":
        # Use streaming to handle large datasets without loading all data into RAM.
        hf_iterable_dataset = load_dataset(
            cfg.dataset.name, cfg.dataset.get("subset"), streaming=True
        )[cfg.dataset.get("eval_split", "validation")]
        
        seq_len = cfg.dataset.get("seq_len", 512)
        stride = cfg.dataset.get("stride", seq_len // 2)
        
        # Calculate reasonable buffer and batch sizes for performance
        batch_size = cfg.dataset.get("batch_size", 8)
        # Ensure batch size is at least 4 for reasonable GPU utilization
        batch_size = max(4, batch_size)
        # Set a larger buffer for higher throughput
        buffer_size = cfg.dataset.get("buffer_size", max(65536, seq_len * 4))

        streaming_dataset = StreamingSlidingWindowDataset(
            hf_dataset=hf_iterable_dataset,
            tokenizer=tokenizer,
            seq_len=seq_len,
            stride=stride,
            buffer_size=buffer_size,
            max_samples=cfg.iasp.get("max_samples", None)
        )
        
        # DataLoader tuning for IterableDataset
        # For streaming datasets, we can parallelize tokenization
        # with a worker that pre-fetches and tokenizes data
        dl_kwargs = dict(
            batch_size=batch_size,
            collate_fn=lm_collate,
            num_workers=0,  # Must be 0 for this implementation
            pin_memory=True,  # Use pinned memory for faster CPU->GPU transfer
        )
        
        data_loader = DataLoader(streaming_dataset, **dl_kwargs)
        logger.info(f"Created language modeling DataLoader with batch_size={batch_size}")
        # In streaming mode, there's no separate text-only dataset to return.
        return model, tokenizer, data_loader, None

    # --- Original path for non-LM tasks (e.g., sequence classification) ---
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
    
    # DataLoader tuning
    num_cpu = max(0, os.cpu_count() // 2)
    dl_kwargs = dict(
        batch_size=cfg.dataset.batch_size,
        collate_fn=default_data_collator,
        num_workers=num_cpu,
        pin_memory=False,
        persistent_workers=False,
    )
    if num_cpu > 0:
        dl_kwargs["prefetch_factor"] = 4

    data_loader = DataLoader(tokenized_dataset, **dl_kwargs)

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
    """A helper function to run evaluation and profiling and collect all metrics."""
    model = wrapped_model.model
    model.eval()

    logger.info("Calculating evaluation metric...")
    task_metric = calculate_task_metric(
        model,
        data_loader,
        metric=cfg.dataset.metric_type,
        max_steps=cfg.dataset.get("max_eval_steps") or 1000  # Safe default
    )
    logger.info(f"{cfg.dataset.metric_type.capitalize()}: {list(task_metric.values())[0]:.4f}")

    logger.info("Measuring latency and hardware performance...")
    device = next(model.parameters()).device
    dummy_input = make_dummy_input(model, tokenizer, device)
    
    try:
        hw_metrics = profiler.profile_all_metrics(model, dummy_input)
    except Exception as e:
        logger.error(f"Profiler failed: {e}. Hardware metrics will be omitted.", exc_info=True)
        hw_metrics = {}
    
    # Log all collected hardware metrics
    if hw_metrics:
        logger.info("--- Hardware Performance Metrics ---")
        for k, v in hw_metrics.items():
            if isinstance(v, float):
                logger.info(f"- {k}: {v:.4f}")
            else:
                logger.info(f"- {k}: {v}")
        logger.info("------------------------------------")

    metrics = {
        **task_metric,
        "modularity": modularity,
    }
    # Add hardware metrics to the final dictionary, handling the case where profiling might fail
    if hw_metrics:
        metrics.update(hw_metrics)

    return metrics


def run_dense(cfg: DictConfig):
    """Runs the dense baseline experiment."""
    set_random_seeds(cfg.seed)
    model, tokenizer, data_loader, eval_dataset = get_model_and_data(cfg)
    
    # Device-agnostic model placement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    wrapped_model = ModelWrapper(model)
    profiler = LatencyProfiler()

    logger.info("--- Dense Baseline Performance ---")
    metrics = _measure_and_collect_metrics(
        wrapped_model, tokenizer, data_loader, eval_dataset, profiler, cfg, modularity=0.0
    )
    save_results(cfg, "dense", metrics)


def run_sparsity_only(cfg: DictConfig):
    """Runs only the HDS sparsity algorithm."""
    print("--- Running Sparsity-Only Experiment ---")
    set_random_seeds(cfg.seed)
    model, tokenizer, data_loader, eval_dataset = get_model_and_data(cfg)

    # Device-agnostic model placement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    wrapped_model = ModelWrapper(model)
    profiler = LatencyProfiler()

    logger.info("--- Applying HDS Sparsification ---")
    apply_hds(wrapped_model, data_loader, cfg.model.hds)
    
    logger.info("--- Measuring Performance for Sparsity-Only ---")
    metrics = _measure_and_collect_metrics(
        wrapped_model, tokenizer, data_loader, eval_dataset, profiler, cfg, modularity=0.0
    )
    wrapped_model.log_sparsity()
    save_results(cfg, "sparsity_only", metrics)


def detect_family(model: nn.Module) -> str:
    """Robustly detects model family by inspecting class hierarchy."""
    for cls in getmro(model.__class__):
        name = cls.__name__.lower()
        if any(k in name for k in ("mamba", "ssm")):
            return "mamba"
        if "bert" in name:
            return "bert"
    return "unknown"

def _run_iasp(model, data_loader, cfg) -> tuple[list[int], float]:
    """Helper to dispatch IASP to the correct model type."""
    family = detect_family(model)
    logger.info(f"Running IASP for model family: {family}")

    # Get the IASP config and ensure target_layers exists with a fallback
    iasp_cfg = cfg.iasp
    
    # Diagnostic logging to understand config structure at runtime
    logger.debug(f"IASP config structure: {OmegaConf.to_container(iasp_cfg, resolve=True)}")
    
    # Convert to structured config with model family information
    try:
        from src.utils.config import create_iasp_config
        
        # Convert to dict for safe manipulation
        config_dict = OmegaConf.to_container(iasp_cfg, resolve=True)
        
        # Add model family if not present
        if "model_family" not in config_dict:
            config_dict["model_family"] = family
            
        # Create structured config - target_layers will be set in __post_init__ if None
        iasp_cfg = create_iasp_config(config_dict)
        logger.info(f"Using target_layers: {iasp_cfg.target_layers}")
    except (ImportError, AttributeError) as e:
        # Fall back to original config if structured configs aren't available
        logger.warning(f"Couldn't use structured config: {e}")

    if family == "mamba":
        return run_iasp_on_mamba(model, data_loader, iasp_cfg)
    elif family == "bert":
        return run_iasp_on_bert(model, data_loader, iasp_cfg)
    else:
        logger.warning(f"IASP not implemented for model family '{family}'. Skipping.")
        return [], 0.0


def run_permute_only(cfg: DictConfig):
    """Runs the permutation-only experiment."""
    print("--- Running Permutation-Only Experiment ---")
    set_random_seeds(cfg.seed)
    model, tokenizer, data_loader, eval_dataset = get_model_and_data(cfg)
    
    # Device-agnostic model placement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    wrapped_model = ModelWrapper(model)
    profiler = LatencyProfiler()
    
    logger.info("--- Applying IASP Permutation ---")

    # --- Pre-IASP weight integrity check ---
    lm_head_weight_before = None
    if hasattr(wrapped_model.model, "lm_head") and wrapped_model.model.lm_head is not None:
        lm_head_weight_before = wrapped_model.model.lm_head.weight.detach().clone()
        logger.info("Saved lm_head weights for integrity check.")

    # --- Enhanced smoke test for functional equivalence and parameter stability ---
    logger.info("--- Running IASP smoke test ---")
    device = next(wrapped_model.model.parameters()).device
    dummy_input = make_dummy_input(wrapped_model.model, tokenizer, device)

    # Check LayerNorm scales before permutation
    ln_norms_before = {
        name: p.norm().item()
        for name, p in wrapped_model.model.named_parameters()
        if "layernorm" in name.lower() or "norm" in name.lower()
    }

    with torch.no_grad():
        logits_before = wrapped_model.model(**dummy_input).logits.clone()

    permutation, modularity_score = _run_iasp(wrapped_model.model, data_loader, cfg)

    with torch.no_grad():
        logits_after = wrapped_model.model(**dummy_input).logits
    
    # Check LayerNorm scales after permutation
    ln_norms_after = {
        name: p.norm().item()
        for name, p in wrapped_model.model.named_parameters()
        if "layernorm" in name.lower() or "norm" in name.lower()
    }

    # 1. Check logit equivalence
    logit_diff = (logits_before - logits_after).abs().max()
    assert logit_diff < 1e-4, f"IASP broke functional equivalence! Max logit diff: {logit_diff}"
    logger.info(f"✅ Logit equivalence test passed. Max difference: {logit_diff:.6f}")

    # 2. Check LayerNorm stability
    norm_diffs = {name: abs(ln_norms_before[name] - ln_norms_after[name]) for name in ln_norms_before}
    max_norm_diff = max(norm_diffs.values()) if norm_diffs else 0.0
    assert max_norm_diff < 1e-4, f"IASP permutation altered LayerNorm scales! Max norm diff: {max_norm_diff}"
    logger.info(f"✅ LayerNorm stability test passed. Max norm difference: {max_norm_diff:.6f}")
    # --- End of smoke test ---

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
    
    # Device-agnostic model placement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

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

    # Device-agnostic model placement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

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

    # Device-agnostic model placement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

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
    except (FileNotFoundError, TypeError, OSError):
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

def check_gpu_headroom(required_gb: float = 4.0) -> bool:
    """Checks if there is enough free GPU memory on ALL devices to proceed."""
    nv_available = False
    try:
        import pynvml as nv
        nv.nvmlInit()
        nv_available = True
    except (ImportError, nv.NVMLError) as e:
        logger.warning(f"pynvml not installed or failed to initialize ({e}). Falling back to torch-only memory check.")

    if not torch.cuda.is_available():
        return True # Continue if on CPU
    
    for i in range(torch.cuda.device_count()):
        free_gb = -1
        try:
            if nv_available:
                handle = nv.nvmlDeviceGetHandleByIndex(i)
                info = nv.nvmlDeviceGetMemoryInfo(handle)
                free_gb = info.free / (1024**3)
            else: # Fallback
                with torch.cuda.device(i):
                    free_bytes, _ = torch.cuda.mem_get_info()
                    free_gb = free_bytes / (1024**3)
            
            logger.info(f"GPU {i}: Available Memory: {free_gb:.2f} GB")
            if free_gb < required_gb:
                logger.error(f"GPU {i} memory headroom is below the required {required_gb} GB. Skipping experiment to prevent OOM.")
                Path.cwd().joinpath(f"skipped_low_gpu_{i}.sentinel").touch()
                if nv_available:
                    nv.nvmlShutdown()
                return False
        except (RuntimeError, nv.NVMLError) as e:
            logger.warning(f"Could not check memory for GPU {i}: {e}. This may happen with older drivers. Proceeding with caution.")
    
    if nv_available:
        nv.nvmlShutdown()
    return True

@hydra.main(config_name="config", version_base=None, config_path=os.getenv("HYDRA_CONFIG_PATH", "../configs"))
def main(cfg: DictConfig):
    # --- Initialize structured configs if available ---
    try:
        from src.utils.config import initialize_structured_configs
        initialize_structured_configs()
        logger.info("Initialized structured configuration system")
    except ImportError:
        logger.info("Structured configuration system not available")

    # --- Pre-run setup ---
    # Disable W&B if not explicitly enabled. Must be done before wandb is used.
    if not cfg.wandb.log:
        os.environ["WANDB_DISABLED"] = "true"

    if cfg.get("dry_run", False):
        print_dry_run_plan(cfg)
        return

    # --- Pre-run checks ---
    if not check_gpu_headroom(required_gb=cfg.get("min_gpu_headroom_gb", 4.0)):
        logger.warning("Gracefully exiting run due to insufficient GPU memory.")
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
