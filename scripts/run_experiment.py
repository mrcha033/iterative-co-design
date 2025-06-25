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
    python scripts/run_experiment.py model=mamba_3b dataset=wikitext103 method=iterative
    python scripts/run_experiment.py model=bert_base dataset=sst2 method=dense dry_run=true
"""

from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import load_dataset
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.logging import initialize_wandb
import wandb

from utils.evaluation import calculate_task_metric
from utils.profiler import LatencyProfiler
from utils.cleanup import cleanup_old_runs
from co_design.iasp import find_optimal_permutation, get_activation_correlation
from co_design.modularity import calculate_modularity
from models.wrapper import ModelWrapper
from co_design.hds import apply_hds
import logging

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Base class for running co-design experiments with common functionality."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize the experiment runner with configuration."""
        self.cfg = cfg
        self.profiler = LatencyProfiler()
        self.model = None
        self.tokenizer = None
        self.data_loader = None
        self.wrapped_model = None
        self.d_model = None
        
        # Setup logging
        self._setup_logging()
        
        # Setup reproducibility
        self._set_random_seeds(self.cfg.seed)
        
    def _setup_logging(self):
        """Setup experiment logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.info(f"Starting experiment with method: {self.cfg.method}")
        
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducible experiments."""
        logger.info(f"Setting random seeds to {seed} for reproducible results")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def load_model_and_data(self):
        """Load model, tokenizer, and dataset based on configuration."""
        self.model, self.tokenizer, self.data_loader = get_model_and_data(self.cfg)
        self.d_model = self.model.config.hidden_size
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model.cuda()
            logger.info("Model moved to CUDA")
    
    def save_results(self, method: str, metrics: dict):
        """Save experiment results."""
        save_results(self.cfg, method, metrics)
        
        if wandb.run:
            wandb.finish()
    
    def measure_performance(self, model, include_modularity=True, 
                          permutation=None, partition=None):
        """Measure common performance metrics."""
        # Task-specific metric
        logger.info(f"Calculating evaluation metric for {self.cfg.model.task}...")
        task_metric = calculate_task_metric(model, self.tokenizer, 
                                          self.data_loader, self.cfg.model.task)
        metric_name = list(task_metric.keys())[0]
        metric_value = task_metric[metric_name]
        logger.info(f"{metric_name.title()}: {metric_value:.4f}")
        
        # Latency measurement
        logger.info("Measuring latency...")
        dummy_input = torch.randint(0, self.cfg.model.vocab_size, (1, 512))
        dummy_input_dict = {"input_ids": dummy_input}
        latency = self.profiler.measure_latency(model, dummy_input_dict)
        logger.info(f"Latency: {latency:.2f} ms")
        
        # Cache hit rate
        logger.info("Measuring L2 cache hit rate...")
        cache_result = self.profiler.measure_cache_hits(model, dummy_input_dict)
        cache_hits = cache_result.get("l2_tex_hit_rate.pct", 0.0) if cache_result else 0.0
        
        metrics = {
            metric_name: metric_value,
            "latency_ms": latency,
            "l2_cache_hit_rate_pct": cache_hits,
        }
        
        # Modularity calculation if requested
        if include_modularity:
            logger.info("Calculating modularity...")
            if permutation is None and partition is None:
                # Dense baseline case
                modularity = 0.0
                logger.info(f"Modularity: {modularity:.4f} (by definition for dense model)")
            else:
                correlation_matrix = get_activation_correlation(
                    model, self.data_loader, self.cfg.model.iasp.target_layer_name
                )
                
                if partition is None:
                    # Calculate partition from permutation
                    partition = self._calculate_partition_from_permutation(permutation)
                
                modularity = calculate_modularity(correlation_matrix, partition)
                logger.info(f"Modularity: {modularity:.4f}")
            
            metrics["modularity"] = modularity
        
        return metrics
    
    def _calculate_partition_from_permutation(self, permutation):
        """Calculate partition from permutation for modularity calculation."""
        cluster_size_range = self.cfg.model.iasp.cluster_size_range
        optimal_cluster_size = min(
            cluster_size_range[1], max(cluster_size_range[0], self.d_model // 8)
        )
        n_clusters = self.d_model // optimal_cluster_size
        nodes_per_cluster = self.d_model // n_clusters if n_clusters > 0 else self.d_model
        
        partition = [
            permutation[i : i + nodes_per_cluster]
            for i in range(0, self.d_model, nodes_per_cluster)
        ]
        return partition


def set_random_seeds(seed: int):
    """Set random seeds for reproducible experiments."""
    print(f"Setting random seeds to {seed} for reproducible results")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Additional settings for reproducibility
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
        dataset = load_dataset(cfg.dataset.name, cfg.dataset.get("config"))
    except FileNotFoundError as e:
        print(f"❌ Dataset '{cfg.dataset.name}' not found locally.")
        print("💡 Please run: bash data/download_datasets.sh")
        print(
            f"   Or download manually with: python -c \"from datasets import load_dataset; load_dataset('{cfg.dataset.name}'{', ' + repr(cfg.dataset.get('config')) if cfg.dataset.get('config') else ''})\""
        )
        raise e

    val_dataset = dataset["validation"].select(range(cfg.dataset.sample_size))

    def tokenize_function(examples):
        return tokenizer(
            examples[cfg.dataset.text_column],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    tokenized_dataset = val_dataset.map(tokenize_function, batched=True)
    # Keep the text column for perplexity calculation
    format_columns = [
        "input_ids",
        "attention_mask",
    ]
    # Add label column if it exists
    if "label" in tokenized_dataset.column_names:
        format_columns.append("label")
    elif "labels" in tokenized_dataset.column_names:
        format_columns.append("labels")

    # Add text column if it exists for perplexity calculation
    if cfg.dataset.text_column in tokenized_dataset.column_names:
        format_columns.append(cfg.dataset.text_column)

    tokenized_dataset.set_format(type="torch", columns=format_columns)

    if "label" in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    data_loader = DataLoader(tokenized_dataset, batch_size=cfg.dataset.batch_size)

    return model, tokenizer, data_loader


def save_results(cfg: DictConfig, method: str, metrics: dict):
    # Hydra automatically creates a directory for each run, so we just save there.
    output_dir = Path.cwd()
    file_path = output_dir / f"{method}_metrics.json"

    print(f"Saving results to {file_path}")
    # Save metrics
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    # Save the config used for this run
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    # Log metrics to wandb
    if wandb.run:
        wandb.log(metrics)


def run_dense(cfg: DictConfig):
    """Runs the dense baseline experiment."""
    print("\n--- Running Method: (1) Dense Baseline ---")
    
    # Use ExperimentRunner for common functionality
    runner = ExperimentRunner(cfg)
    runner.load_model_and_data()
    
    # Measure performance metrics
    metrics = runner.measure_performance(runner.model, include_modularity=True)
    
    # Save results
    runner.save_results("dense", metrics)


def run_sparsity_only(cfg: DictConfig):
    """Runs the sparsity-only (HDS) baseline experiment."""
    print("\n--- Running Method: (2) Sparsity-Only (HDS) ---")
    
    # Use ExperimentRunner for common functionality
    runner = ExperimentRunner(cfg)
    runner.load_model_and_data()
    
    # Wrap model and apply HDS
    wrapped_model = ModelWrapper(runner.model)
    if torch.cuda.is_available():
        wrapped_model.cuda()
    
    wrapped_model.model = apply_hds(
        wrapped_model.model, runner.data_loader, OmegaConf.to_container(cfg, resolve=True)
    )
    
    # Calculate partition for modularity (identity permutation)
    identity_permutation = list(range(runner.d_model))
    cluster_step = max(runner.d_model // 32, 1)
    nodes_per_cluster = runner.d_model // cluster_step
    partition = [
        identity_permutation[i : i + nodes_per_cluster]
        for i in range(0, runner.d_model, nodes_per_cluster)
    ]
    
    # Measure performance
    metrics = runner.measure_performance(wrapped_model, include_modularity=True,
                                       permutation=identity_permutation, 
                                       partition=partition)
    
    # Save results
    runner.save_results("sparsity_only", metrics)


def run_permute_only(cfg: DictConfig):
    """Runs the permutation-only baseline experiment."""
    print("\n--- Running Method: (3) Permutation-Only (IASP) ---")
    
    # Use ExperimentRunner for common functionality
    runner = ExperimentRunner(cfg)
    runner.load_model_and_data()
    
    # Wrap model
    wrapped_model = ModelWrapper(runner.model)
    if torch.cuda.is_available():
        wrapped_model.cuda()
    
    # Find and apply optimal permutation
    permutation = find_optimal_permutation(
        model=wrapped_model,
        data_loader=runner.data_loader,
        target_layer_name=cfg.model.iasp.target_layer_name,
        cluster_size_range=tuple(cfg.model.iasp.cluster_size_range),
    )
    
    wrapped_model.permute_model_weights(permutation)
    
    # Measure performance with modularity
    metrics = runner.measure_performance(wrapped_model, include_modularity=True,
                                       permutation=permutation)
    
    # Save results
    runner.save_results("permute_only", metrics)


def run_linear_pipeline(cfg: DictConfig):
    """Runs the linear pipeline (IASP-then-HDS) experiment."""
    print("\n--- Running Method: (4) Linear Pipeline (IASP-then-HDS) ---")

    model, tokenizer, data_loader = get_model_and_data(cfg)
    wrapped_model = ModelWrapper(model)
    d_model = model.config.hidden_size
    profiler = LatencyProfiler()

    if torch.cuda.is_available():
        wrapped_model.cuda()

    initial_permutation = find_optimal_permutation(
        model=wrapped_model,
        data_loader=data_loader,
        target_layer_name=cfg.model.iasp.target_layer_name,
        cluster_size_range=tuple(cfg.model.iasp.cluster_size_range),
    )
    wrapped_model.permute_model_weights(initial_permutation)

    wrapped_model.model = apply_hds(
        wrapped_model.model, data_loader, OmegaConf.to_container(cfg, resolve=True)
    )

    task_metric = calculate_task_metric(
        wrapped_model, tokenizer, data_loader, cfg.model.task
    )
    metric_name = list(task_metric.keys())[0]
    metric_value = task_metric[metric_name]

    dummy_input = torch.randint(0, cfg.model.vocab_size, (1, 512))
    dummy_dict = {"input_ids": dummy_input}
    latency = profiler.measure_latency(wrapped_model, dummy_dict)
    cache_result = profiler.measure_cache_hits(wrapped_model, dummy_dict)
    cache_hits = cache_result.get("l2_tex_hit_rate.pct", 0.0) if cache_result else 0.0

    correlation_matrix = get_activation_correlation(
        wrapped_model, data_loader, cfg.model.iasp.target_layer_name
    )
    # Calculate cluster size from IASP configuration
    cluster_size_range = cfg.model.iasp.cluster_size_range
    optimal_cluster_size = min(
        cluster_size_range[1], max(cluster_size_range[0], d_model // 8)
    )
    n_clusters = d_model // optimal_cluster_size
    nodes_per_cluster = d_model // n_clusters if n_clusters > 0 else d_model
    partition = [
        initial_permutation[i : i + nodes_per_cluster]
        for i in range(0, d_model, nodes_per_cluster)
    ]
    modularity = calculate_modularity(correlation_matrix, partition)

    metrics = {
        metric_name: metric_value,
        "latency_ms": latency,
        "l2_cache_hit_rate_pct": cache_hits,
        "modularity": modularity,
    }
    save_results(cfg, "linear_pipeline", metrics)

    if wandb.run:
        wandb.finish()


def run_iterative(cfg: DictConfig):
    """Runs the full iterative co-design experiment."""
    print("\n--- Running Method: (5) Iterative Co-Design (Ours) ---")

    model, tokenizer, data_loader = get_model_and_data(cfg)
    wrapped_model = ModelWrapper(model)
    d_model = model.config.hidden_size
    profiler = LatencyProfiler()

    if torch.cuda.is_available():
        wrapped_model.cuda()

    iteration_metrics = {"latency": [], "modularity": [], "l2_cache_hit_rate": []}

    for i in range(cfg.num_iterations):
        print(f"\n--- Starting Co-Design Iteration {i + 1}/{cfg.num_iterations} ---")
        wrapped_model.model = apply_hds(
            wrapped_model.model, data_loader, OmegaConf.to_container(cfg, resolve=True)
        )

        permutation = find_optimal_permutation(
            model=wrapped_model,
            data_loader=data_loader,
            target_layer_name=cfg.model.iasp.target_layer_name,
            cluster_size_range=tuple(cfg.model.iasp.cluster_size_range),
        )
        wrapped_model.permute_model_weights(permutation)

        dummy_input = torch.randint(0, cfg.model.vocab_size, (1, 512))
        dummy_dict = {"input_ids": dummy_input}
        lat = profiler.measure_latency(wrapped_model, dummy_dict)
        iteration_metrics["latency"].append(lat)

        cache_result = profiler.measure_cache_hits(wrapped_model, dummy_dict)
        cache_value = (
            cache_result.get("l2_tex_hit_rate.pct", 0.0) if cache_result else 0.0
        )
        iteration_metrics["l2_cache_hit_rate"].append(cache_value)

        corr_matrix = get_activation_correlation(
            wrapped_model, data_loader, cfg.model.iasp.target_layer_name
        )
        # Calculate cluster size from IASP configuration
        cluster_size_range = cfg.model.iasp.cluster_size_range
        optimal_cluster_size = min(
            cluster_size_range[1], max(cluster_size_range[0], d_model // 8)
        )
        n_clusters = d_model // optimal_cluster_size
        nodes_per_cluster = d_model // n_clusters if n_clusters > 0 else d_model
        part = [
            permutation[j : j + nodes_per_cluster]
            for j in range(0, d_model, nodes_per_cluster)
        ]
        mod = calculate_modularity(corr_matrix, part)
        iteration_metrics["modularity"].append(mod)

    task_metric = calculate_task_metric(
        wrapped_model, tokenizer, data_loader, cfg.model.task
    )
    metric_name = list(task_metric.keys())[0]
    metric_value = task_metric[metric_name]

    final_metrics = {
        metric_name: metric_value,
        "latency_ms": iteration_metrics["latency"][-1],
        "l2_cache_hit_rate_pct": iteration_metrics["l2_cache_hit_rate"][-1],
        "modularity": iteration_metrics["modularity"][-1],
        "num_iterations": cfg.num_iterations,
        "iteration_metrics": iteration_metrics,
    }
    save_results(cfg, "iterative", final_metrics)

    if wandb.run:
        wandb.finish()


def run_cleanup_if_configured(cfg: DictConfig, dry_run: bool = False):
    """Run cleanup of old runs if configured, respecting dry_run flag."""
    if hasattr(cfg, "cleanup") and cfg.cleanup:
        try:
            print("🧹 Running cleanup of old experiment outputs...")
            cleanup_old_runs(
                base_dirs=cfg.cleanup.base_dirs,
                max_age_days=cfg.cleanup.max_age_days,
                dry_run=dry_run,
            )
            if not dry_run:
                print("✅ Cleanup completed successfully")
            else:
                print("✅ Cleanup dry run completed (no files deleted)")
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")
            print("Continuing with experiment...")


def print_dry_run_plan(cfg: DictConfig):
    """Print the planned operations for a dry run without executing them."""
    method = cfg.method
    print(f"\n🔍 DRY RUN MODE - Showing planned operations for method: {method}")
    print("=" * 60)

    if method == "dense":
        print("📋 Dense Baseline Plan:")
        print("  1. Load model and dataset")
        print("  2. Measure baseline task metric (perplexity/accuracy)")
        print("  3. Measure baseline latency")
        print("  4. Measure L2 cache hit rate")
        print("  5. Record modularity = 0.0 (identity permutation)")
        print("  6. Save results")

    elif method == "sparsity_only":
        print("📋 Sparsity-Only (HDS) Plan:")
        print("  1. Load model and dataset")
        print("  2. Apply HDS to target layers (fine-tune sparsity masks)")
        print("  3. Measure task metric on sparse model")
        print("  4. Measure latency on sparse model")
        print("  5. Measure L2 cache hit rate")
        print("  6. Calculate modularity with identity permutation")
        print("  7. Save results")

    elif method == "permute_only":
        print("📋 Permutation-Only (IASP) Plan:")
        print("  1. Load model and dataset")
        print("  2. Collect activation correlations")
        print("  3. Find optimal permutation using spectral clustering")
        print("  4. Apply permutation to model weights")
        print("  5. Measure task metric on permuted model")
        print("  6. Measure latency and cache hit rate")
        print("  7. Calculate modularity score")
        print("  8. Save results")

    elif method == "linear_pipeline":
        print("📋 Linear Pipeline (IASP-then-HDS) Plan:")
        print("  1. Load model and dataset")
        print("  2. Find initial optimal permutation (IASP)")
        print("  3. Apply permutation to model weights")
        print("  4. Apply HDS to permuted model (fine-tune sparsity)")
        print("  5. Measure task metric on permuted + sparse model")
        print("  6. Measure latency and cache hit rate")
        print("  7. Calculate modularity score")
        print("  8. Save results")

    elif method == "iterative":
        print("📋 Iterative Co-Design Plan:")
        print("  1. Load model and dataset")
        print(f"  2. Iterate {cfg.num_iterations} times:")
        for i in range(cfg.num_iterations):
            print(f"     Iteration {i + 1}:")
            print("       - Apply HDS (fine-tune sparsity masks)")
            print("       - Find optimal permutation for current state")
            print("       - Apply permutation to weights")
            print("       - Measure iteration metrics (latency, modularity, cache)")
        print("  3. Calculate final task metric")
        print("  4. Save comprehensive results with iteration history")

    else:
        print(f"❌ Unknown method: {method}")
        return

    print("\n📊 Expected outputs:")
    print(f"  • Results saved to: outputs/<timestamp>/{method}_metrics.json")
    print("  • Config saved to: outputs/<timestamp>/config.yaml")
    if cfg.get("wandb", {}).get("mode") != "disabled":
        print(f"  • Metrics logged to W&B project: {cfg.project_name}")

    print("\n⏱️  Estimated runtime:")
    if method == "dense":
        print("  • ~2-5 minutes (baseline measurements only)")
    elif method in ["sparsity_only", "permute_only"]:
        print("  • ~10-30 minutes (includes optimization step)")
    elif method == "linear_pipeline":
        print("  • ~20-45 minutes (sequential optimization)")
    elif method == "iterative":
        iter_time = cfg.num_iterations * 15
        print(
            f"  • ~{iter_time}-{iter_time * 2} minutes ({cfg.num_iterations} iterations)"
        )

    print("\n✅ Dry run complete - no actual computation performed")
    print("💡 Remove 'dry_run=true' to execute the full experiment")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Initialize random seeds for reproducible experiments
    set_random_seeds(cfg.seed)

    # The 'method' is now chosen from the command line, e.g., `python script.py method=dense`
    method = OmegaConf.select(cfg, "method", default="dense")
    dry_run = OmegaConf.select(cfg, "dry_run", default=False)

    # Add method to config for logging
    OmegaConf.set_struct(cfg, False)
    cfg.method = method

    # Run cleanup before starting experiment (once per run)
    run_cleanup_if_configured(cfg, dry_run=dry_run)

    # Handle dry run mode
    if dry_run:
        print_dry_run_plan(cfg)
        return

    initialize_wandb(cfg)

    if method == "dense":
        run_dense(cfg)
    elif method == "sparsity_only":
        run_sparsity_only(cfg)
    elif method == "permute_only":
        run_permute_only(cfg)
    elif method == "linear_pipeline":
        run_linear_pipeline(cfg)
    elif method == "iterative":
        run_iterative(cfg)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    main()
