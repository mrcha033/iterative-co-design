from pathlib import Path
import json
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
from src.utils.logging import initialize_wandb
import wandb

from src.utils.evaluation import calculate_perplexity
from src.utils.profiler import LatencyProfiler
from src.co_design.iasp import find_optimal_permutation, get_activation_correlation
from src.co_design.modularity import calculate_modularity
from src.models.wrapper import ModelWrapper
from src.co_design.hds import apply_hds


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
    dataset = load_dataset(cfg.dataset.name, cfg.dataset.get("config"))

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
        "label" if "label" in tokenized_dataset.column_names else "labels",
    ]
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

    # --- 1. Setup ---
    model, tokenizer, data_loader = get_model_and_data(cfg)
    profiler = LatencyProfiler()

    if torch.cuda.is_available():
        model.cuda()

    # --- 2. Run Measurements ---
    print("Calculating perplexity...")
    perplexity = calculate_perplexity(model, tokenizer, data_loader)
    print(f"Perplexity: {perplexity:.4f}")

    print("Measuring latency...")
    dummy_input = torch.randint(0, cfg.model.vocab_size, (1, 512))
    dummy_input_dict = {"input_ids": dummy_input}
    latency = profiler.measure_latency(model, dummy_input_dict)
    print(f"Latency: {latency:.2f} ms")

    print("Measuring L2 cache hit rate...")
    cache_result = profiler.measure_cache_hits(model, dummy_input_dict)
    cache_hits = cache_result.get("l2_tex_hit_rate.pct", 0.0) if cache_result else 0.0

    print("Calculating modularity...")
    modularity = 0.0
    print(f"Modularity: {modularity:.4f} (by definition for dense model)")

    # --- 3. Save Results ---
    metrics = {
        "perplexity": perplexity,
        "latency_ms": latency,
        "l2_cache_hit_rate_pct": cache_hits,
        "modularity": modularity,
    }
    save_results(cfg, "dense", metrics)

    if wandb.run:
        wandb.finish()


def run_sparsity_only(cfg: DictConfig):
    """Runs the sparsity-only (HDS) baseline experiment."""
    print("\n--- Running Method: (2) Sparsity-Only (HDS) ---")

    model, tokenizer, data_loader = get_model_and_data(cfg)
    wrapped_model = ModelWrapper(model)
    d_model = model.config.hidden_size
    profiler = LatencyProfiler()

    if torch.cuda.is_available():
        wrapped_model.cuda()

    wrapped_model.model = apply_hds(
        wrapped_model.model, data_loader, OmegaConf.to_container(cfg, resolve=True)
    )

    perplexity = calculate_perplexity(wrapped_model, tokenizer, data_loader)
    dummy_input = torch.randint(0, cfg.model.vocab_size, (1, 512))
    dummy_dict = {"input_ids": dummy_input}
    latency = profiler.measure_latency(wrapped_model, dummy_dict)
    cache_result = profiler.measure_cache_hits(wrapped_model, dummy_dict)
    cache_hits = cache_result.get("l2_tex_hit_rate.pct", 0.0) if cache_result else 0.0

    correlation_matrix = get_activation_correlation(
        wrapped_model.model, data_loader, cfg.model.iasp.target_layer_name
    )
    identity_permutation = list(range(d_model))
    nodes_per_cluster = d_model // (d_model // 32)  # Heuristic
    partition = [
        identity_permutation[i : i + nodes_per_cluster]
        for i in range(0, d_model, nodes_per_cluster)
    ]
    modularity = calculate_modularity(correlation_matrix, partition)

    metrics = {
        "perplexity": perplexity,
        "latency_ms": latency,
        "l2_cache_hit_rate_pct": cache_hits,
        "modularity": modularity,
    }
    save_results(cfg, "sparsity_only", metrics)

    if wandb.run:
        wandb.finish()


def run_permute_only(cfg: DictConfig):
    """Runs the permutation-only baseline experiment."""
    print("\n--- Running Method: (3) Permutation-Only (IASP) ---")

    model, tokenizer, data_loader = get_model_and_data(cfg)
    wrapped_model = ModelWrapper(model)
    d_model = model.config.hidden_size
    profiler = LatencyProfiler()

    if torch.cuda.is_available():
        wrapped_model.cuda()

    permutation = find_optimal_permutation(
        model=wrapped_model,
        data_loader=data_loader,
        target_layer_name=cfg.model.iasp.target_layer_name,
        cluster_size_range=tuple(cfg.model.iasp.cluster_size_range),
    )

    wrapped_model.permute_model_weights(permutation)

    perplexity = calculate_perplexity(wrapped_model, tokenizer, data_loader)
    dummy_input = torch.randint(0, cfg.model.vocab_size, (1, 512))
    dummy_dict = {"input_ids": dummy_input}
    latency = profiler.measure_latency(wrapped_model, dummy_dict)
    cache_result = profiler.measure_cache_hits(wrapped_model, dummy_dict)
    cache_hits = cache_result.get("l2_tex_hit_rate.pct", 0.0) if cache_result else 0.0

    original_model, _, _ = get_model_and_data(cfg)
    correlation_matrix = get_activation_correlation(
        original_model, data_loader, cfg.model.iasp.target_layer_name
    )

    # Calculate cluster size from IASP configuration
    cluster_size_range = cfg.model.iasp.cluster_size_range
    optimal_cluster_size = min(
        cluster_size_range[1], max(cluster_size_range[0], d_model // 8)
    )
    n_clusters = d_model // optimal_cluster_size
    nodes_per_cluster = d_model // n_clusters if n_clusters > 0 else d_model
    partition = [
        permutation[i : i + nodes_per_cluster]
        for i in range(0, d_model, nodes_per_cluster)
    ]
    modularity = calculate_modularity(correlation_matrix, partition)

    metrics = {
        "perplexity": perplexity,
        "latency_ms": latency,
        "l2_cache_hit_rate_pct": cache_hits,
        "modularity": modularity,
    }
    save_results(cfg, "permute_only", metrics)

    if wandb.run:
        wandb.finish()


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

    perplexity = calculate_perplexity(wrapped_model, tokenizer, data_loader)
    dummy_input = torch.randint(0, cfg.model.vocab_size, (1, 512))
    dummy_dict = {"input_ids": dummy_input}
    latency = profiler.measure_latency(wrapped_model, dummy_dict)
    cache_result = profiler.measure_cache_hits(wrapped_model, dummy_dict)
    cache_hits = cache_result.get("l2_tex_hit_rate.pct", 0.0) if cache_result else 0.0

    correlation_matrix = get_activation_correlation(
        wrapped_model.model, data_loader, cfg.model.iasp.target_layer_name
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
        "perplexity": perplexity,
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
        print(f"\n--- Starting Co-Design Iteration {i+1}/{cfg.num_iterations} ---")
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
            wrapped_model.model, data_loader, cfg.model.iasp.target_layer_name
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

    final_metrics = {
        "perplexity": calculate_perplexity(wrapped_model, tokenizer, data_loader),
        "latency_ms": iteration_metrics["latency"][-1],
        "l2_cache_hit_rate_pct": iteration_metrics["l2_cache_hit_rate"][-1],
        "modularity": iteration_metrics["modularity"][-1],
        "num_iterations": cfg.num_iterations,
        "iteration_metrics": iteration_metrics,
    }
    save_results(cfg, "iterative", final_metrics)

    if wandb.run:
        wandb.finish()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # The 'method' is now chosen from the command line, e.g., `python script.py method=dense`
    method = OmegaConf.select(cfg, "method", default="dense")
    # Add method to config for logging
    OmegaConf.set_struct(cfg, False)
    cfg.method = method

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


if __name__ == "__main__":
    main()
