import argparse
import sys
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.utils.evaluation import calculate_perplexity, calculate_accuracy
from src.utils.profiler import measure_latency, measure_cache_hits
from src.co_design.iasp import find_optimal_permutation, get_activation_correlation
from src.co_design.modularity import calculate_modularity
from src.models.wrapper import ModelWrapper
from src.co_design.hds import apply_hds

def get_model_and_data(config: dict):
    """Loads model, tokenizer, and dataset based on the config."""
    print(f"Loading model: {config['model_name']}")
    if config['task'] == 'language_modeling':
        model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    elif config['task'] == 'sequence_classification':
        model = AutoModelForSequenceClassification.from_pretrained(config['model_name'])
    else:
        raise ValueError(f"Unknown task: {config['task']}")

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {config['dataset_name']}")
    dataset = load_dataset(config['dataset_name'], config.get('dataset_config'))
    
    # For simplicity, we'll use a small subset for correlation/perplexity calculation
    # In a real run, this would be the validation set.
    val_dataset = dataset['validation'].select(range(config.get('sample_size', 16)))
    
    def tokenize_function(examples):
        return tokenizer(examples[config['text_column']], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = val_dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label' if 'label' in tokenized_dataset.column_names else 'labels'])
    
    # Correctly handle column names for the data loader
    if 'label' in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')

    data_loader = DataLoader(tokenized_dataset, batch_size=config.get('batch_size', 4))
    
    return model, tokenizer, data_loader

def save_results(config: dict, method: str, metrics: dict):
    results_dir = Path("results") / config['model_name'].replace("/", "_")
    results_dir.mkdir(exist_ok=True, parents=True)
    file_path = results_dir / f"{method}_metrics.json"
    
    print(f"Saving results to {file_path}")
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def run_dense(config, args):
    """Runs the dense baseline experiment."""
    print("\n--- Running Method: (1) Dense Baseline ---")
    if args.dry_run:
        print("1. Loading model...")
        print("2. Measuring perplexity/accuracy.")
        print("3. Measuring latency.")
        print("4. Measuring L2 cache hit rate.")
        print("5. Measuring modularity.")
        print("6. Saving results to results/.../dense_metrics.json")
        return

    # --- 1. Setup ---
    model, tokenizer, data_loader = get_model_and_data(config)
    d_model = model.config.hidden_size
    temp_state_dict_path = "temp_dense_state_dict.pt"

    if torch.cuda.is_available():
        model.cuda()
    
    # --- 2. Run Measurements ---
    print("Calculating perplexity...")
    perplexity = calculate_perplexity(model, tokenizer, data_loader)
    print(f"Perplexity: {perplexity:.4f}")

    print("Measuring latency...")
    dummy_input = torch.randint(0, config['vocab_size'], (1, 512))
    latency = measure_latency(model, dummy_input)
    print(f"Latency: {latency:.2f} ms")

    print("Measuring L2 cache hit rate...")
    torch.save(model.state_dict(), temp_state_dict_path)
    cache_hits = measure_cache_hits(args.config, temp_state_dict_path)
    
    print("Calculating modularity...")
    # For a dense model, the permutation is just the identity, so we have one single community.
    # The modularity for a graph with one community is always 0.
    modularity = 0.0
    print(f"Modularity: {modularity:.4f} (by definition for dense model)")

    # --- 3. Save Results ---
    metrics = {
        "perplexity": perplexity,
        "latency_ms": latency,
        "l2_cache_hit_rate_pct": cache_hits,
        "modularity": modularity
    }
    save_results(config, 'dense', metrics)
    
    # Cleanup
    Path(temp_state_dict_path).unlink(missing_ok=True)

def run_sparsity_only(config, args):
    """Runs the sparsity-only (HDS) baseline experiment."""
    print("\n--- Running Method: (2) Sparsity-Only (HDS) ---")
    if args.dry_run:
        print("1. Loading model.")
        print("2. Applying HDS fine-tuning to learn 2:4 sparsity.")
        print("3. Measuring perplexity/accuracy on sparse model.")
        print("4. Measuring latency on sparse model.")
        print("5. Measuring L2 cache hit rate on sparse model.")
        print("6. Measuring modularity on sparse model.")
        print("7. Saving results to results/.../sparsity_only_metrics.json")
        return

    # --- 1. Setup ---
    model, tokenizer, data_loader = get_model_and_data(config)
    wrapped_model = ModelWrapper(model)
    d_model = model.config.hidden_size
    temp_state_dict_path = "temp_sparse_state_dict.pt"

    if torch.cuda.is_available():
        wrapped_model.cuda()

    # --- 2. Apply HDS ---
    wrapped_model.model = apply_hds(wrapped_model.model, data_loader, config)

    # --- 3. Run Measurements ---
    print("Calculating perplexity...")
    perplexity = calculate_perplexity(wrapped_model, tokenizer, data_loader)
    print(f"Perplexity: {perplexity:.4f}")

    print("Measuring latency...")
    dummy_input = torch.randint(0, config['vocab_size'], (1, 512))
    latency = measure_latency(wrapped_model, dummy_input)
    print(f"Latency: {latency:.2f} ms")

    print("Measuring L2 cache hit rate...")
    torch.save(wrapped_model.model.state_dict(), temp_state_dict_path)
    cache_hits = measure_cache_hits(args.config, temp_state_dict_path)

    print("Calculating modularity...")
    # Since no permutation is applied, modularity is calculated on the natural order
    # after the model's weights have been changed by HDS.
    correlation_matrix = get_activation_correlation(wrapped_model.model, data_loader, config['iasp']['target_layer_name'])
    
    # We assume a default partition (no reordering)
    nodes_per_cluster = d_model // config['iasp']['n_clusters']
    identity_permutation = list(range(d_model))
    partition = [
        identity_permutation[i:i + nodes_per_cluster] for i in range(0, d_model, nodes_per_cluster)
    ]
    modularity = calculate_modularity(correlation_matrix, partition)
    print(f"Modularity: {modularity:.4f}")

    # --- 4. Save Results ---
    metrics = {
        "perplexity": perplexity,
        "latency_ms": latency,
        "l2_cache_hit_rate_pct": cache_hits,
        "modularity": modularity
    }
    save_results(config, 'sparsity_only', metrics)
    
    # Cleanup
    Path(temp_state_dict_path).unlink(missing_ok=True)

def run_permute_only(config, args):
    """Runs the permutation-only baseline experiment."""
    print("\n--- Running Method: (3) Permutation-Only (IASP) ---")
    if args.dry_run:
        print("1. Loading model...")
        print("2. Running IASP to find optimal permutation.")
        print("3. Applying permutation to the model.")
        print("4. Measuring perplexity/accuracy on permuted model.")
        print("5. Measuring latency on permuted model.")
        print("6. Measuring L2 cache hit rate on permuted model.")
        print("7. Measuring modularity on permuted model.")
        print("8. Saving results to results/.../permute_only_metrics.json")
        return
        
    # --- 1. Setup ---
    model, tokenizer, data_loader = get_model_and_data(config)
    wrapped_model = ModelWrapper(model)
    d_model = model.config.hidden_size
    temp_state_dict_path = "temp_permuted_state_dict.pt"

    if torch.cuda.is_available():
        wrapped_model.cuda()

    # --- 2. Run IASP ---
    print("Finding optimal permutation with IASP...")
    permutation = find_optimal_permutation(
        model=wrapped_model,
        data_loader=data_loader,
        target_layer_name=config['iasp']['target_layer_name'],
        n_clusters=config['iasp']['n_clusters']
    )
    
    # --- 3. Apply Permutation ---
    print("Applying permutation to model weights...")
    wrapped_model.permute_model_weights(permutation)

    # --- 4. Run Measurements on Permuted Model ---
    print("Calculating perplexity...")
    perplexity = calculate_perplexity(wrapped_model, tokenizer, data_loader)
    print(f"Perplexity: {perplexity:.4f}")
    
    print("Measuring latency...")
    dummy_input = torch.randint(0, config['vocab_size'], (1, 512))
    latency = measure_latency(wrapped_model, dummy_input)
    print(f"Latency: {latency:.2f} ms")

    print("Measuring L2 cache hit rate...")
    torch.save(wrapped_model.model.state_dict(), temp_state_dict_path)
    cache_hits = measure_cache_hits(args.config, temp_state_dict_path)

    print("Calculating modularity...")
    # We need the correlation matrix from the original model state for a fair comparison
    # And the partition from the permutation we just found
    original_model, _, _ = get_model_and_data(config)
    correlation_matrix = get_activation_correlation(original_model, data_loader, config['iasp']['target_layer_name'])
    
    nodes_per_cluster = d_model // config['iasp']['n_clusters']
    partition = [
        permutation[i:i + nodes_per_cluster] for i in range(0, d_model, nodes_per_cluster)
    ]
    modularity = calculate_modularity(correlation_matrix, partition)
    print(f"Modularity: {modularity:.4f}")
    
    # --- 5. Save Results ---
    metrics = {
        "perplexity": perplexity,
        "latency_ms": latency,
        "l2_cache_hit_rate_pct": cache_hits,
        "modularity": modularity,
        "permutation": permutation
    }
    save_results(config, 'permute_only', metrics)
    
    # Cleanup
    Path(temp_state_dict_path).unlink(missing_ok=True)

def run_linear_pipeline(config, args):
    """Runs the linear pipeline (IASP-then-HDS) experiment."""
    print("\n--- Running Method: (4) Linear Pipeline (IASP-then-HDS) ---")
    if args.dry_run:
        print("1. Loading model.")
        print("2. Running IASP to find initial permutation.")
        print("3. Applying initial permutation to the model.")
        print("4. Applying HDS fine-tuning to the permuted model.")
        print("5. Measuring final metrics (perplexity, latency, cache, modularity).")
        print("6. Saving results to results/.../linear_pipeline_metrics.json")
        return

    # --- 1. Setup ---
    model, tokenizer, data_loader = get_model_and_data(config)
    wrapped_model = ModelWrapper(model)
    d_model = model.config.hidden_size
    temp_state_dict_path = "temp_linear_state_dict.pt"

    if torch.cuda.is_available():
        wrapped_model.cuda()

    # --- 2. Run IASP ---
    print("Finding initial permutation with IASP...")
    initial_permutation = find_optimal_permutation(
        model=wrapped_model,
        data_loader=data_loader,
        target_layer_name=config['iasp']['target_layer_name'],
        n_clusters=config['iasp']['n_clusters']
    )
    
    # --- 3. Apply Permutation ---
    print("Applying initial permutation to model weights...")
    wrapped_model.permute_model_weights(initial_permutation)

    # --- 4. Apply HDS ---
    wrapped_model.model = apply_hds(wrapped_model.model, data_loader, config)

    # --- 5. Run Measurements ---
    print("Calculating perplexity...")
    perplexity = calculate_perplexity(wrapped_model, tokenizer, data_loader)
    print(f"Perplexity: {perplexity:.4f}")
    
    print("Measuring latency...")
    dummy_input = torch.randint(0, config['vocab_size'], (1, 512))
    latency = measure_latency(wrapped_model, dummy_input)
    print(f"Latency: {latency:.2f} ms")

    print("Measuring L2 cache hit rate...")
    torch.save(wrapped_model.model.state_dict(), temp_state_dict_path)
    cache_hits = measure_cache_hits(args.config, temp_state_dict_path)

    print("Calculating modularity...")
    correlation_matrix = get_activation_correlation(wrapped_model.model, data_loader, config['iasp']['target_layer_name'])
    nodes_per_cluster = d_model // config['iasp']['n_clusters']
    partition = [
        initial_permutation[i:i + nodes_per_cluster] for i in range(0, d_model, nodes_per_cluster)
    ]
    modularity = calculate_modularity(correlation_matrix, partition)
    print(f"Modularity: {modularity:.4f}")

    # --- 6. Save Results ---
    metrics = {
        "perplexity": perplexity,
        "latency_ms": latency,
        "l2_cache_hit_rate_pct": cache_hits,
        "modularity": modularity
    }
    save_results(config, 'linear_pipeline', metrics)
    
    # Cleanup
    Path(temp_state_dict_path).unlink(missing_ok=True)

def run_iterative(config, args):
    """Runs the full iterative co-design experiment."""
    print("\n--- Running Method: (5) Iterative Co-Design (Ours) ---")
    if args.dry_run:
        print("1. Loading model.")
        print("--- Iteration 1 ---")
        print("2a. Apply HDS fine-tuning.")
        print("2b. Run IASP on the new state to find permutation π_1.")
        print("2c. Apply π_1 to the model.")
        print("--- Iteration 2 ---")
        print("3a. Apply HDS fine-tuning again.")
        print("3b. Run IASP on the new state to find permutation π_2.")
        print("3c. Apply π_2 to the model.")
        print("4. Measuring final metrics (perplexity, latency, cache, modularity).")
        print("5. Saving results to results/.../iterative_metrics.json")
        return

    # --- 1. Setup ---
    model, tokenizer, data_loader = get_model_and_data(config)
    wrapped_model = ModelWrapper(model)
    d_model = model.config.hidden_size
    temp_state_dict_path = "temp_iterative_state_dict.pt"
    num_iterations = config.get('num_iterations', 2)
    
    if torch.cuda.is_available():
        wrapped_model.cuda()

    final_permutation = list(range(d_model))
    iteration_metrics = {"latency": [], "modularity": [], "l2_cache_hit_rate": []}

    # --- 2. Iterative Loop ---
    for i in range(num_iterations):
        print(f"\n--- Starting Co-Design Iteration {i+1}/{num_iterations} ---")
        
        # a. Apply HDS
        wrapped_model.model = apply_hds(wrapped_model.model, data_loader, config)
        
        # b. Run IASP
        print(f"Finding optimal permutation for iteration {i+1}...")
        permutation = find_optimal_permutation(
            model=wrapped_model,
            data_loader=data_loader,
            target_layer_name=config['iasp']['target_layer_name'],
            n_clusters=config['iasp']['n_clusters']
        )
        
        # c. Apply Permutation
        print(f"Applying permutation for iteration {i+1}...")
        wrapped_model.permute_model_weights(permutation)
        final_permutation = permutation # Keep track of the last permutation

        # d. Measure metrics for this iteration
        print(f"Measuring metrics for iteration {i+1}...")
        dummy_input = torch.randint(0, config['vocab_size'], (1, 512))
        
        # Latency
        lat = measure_latency(wrapped_model, dummy_input)
        iteration_metrics["latency"].append(lat)
        print(f"  - Latency: {lat:.2f} ms")

        # L2 Cache Hits
        torch.save(wrapped_model.model.state_dict(), temp_state_dict_path)
        cache = measure_cache_hits(args.config, temp_state_dict_path)
        iteration_metrics["l2_cache_hit_rate"].append(cache)
        print(f"  - L2 Cache Hit Rate: {cache:.2f}%")

        # Modularity
        corr_matrix = get_activation_correlation(wrapped_model.model, data_loader, config['iasp']['target_layer_name'])
        nodes_per_cluster = d_model // config['iasp']['n_clusters']
        part = [permutation[j:j + nodes_per_cluster] for j in range(0, d_model, nodes_per_cluster)]
        mod = calculate_modularity(corr_matrix, part)
        iteration_metrics["modularity"].append(mod)
        print(f"  - Modularity: {mod:.4f}")

    # --- 3. Final Measurements ---
    print("\n--- Final Measurements after Iterative Co-Design ---")
    print("Calculating perplexity...")
    perplexity = calculate_perplexity(wrapped_model, tokenizer, data_loader)
    print(f"Perplexity: {perplexity:.4f}")
    
    print("Measuring latency...")
    dummy_input = torch.randint(0, config['vocab_size'], (1, 512))
    latency = measure_latency(wrapped_model, dummy_input)
    print(f"Latency: {latency:.2f} ms")

    print("Measuring L2 cache hit rate...")
    torch.save(wrapped_model.model.state_dict(), temp_state_dict_path)
    cache_hits = measure_cache_hits(args.config, temp_state_dict_path)

    print("Calculating modularity...")
    correlation_matrix = get_activation_correlation(wrapped_model.model, data_loader, config['iasp']['target_layer_name'])
    nodes_per_cluster = d_model // config['iasp']['n_clusters']
    partition = [
        final_permutation[i:i + nodes_per_cluster] for i in range(0, d_model, nodes_per_cluster)
    ]
    modularity = calculate_modularity(correlation_matrix, partition)
    print(f"Modularity: {modularity:.4f}")

    # --- 4. Save Results ---
    metrics = {
        "perplexity": perplexity,
        "latency_ms": latency,
        "l2_cache_hit_rate_pct": cache_hits,
        "modularity": modularity,
        "num_iterations": num_iterations,
        "iteration_metrics": iteration_metrics
    }
    save_results(config, 'iterative', metrics)

    # Cleanup
    Path(temp_state_dict_path).unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Run co-design experiments.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument(
        '--method', 
        type=str, 
        choices=['dense', 'sparsity_only', 'permute_only', 'linear_pipeline', 'iterative'], 
        help='The experimental method to run.'
    )
    parser.add_argument('--dry_run', action='store_true', help='Print the sequence of operations without executing them.')
    parser.add_argument('--_profile_for_ncu', type=str, help=argparse.SUPPRESS) # Internal flag for ncu

    args = parser.parse_args()
    config = load_config(args.config)

    # Handle the special ncu profiling mode
    if args._profile_for_ncu:
        print("--- NCU Profiling Mode ---")
        model, _, _ = get_model_and_data(config)
        model.load_state_dict(torch.load(args._profile_for_ncu))
        if torch.cuda.is_available():
            model.cuda()
            dummy_input = torch.randint(0, config.get('vocab_size', 30522), (1, 512), device='cuda')
            with torch.no_grad():
                _ = model(dummy_input)
            print("--- NCU Profiling Run Complete ---")
        else:
            print("CUDA not available. Cannot profile.")
        return # Exit after profiling run

    print(f"Loading configuration from: {args.config}")
    
    if args.method == 'dense':
        run_dense(config, args)
    elif args.method == 'sparsity_only':
        run_sparsity_only(config, args)
    elif args.method == 'permute_only':
        run_permute_only(config, args)
    elif args.method == 'linear_pipeline':
        run_linear_pipeline(config, args)
    elif args.method == 'iterative':
        run_iterative(config, args)

if __name__ == '__main__':
    main() 