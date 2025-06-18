import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import hydra
from omegaconf import DictConfig
import json
from tqdm import tqdm

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.profiler import LatencyProfiler
from src.co_design.iasp import find_optimal_permutation, get_activation_correlation
from src.models.wrapper import ModelWrapper

def generate_random_permutations(model_size: int, n_samples: int = 20) -> list:
    """Generate multiple random permutations for comparison."""
    permutations = []
    for _ in range(n_samples):
        perm = list(range(model_size))
        np.random.shuffle(perm)
        permutations.append(perm)
    return permutations

def measure_permutation_latency(model, permutation, dummy_input_dict, profiler):
    """Measure latency for a specific permutation."""
    wrapped_model = ModelWrapper(model)
    if torch.cuda.is_available():
        wrapped_model.cuda()
    
    # Apply permutation
    wrapped_model.permute_model_weights(permutation)
    
    # Measure latency
    latency = profiler.measure_latency(wrapped_model, dummy_input_dict)
    return latency

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Generate Figure 1: Latency comparison of random vs optimized permutations.
    
    This script:
    1. Loads a Mamba model
    2. Generates multiple random permutations
    3. Finds the optimal permutation using IASP
    4. Measures latency for all permutations
    5. Creates a visualization comparing random vs optimized latency
    """
    
    print("🎯 Generating Figure 1: Random vs Optimized Permutation Latency")
    
    # Load model and data
    print("📥 Loading model and data...")
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset(cfg.dataset.name, cfg.dataset.get('config'))
    sample_dataset = dataset['validation'].select(range(100))  # Small sample for speed
    
    def tokenize_function(examples):
        return tokenizer(examples[cfg.dataset.text_column], padding="max_length", 
                        truncation=True, max_length=512)
    
    tokenized_dataset = sample_dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    data_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=4)
    
    if torch.cuda.is_available():
        model.cuda()
    
    # Setup profiler and dummy input
    profiler = LatencyProfiler()
    dummy_input = torch.randint(0, cfg.model.vocab_size, (1, 512))
    dummy_input_dict = {"input_ids": dummy_input}
    if torch.cuda.is_available():
        dummy_input_dict = {k: v.cuda() for k, v in dummy_input_dict.items()}
    
    model_size = model.config.hidden_size
    
    # 1. Generate random permutations and measure their latency
    print("🎲 Generating and testing random permutations...")
    random_permutations = generate_random_permutations(model_size, n_samples=20)
    random_latencies = []
    
    # Measure baseline (identity) latency first
    identity_permutation = list(range(model_size))
    baseline_latency = measure_permutation_latency(
        model, identity_permutation, dummy_input_dict, profiler
    )
    print(f"   📊 Baseline (identity) latency: {baseline_latency:.2f} ms")
    
    for i, perm in enumerate(tqdm(random_permutations, desc="Testing random permutations")):
        latency = measure_permutation_latency(model, perm, dummy_input_dict, profiler)
        random_latencies.append(latency)
    
    # 2. Find optimal permutation using IASP
    print("🔍 Finding optimal permutation using IASP...")
    wrapped_model = ModelWrapper(model)
    if torch.cuda.is_available():
        wrapped_model.cuda()
    
    optimal_permutation = find_optimal_permutation(
        model=wrapped_model,
        data_loader=data_loader,
        target_layer_name=cfg.model.iasp.target_layer_name,
        cluster_size_range=tuple(cfg.model.iasp.cluster_size_range)
    )
    
    # 3. Measure optimal permutation latency
    print("⚡ Measuring optimized permutation latency...")
    optimal_latency = measure_permutation_latency(
        model, optimal_permutation, dummy_input_dict, profiler
    )
    print(f"   📊 Optimized permutation latency: {optimal_latency:.2f} ms")
    
    # 4. Create visualization
    print("📊 Creating Figure 1 visualization...")
    
    # Calculate improvement
    improvement_vs_baseline = ((baseline_latency - optimal_latency) / baseline_latency) * 100
    improvement_vs_random_avg = ((np.mean(random_latencies) - optimal_latency) / np.mean(random_latencies)) * 100
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot random permutation latencies as scatter points
    x_random = ['Random'] * len(random_latencies)
    plt.scatter(x_random, random_latencies, alpha=0.6, color='lightcoral', s=60, label='Random Permutations')
    
    # Plot mean of random permutations
    plt.bar(['Random (Mean)'], [np.mean(random_latencies)], 
            color='red', alpha=0.7, width=0.4, label='Random (Average)')
    
    # Plot baseline
    plt.bar(['Baseline (Identity)'], [baseline_latency], 
            color='orange', alpha=0.7, width=0.4, label='Baseline (Identity)')
    
    # Plot optimized permutation
    plt.bar(['Optimized (IASP)'], [optimal_latency], 
            color='green', alpha=0.7, width=0.4, label='Optimized (IASP)')
    
    # Add improvement annotations
    plt.annotate(f'-{improvement_vs_baseline:.1f}%', 
                xy=('Optimized (IASP)', optimal_latency), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.ylabel('Latency (ms)', fontsize=14)
    plt.xlabel('Permutation Type', fontsize=14)
    plt.title('Figure 1: Latency of Mamba Layer with Random vs. Optimized Memory Permutations\n'
              f'(Optimized layout reduces latency by {improvement_vs_baseline:.1f}% vs baseline)', 
              fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Improve styling
    sns.set_style("whitegrid")
    plt.tight_layout()
    
    # Save the figure
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    figure_path = output_dir / "figure1_mamba_latency_scan_vs_perm.pdf"
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_dir / "figure1_mamba_latency_scan_vs_perm.png", dpi=300, bbox_inches='tight')
    
    print(f"✅ Figure 1 saved to: {figure_path}")
    
    # Save raw data
    results = {
        "baseline_latency": baseline_latency,
        "random_latencies": random_latencies,
        "random_mean": np.mean(random_latencies),
        "random_std": np.std(random_latencies),
        "optimal_latency": optimal_latency,
        "improvement_vs_baseline_percent": improvement_vs_baseline,
        "improvement_vs_random_avg_percent": improvement_vs_random_avg,
        "model_name": cfg.model.name,
        "model_size": model_size
    }
    
    with open(output_dir / "figure1_data.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print("📊 Summary:")
    print(f"   • Baseline latency: {baseline_latency:.2f} ms")
    print(f"   • Random permutations average: {np.mean(random_latencies):.2f} ± {np.std(random_latencies):.2f} ms")
    print(f"   • Optimized latency: {optimal_latency:.2f} ms")
    print(f"   • Improvement vs baseline: {improvement_vs_baseline:.1f}%")
    print(f"   • Improvement vs random average: {improvement_vs_random_avg:.1f}%")
    
    plt.show()

if __name__ == '__main__':
    main() 