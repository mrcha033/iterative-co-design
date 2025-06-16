import argparse
import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.quantization

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.utils.profiler import measure_latency
from src.co_design.iasp import find_optimal_permutation
from src.models.wrapper import ModelWrapper

def get_model_and_data(config: dict):
    """Loads model, tokenizer, and a data sample based on the config."""
    print(f"Loading model: {config['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    print(f"Loading dataset: {config['dataset_name']}")
    dataset = load_dataset(config['dataset_name'], config.get('dataset_config'))
    sample_dataset = dataset['validation'].select(range(config.get('sample_size', 16)))
    
    def tokenize_function(examples):
        return tokenizer(examples[config['text_column']], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = sample_dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    data_loader = DataLoader(tokenized_dataset, batch_size=config.get('batch_size', 4))
    
    return model, tokenizer, data_loader

def apply_ptq(model: torch.nn.Module) -> torch.nn.Module:
    """Applies Post-Training Dynamic Quantization to the model."""
    print(">>> Applying Post-Training Quantization (Dynamic)...")
    model_cpu = model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def save_quant_results(config: dict, method: str, metrics: dict):
    """Saves quantization experiment results."""
    results_dir = Path("results") / config['model_name'].replace("/", "_") / "quantization"
    results_dir.mkdir(exist_ok=True, parents=True)
    file_path = results_dir / f"{method}_metrics.json"
    
    print(f"Saving quantization results to {file_path}")
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def run_quant_then_permute(config, model, tokenizer, data_loader):
    """Strategy 1: Quantize the model, then find and apply the best permutation."""
    print("\n--- Running Strategy: (1) Quant-then-Permute ---")
    
    # 1. Apply PTQ
    quantized_model = apply_ptq(model)
    wrapped_model = ModelWrapper(quantized_model)
    
    # 2. Find optimal permutation on the quantized model
    print("Finding optimal permutation on INT8 model...")
    permutation = find_optimal_permutation(
        model=wrapped_model,
        data_loader=data_loader,
        target_layer_name=config['iasp']['target_layer_name'],
        n_clusters=config['iasp']['n_clusters']
    )
    
    # 3. Apply permutation
    wrapped_model.permute_model_weights(permutation)
    
    # 4. Measure Latency
    print("Measuring latency...")
    dummy_input = torch.randint(0, config['vocab_size'], (1, 512))
    latency = measure_latency(wrapped_model, dummy_input, on_gpu=False)
    print(f"Final Latency: {latency:.2f} ms")
    
    # 5. Save Results
    save_quant_results(config, 'quant_then_permute', {"latency": latency})

def run_permute_then_quant(config, model, tokenizer, data_loader):
    """Strategy 2: Find permutation on FP32, apply it, then quantize."""
    print("\n--- Running Strategy: (2) Permute-then-Quant ---")
    
    # 1. Find optimal permutation on the FP32 model
    print("Finding optimal permutation on FP32 model...")
    wrapped_model = ModelWrapper(model)
    if torch.cuda.is_available():
        wrapped_model.cuda()

    permutation = find_optimal_permutation(
        model=wrapped_model,
        data_loader=data_loader,
        target_layer_name=config['iasp']['target_layer_name'],
        n_clusters=config['iasp']['n_clusters']
    )
    
    # 2. Apply permutation to FP32 model
    wrapped_model.permute_model_weights(permutation)
    
    # 3. Apply PTQ
    final_model = apply_ptq(wrapped_model.model)
    
    # 4. Measure Latency
    print("Measuring latency...")
    dummy_input = torch.randint(0, config['vocab_size'], (1, 512))
    latency = measure_latency(final_model, dummy_input, on_gpu=False)
    print(f"Final Latency: {latency:.2f} ms")
    
    # 5. Save Results
    save_quant_results(config, 'permute_then_quant', {"latency": latency})

def run_permute_quant_repermute(config, model, tokenizer, data_loader):
    """Strategy 3: Permute on FP32, quantize, then re-permute on INT8."""
    print("\n--- Running Strategy: (3) Permute-Quant-RePermute ---")
    
    # 1. Find initial permutation on the FP32 model
    print("Finding initial permutation on FP32 model (Round 1)...")
    wrapped_model = ModelWrapper(model)
    if torch.cuda.is_available():
        wrapped_model.cuda()

    perm1 = find_optimal_permutation(
        model=wrapped_model,
        data_loader=data_loader,
        target_layer_name=config['iasp']['target_layer_name'],
        n_clusters=config['iasp']['n_clusters']
    )
    
    # 2. Apply initial permutation
    wrapped_model.permute_model_weights(perm1)
    
    # 3. Apply PTQ
    quantized_permuted_model = apply_ptq(wrapped_model.model)
    
    # 4. Find second permutation on the INT8 model
    print("Finding new permutation on INT8 model (Round 2)...")
    wrapped_quant_model = ModelWrapper(quantized_permuted_model)
    perm2 = find_optimal_permutation(
        model=wrapped_quant_model,
        data_loader=data_loader,
        target_layer_name=config['iasp']['target_layer_name'],
        n_clusters=config['iasp']['n_clusters']
    )
    
    # 5. Apply re-permutation
    wrapped_quant_model.permute_model_weights(perm2)
    
    # 6. Measure Latency
    print("Measuring latency...")
    dummy_input = torch.randint(0, config['vocab_size'], (1, 512))
    latency = measure_latency(wrapped_quant_model, dummy_input, on_gpu=False)
    print(f"Final Latency: {latency:.2f} ms")
    
    # 7. Save Results
    save_quant_results(config, 'permute_quant_repermute', {"latency": latency})


def main():
    parser = argparse.ArgumentParser(description="Run quantization co-design experiments.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument(
        '--method', 
        type=str, 
        required=True,
        choices=['quant_then_permute', 'permute_then_quant', 'permute_quant_repermute'], 
        help='The quantization strategy to run.'
    )
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Setup model and data
    model, tokenizer, data_loader = get_model_and_data(config)
    
    # Run the selected method
    if args.method == 'quant_then_permute':
        run_quant_then_permute(config, model, tokenizer, data_loader)
    elif args.method == 'permute_then_quant':
        run_permute_then_quant(config, model, tokenizer, data_loader)
    elif args.method == 'permute_quant_repermute':
        run_permute_quant_repermute(config, model, tokenizer, data_loader)

if __name__ == '__main__':
    main() 