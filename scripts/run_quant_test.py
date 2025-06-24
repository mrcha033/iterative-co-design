"""
Quantization co-design experiment runner.

This script tests the iterative co-design principle with quantization instead of
sparsity. It compares three strategies:
1. Quant-then-Permute: Apply quantization first, then find optimal permutation
2. Permute-then-Quant: Find permutation on FP32, then apply quantization
3. Iterative: Permute, quantize, then re-permute (tests iteration value)

This experiment demonstrates that the co-design principle applies beyond sparsity
to other model optimization techniques like quantization.

Usage:
    python scripts/run_quant_test.py model=mamba_3b dataset=wikitext103 method=permute_quant_repermute
"""

import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.quantization
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.profiler import LatencyProfiler
from co_design.iasp import find_optimal_permutation
from models.wrapper import ModelWrapper

profiler = LatencyProfiler()


def set_random_seeds(seed: int):
    """Set random seeds for reproducible quantization experiments."""
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
    """Loads model, tokenizer, and a data sample based on the config."""
    print(f"Loading model: {cfg.model.name}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

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
    sample_dataset = dataset["validation"].select(range(cfg.dataset.sample_size))

    def tokenize_function(examples):
        return tokenizer(
            examples[cfg.dataset.text_column],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    tokenized_dataset = sample_dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    data_loader = DataLoader(tokenized_dataset, batch_size=cfg.dataset.batch_size)

    return model, tokenizer, data_loader


def apply_ptq(model: torch.nn.Module, device: str = "cpu") -> torch.nn.Module:
    """
    Applies Post-Training Dynamic Quantization to the model.

    Note: PyTorch's dynamic quantization currently only supports CPU inference.
    This is why quantization experiments measure latency on CPU while other
    experiments default to GPU when available.

    Args:
        model: Model to quantize
        device: Target device ('cpu' only, as GPU quantization not supported)
    """
    print(">>> Applying Post-Training Quantization (Dynamic)...")
    print("⚠️  Note: Quantized models will run on CPU (PyTorch limitation)")
    model_cpu = model.cpu()
    # Ensure model is in evaluation mode for quantization
    model_cpu.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu, {torch.nn.Linear}, dtype=torch.qint8
    )
    # Keep quantized model in evaluation mode for inference
    quantized_model.eval()
    return quantized_model


def save_quant_results(cfg: DictConfig, method: str, metrics: dict):
    """Saves quantization experiment results."""
    # Save to outputs/quantization so generate_all_figures.py can find the results
    output_dir = Path("outputs") / "quantization"
    output_dir.mkdir(exist_ok=True, parents=True)
    file_path = output_dir / f"{method}_metrics.json"

    print(f"Saving quantization results to {file_path}")
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


def run_quant_then_permute(cfg, model, tokenizer, data_loader):
    """Strategy 1: Quantize the model, then find and apply the best permutation."""
    print("\n--- Running Strategy: (1) Quant-then-Permute ---")
    quantized_model = apply_ptq(model)
    wrapped_model = ModelWrapper(quantized_model)

    permutation = find_optimal_permutation(
        model=wrapped_model,
        data_loader=data_loader,
        target_layer_name=cfg.model.iasp.target_layer_name,
        cluster_size_range=tuple(cfg.model.iasp.cluster_size_range),
        device="cpu",  # Quantized models run on CPU
    )
    wrapped_model.permute_model_weights(permutation)

    dummy_input = torch.randint(0, cfg.model.vocab_size, (1, 512))
    latency = profiler.measure_latency(wrapped_model, {"input_ids": dummy_input})
    print(f"Final Latency: {latency:.2f} ms")
    save_quant_results(cfg, "quant_then_permute", {"latency": latency})


def run_permute_then_quant(cfg, model, tokenizer, data_loader):
    """Strategy 2: Find permutation on FP32, apply it, then quantize."""
    print("\n--- Running Strategy: (2) Permute-then-Quant ---")
    wrapped_model = ModelWrapper(model)
    if torch.cuda.is_available():
        wrapped_model.cuda()

    permutation = find_optimal_permutation(
        model=wrapped_model,
        data_loader=data_loader,
        target_layer_name=cfg.model.iasp.target_layer_name,
        cluster_size_range=tuple(cfg.model.iasp.cluster_size_range),
        device="cuda" if torch.cuda.is_available() else "cpu",  # FP32 model can use GPU
    )
    wrapped_model.permute_model_weights(permutation)

    final_model = apply_ptq(wrapped_model.model)
    dummy_input = torch.randint(0, cfg.model.vocab_size, (1, 512))
    latency = profiler.measure_latency(final_model, {"input_ids": dummy_input})
    print(f"Final Latency: {latency:.2f} ms")
    save_quant_results(cfg, "permute_then_quant", {"latency": latency})


def run_permute_quant_repermute(cfg, model, tokenizer, data_loader):
    """Strategy 3: Permute on FP32, quantize, then re-permute on INT8."""
    print("\n--- Running Strategy: (3) Permute-Quant-RePermute ---")
    wrapped_model = ModelWrapper(model)
    if torch.cuda.is_available():
        wrapped_model.cuda()

    perm1 = find_optimal_permutation(
        model=wrapped_model,
        data_loader=data_loader,
        target_layer_name=cfg.model.iasp.target_layer_name,
        cluster_size_range=tuple(cfg.model.iasp.cluster_size_range),
        device="cuda" if torch.cuda.is_available() else "cpu",  # FP32 model can use GPU
    )
    wrapped_model.permute_model_weights(perm1)

    quantized_permuted_model = apply_ptq(wrapped_model.model)

    wrapped_quant_model = ModelWrapper(quantized_permuted_model)
    perm2 = find_optimal_permutation(
        model=wrapped_quant_model,
        data_loader=data_loader,
        target_layer_name=cfg.model.iasp.target_layer_name,
        cluster_size_range=tuple(cfg.model.iasp.cluster_size_range),
        device="cpu",  # Quantized models run on CPU
    )
    wrapped_quant_model.permute_model_weights(perm2)

    dummy_input = torch.randint(0, cfg.model.vocab_size, (1, 512))
    latency = profiler.measure_latency(wrapped_quant_model, {"input_ids": dummy_input})
    print(f"Final Latency: {latency:.2f} ms")
    save_quant_results(cfg, "permute_quant_repermute", {"latency": latency})


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Initialize random seeds for reproducible experiments
    set_random_seeds(cfg.seed)

    method = OmegaConf.select(cfg, "method", default="permute_then_quant")

    model, tokenizer, data_loader = get_model_and_data(cfg)

    if method == "quant_then_permute":
        run_quant_then_permute(cfg, model, tokenizer, data_loader)
    elif method == "permute_then_quant":
        run_permute_then_quant(cfg, model, tokenizer, data_loader)
    elif method == "permute_quant_repermute":
        run_permute_quant_repermute(cfg, model, tokenizer, data_loader)
    else:
        raise ValueError(f"Unknown quantization method: {method}. "
                        f"Supported methods: quant_then_permute, permute_then_quant, permute_quant_repermute")


if __name__ == "__main__":
    main()
