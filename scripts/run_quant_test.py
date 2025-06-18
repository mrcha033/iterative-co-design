import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.quantization
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.profiler import LatencyProfiler
from src.co_design.iasp import find_optimal_permutation
from src.models.wrapper import ModelWrapper

profiler = LatencyProfiler()


def get_model_and_data(cfg: DictConfig):
    """Loads model, tokenizer, and a data sample based on the config."""
    print(f"Loading model: {cfg.model.name}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    print(f"Loading dataset: {cfg.dataset.name}")
    dataset = load_dataset(cfg.dataset.name, cfg.dataset.get("config"))
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


def apply_ptq(model: torch.nn.Module) -> torch.nn.Module:
    """Applies Post-Training Dynamic Quantization to the model."""
    print(">>> Applying Post-Training Quantization (Dynamic)...")
    model_cpu = model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model


def save_quant_results(cfg: DictConfig, method: str, metrics: dict):
    """Saves quantization experiment results."""
    output_dir = Path.cwd() / "quantization"
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
    )
    wrapped_model.permute_model_weights(perm1)

    quantized_permuted_model = apply_ptq(wrapped_model.model)

    wrapped_quant_model = ModelWrapper(quantized_permuted_model)
    perm2 = find_optimal_permutation(
        model=wrapped_quant_model,
        data_loader=data_loader,
        target_layer_name=cfg.model.iasp.target_layer_name,
        cluster_size_range=tuple(cfg.model.iasp.cluster_size_range),
    )
    wrapped_quant_model.permute_model_weights(perm2)

    dummy_input = torch.randint(0, cfg.model.vocab_size, (1, 512))
    latency = profiler.measure_latency(wrapped_quant_model, {"input_ids": dummy_input})
    print(f"Final Latency: {latency:.2f} ms")
    save_quant_results(cfg, "permute_quant_repermute", {"latency": latency})


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    method = OmegaConf.select(cfg, "method", default="permute_then_quant")

    model, tokenizer, data_loader = get_model_and_data(cfg)

    if method == "quant_then_permute":
        run_quant_then_permute(cfg, model, tokenizer, data_loader)
    elif method == "permute_then_quant":
        run_permute_then_quant(cfg, model, tokenizer, data_loader)
    elif method == "permute_quant_repermute":
        run_permute_quant_repermute(cfg, model, tokenizer, data_loader)


if __name__ == "__main__":
    main()
