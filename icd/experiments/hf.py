"""HuggingFace-based model loaders for executable experiments."""

from __future__ import annotations

from typing import Any, Iterable, Tuple

from icd.utils.imports import load_object


def _resolve_device(device: str | None, torch_module) -> str:
    if device:
        return device
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _resolve_dtype(dtype: str | None, torch_module):
    mapping = {
        None: torch_module.float32,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
    }
    key = (dtype or "float32").lower()
    if key not in mapping:
        raise ValueError(f"unsupported torch dtype '{dtype}'")
    return mapping[key]


def _repeat_text(prompt: str | Iterable[str], batch_size: int) -> list[str]:
    if isinstance(prompt, (list, tuple)):
        texts = list(prompt)
    else:
        texts = [str(prompt)]
    if len(texts) >= batch_size:
        return texts[:batch_size]
    return texts + [texts[-1]] * (batch_size - len(texts))


def load_hf_sequence_classifier(
    model_name: str,
    *,
    sequence_length: int = 128,
    batch_size: int = 1,
    prompt: str | Iterable[str] = "This sentence is neutral.",
    tokenizer_name: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    tokenizer_loader: str | None = None,
    model_loader: str | None = None,
) -> Tuple[Any, Tuple[Any, ...]]:
    """Load a HuggingFace sequence classification model and example inputs.

    Returns the `(model, example_inputs)` tuple required by the graph builder.
    """

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "transformers and torch are required for the HuggingFace loaders."
        ) from exc

    tokenizer_loader_fn = load_object(tokenizer_loader) if tokenizer_loader else AutoTokenizer.from_pretrained
    model_loader_fn = load_object(model_loader) if model_loader else AutoModelForSequenceClassification.from_pretrained

    tok = tokenizer_loader_fn(tokenizer_name or model_name)
    texts = _repeat_text(prompt, batch_size)
    encoded = tok(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
    )

    torch_dtype = _resolve_dtype(dtype, torch)
    device_name = _resolve_device(device, torch)

    model = model_loader_fn(model_name, torch_dtype=torch_dtype)
    model.to(device_name)
    model.eval()

    example = []
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        if key in encoded:
            example.append(encoded[key].to(device_name))

    return model, tuple(example)


def load_hf_causal_lm(
    model_name: str,
    *,
    sequence_length: int = 256,
    batch_size: int = 1,
    prompt: str | Iterable[str] = "Once upon a time",
    tokenizer_name: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    tokenizer_loader: str | None = None,
    model_loader: str | None = None,
) -> Tuple[Any, Tuple[Any, ...]]:
    """Load a HuggingFace causal language model (e.g., Mamba)."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "transformers and torch are required for the HuggingFace loaders."
        ) from exc

    tokenizer_loader_fn = load_object(tokenizer_loader) if tokenizer_loader else AutoTokenizer.from_pretrained
    model_loader_fn = load_object(model_loader) if model_loader else AutoModelForCausalLM.from_pretrained

    tok = tokenizer_loader_fn(tokenizer_name or model_name)
    texts = _repeat_text(prompt, batch_size)
    encoded = tok(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
    )

    torch_dtype = _resolve_dtype(dtype, torch)
    device_name = _resolve_device(device, torch)

    model = model_loader_fn(model_name, torch_dtype=torch_dtype)
    model.to(device_name)
    model.eval()

    example = [encoded["input_ids"].to(device_name)]
    if "attention_mask" in encoded:
        example.append(encoded["attention_mask"].to(device_name))

    return model, tuple(example)


__all__ = [
    "load_hf_sequence_classifier",
    "load_hf_causal_lm",
]

