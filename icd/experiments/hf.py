"""HuggingFace-based model loaders for executable experiments."""

from __future__ import annotations

from typing import Any, Iterable, Tuple

from icd.utils.imports import load_object

from ._torch_utils import resolve_device, resolve_dtype


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

    torch_dtype = resolve_dtype(dtype, torch)
    device_name = resolve_device(device, torch)

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

    torch_dtype = resolve_dtype(dtype, torch)
    device_name = resolve_device(device, torch)

    model = model_loader_fn(model_name, torch_dtype=torch_dtype)
    model.to(device_name)
    model.eval()

    example = [encoded["input_ids"].to(device_name)]
    if "attention_mask" in encoded:
        example.append(encoded["attention_mask"].to(device_name))

    return model, tuple(example)


def load_mamba_ssm_causal_lm(
    model_name: str,
    *,
    sequence_length: int = 256,
    batch_size: int = 1,
    prompt: str | Iterable[str] = "Once upon a time",
    tokenizer_name: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
) -> Tuple[Any, Tuple[Any, ...]]:
    """Load original mamba-ssm model (not HuggingFace Transformers).

    This loader uses the original mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel
    which has the A, B, C module structure that permutation code expects.
    """

    try:
        import torch
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "mamba-ssm and transformers are required for original Mamba models. "
            "Install with: pip install mamba-ssm causal-conv1d"
        ) from exc

    # Use GPT-NeoX tokenizer (Mamba was trained with this)
    tokenizer_name = tokenizer_name or "EleutherAI/gpt-neox-20b"
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    texts = _repeat_text(prompt, batch_size)
    encoded = tok(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
    )

    torch_dtype = resolve_dtype(dtype, torch)
    device_name = resolve_device(device, torch)

    # Load original mamba-ssm model
    model = MambaLMHeadModel.from_pretrained(
        model_name,
        device=device_name,
        dtype=torch_dtype
    )
    model.eval()

    example = [encoded["input_ids"].to(device_name)]

    return model, tuple(example)


__all__ = [
    "load_hf_sequence_classifier",
    "load_hf_causal_lm",
    "load_mamba_ssm_causal_lm",
]

