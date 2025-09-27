"""Utility helpers shared by experiment loaders that depend on PyTorch."""

from __future__ import annotations

from typing import Any


def resolve_device(device: str | None, torch_module: Any) -> str:
    """Return a device string, defaulting to CUDA when available."""

    if device:
        return str(device)
    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except AttributeError:  # pragma: no cover - defensive guard
        cuda_available = False
    return "cuda" if cuda_available else "cpu"


def resolve_dtype(dtype: str | None, torch_module: Any):
    """Translate a user provided dtype string into a ``torch.dtype``."""

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


__all__ = ["resolve_device", "resolve_dtype"]
