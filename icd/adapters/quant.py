"""Phase-1 quantization adapters (bitsandbytes + lightweight fallbacks)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Type
import warnings

import torch

try:  # optional nn import (can be stubbed out in tests)
    import torch.nn as nn  # type: ignore
except Exception:  # pragma: no cover
    nn = None  # type: ignore[assignment]

try:  # optional dependency
    import bitsandbytes as bnb  # type: ignore
except Exception:  # pragma: no cover - optional
    bnb = None

try:
    from torch.ao.quantization import quantize_dynamic as _quantize_dynamic
except Exception:  # pragma: no cover - optional
    _quantize_dynamic = None  # type: ignore[assignment]

__all__ = [
    "QuantConfig",
    "apply_bnb_int8",
    "apply_quant_from_config",
    "repack_linear_after_permutation",
    "apply_quant",
    "apply_post_training_quantization",
]


@dataclass
class QuantConfig:
    """Minimal configuration wrapper used by runners and orchestrator."""

    type: str = "none"
    order: str = "permute-then-quant"

    @classmethod
    def from_dict(cls, data: dict | None) -> "QuantConfig":
        if not data:
            return cls()
        return cls(
            type=str(data.get("type", "none")).lower(),
            order=str(data.get("order", "permute-then-quant")).lower(),
        )


_BNB_AVAILABLE = bnb is not None


def _ensure_bnb() -> None:
    if not _BNB_AVAILABLE:
        raise RuntimeError("bitsandbytes is required for quantization but not installed")


def apply_bnb_int8(
    model: torch.nn.Module,
    *,
    modules_to_not_convert: Sequence[str] | None = None,
    compute_dtype: torch.dtype | None = torch.float16,
) -> torch.nn.Module:
    """Replace supported Linear modules with bitsandbytes 8-bit kernels."""

    _ensure_bnb()
    from transformers import replace_with_bnb_linear  # lazy import

    skip = list(modules_to_not_convert or ("lm_head", "classifier"))
    quantized_model = replace_with_bnb_linear(
        model,
        modules_to_not_convert=skip,
        quantization_config=None,
    )

    if compute_dtype is not None:
        for module in quantized_model.modules():  # type: ignore[attr-defined]
            if hasattr(module, "state") and hasattr(module.state, "set_compute_type"):
                module.state.set_compute_type(compute_dtype)
    return quantized_model


def _clone_to_float(linear: torch.nn.Linear) -> torch.nn.Linear:
    new_linear = torch.nn.Linear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        dtype=linear.weight.dtype,
        device=linear.weight.device,
    )
    new_linear.weight.data.copy_(linear.weight.detach())
    if linear.bias is not None and new_linear.bias is not None:
        new_linear.bias.data.copy_(linear.bias.detach())
    return new_linear


def _bnb_linear_type() -> type:
    _ensure_bnb()
    return bnb.nn.Linear8bitLt  # type: ignore[attr-defined]


def repack_linear_after_permutation(linear_module: torch.nn.Module) -> torch.nn.Module:
    """Recreate a quantized Linear module after permutation adjustments."""

    if not _BNB_AVAILABLE:
        return linear_module
    bnb_linear_cls = _bnb_linear_type()
    if not isinstance(linear_module, bnb_linear_cls):
        return linear_module
    float_clone = _clone_to_float(linear_module.to(dtype=torch.float32))
    new_linear = bnb_linear_cls(
        float_clone.in_features,
        float_clone.out_features,
        bias=float_clone.bias is not None,
    )
    new_linear.weight = torch.nn.Parameter(float_clone.weight.detach().contiguous())
    if float_clone.bias is not None:
        new_linear.bias = torch.nn.Parameter(float_clone.bias.detach().contiguous())
    new_linear.to(linear_module.weight.device)
    return new_linear


def apply_quant_from_config(model: torch.nn.Module, cfg: QuantConfig) -> torch.nn.Module:
    if cfg.type == "none":
        return model
    if cfg.type in {"bnb-int8", "bnb8", "bnb-int8lt"}:
        return apply_bnb_int8(model)
    if cfg.type in {"bnb-4bit", "bnb4"}:
        _ensure_bnb()
        from transformers import BitsAndBytesConfig, replace_with_bnb_linear

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        skip = ["lm_head", "classifier"]
        return replace_with_bnb_linear(model, modules_to_not_convert=skip, quantization_config=quant_config)
    if cfg.type in {"dynamic", "torch-dynamic"}:
        quantized, _ = apply_quant(model, method="dynamic")
        return quantized or model
    raise ValueError(f"unsupported quant type '{cfg.type}' in Phase-1")


# -------------------------------
# Back-compat quantization shim
# -------------------------------

def _normalize_dtype(dt: torch.dtype | str) -> torch.dtype:
    if isinstance(dt, torch.dtype):
        return dt
    s = str(dt).lower()
    if s in ("qint8", "int8"):
        return torch.qint8
    if s in ("float16", "fp16", "half"):
        return torch.float16
    return torch.qint8


def _default_quant_modules(model: nn.Module | None) -> Tuple[Type[nn.Module], ...]:
    if nn is None:
        return tuple()
    return (nn.Linear,)


def _dtype_label(dtype: torch.dtype) -> str:
    if dtype == torch.qint8:
        return "int8"
    if dtype == torch.float16:
        return "float16"
    return str(dtype).replace("torch.", "")


def apply_quant(
    model: nn.Module | None,
    *,
    method: str = "dynamic",
    dtype: "torch.dtype | str" = "qint8",
    modules: Optional[Iterable[Type[nn.Module]]] = None,
    inplace: bool = True,
) -> Tuple[nn.Module | None, dict[str, object]]:
    """Lightweight quantization entrypoint used by tests and metadata hooks."""

    method_norm = (method or "dynamic").lower()
    dtype_norm = _normalize_dtype(dtype)
    target_modules = tuple(modules) if modules else _default_quant_modules(model)

    meta: dict[str, object] = {
        "delta_layout": method_norm not in {"none", "identity"},
        "quant": {
            "method": method_norm,
            "dtype": _dtype_label(dtype_norm),
        },
    }

    if model is None or method_norm in {"none", "identity"}:
        return model, meta

    if method_norm in {"dynamic", "ptq_dynamic", "ptq-minmax"}:
        if _quantize_dynamic is None:
            warnings.warn("torch.ao.quantization unavailable; returning model unchanged.")
            return model, meta
        quantized = _quantize_dynamic(model, {t for t in target_modules}, dtype=dtype_norm, inplace=inplace)
        return quantized, meta

    if method_norm in {"fp16", "half", "float16"}:
        for module in model.modules():
            if isinstance(module, target_modules):
                try:
                    module.to(dtype=torch.float16)
                except Exception:
                    for param in module.parameters(recurse=False):
                        param.data = param.data.half()
        return model, meta

    if method_norm in {"bnb", "bitsandbytes"}:
        if not _BNB_AVAILABLE:
            warnings.warn("bitsandbytes not available; falling back to dynamic quantization")
            return apply_quant(model, method="dynamic", dtype=dtype_norm, modules=target_modules, inplace=inplace)
        quantized = apply_bnb_int8(model)
        return quantized, meta

    warnings.warn(f"Unknown quantization method '{method}'. No changes applied.")
    return model, meta


def apply_post_training_quantization(
    model: nn.Module,
    *,
    dtype: str = "int8",
    calibration_samples: int = 100,
) -> nn.Module:
    """Apply post-training quantization (wrapper for experiment scripts).

    Args:
        model: PyTorch model to quantize
        dtype: Quantization dtype ("int8", "int4", "fp16")
        calibration_samples: Number of calibration samples (unused for PTQ)

    Returns:
        Quantized model
    """
    if dtype in ("int8", "qint8"):
        quantized, _ = apply_quant(model, method="dynamic", dtype="qint8")
        return quantized or model
    elif dtype in ("int4", "4bit"):
        cfg = QuantConfig(type="bnb-4bit")
        return apply_quant_from_config(model, cfg)
    elif dtype in ("fp16", "float16", "half"):
        quantized, _ = apply_quant(model, method="fp16")
        return quantized or model
    else:
        warnings.warn(f"Unknown dtype '{dtype}', returning model unchanged")
        return model
