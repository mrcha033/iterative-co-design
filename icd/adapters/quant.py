from __future__ import annotations

from typing import Any, Dict, Tuple


def apply_quant(tensor: Any, *, dtype: str = "int8", method: str = "ptq-minmax") -> Tuple[Any, Dict[str, any]]:
    meta = {
        "quant": {"dtype": dtype, "method": method},
        "delta_layout": dtype.lower() in {"int8", "fp8"},
        "quality_delta_pp": 0.0,
    }
    return tensor, meta


__all__ = ["apply_quant"]

