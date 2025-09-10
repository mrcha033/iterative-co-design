from __future__ import annotations

from typing import Any, Dict, Tuple


def apply_kvcache(cache: Any, *, block: int = 128, drop: float = 0.1) -> Tuple[Any, Dict[str, any]]:
    meta = {
        "kv": {"block": block, "drop": drop},
        "delta_layout": bool(block >= 64),
        "quality_delta_pp": 0.0,
    }
    return cache, meta


__all__ = ["apply_kvcache"]

