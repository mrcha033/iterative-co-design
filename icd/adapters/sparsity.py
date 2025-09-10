from __future__ import annotations

from typing import Any, Dict, Tuple


def apply_sparsity(tensor: Any, *, type: str = "2:4", rate: float = 0.5) -> Tuple[Any, Dict[str, Any]]:
    """Stub: return input tensor and sparsity meta with delta_layout heuristic.

    delta_layout is true when type or rate crosses simple thresholds.
    """
    meta = {
        "sparsity": {"type": type, "rate": rate},
        "delta_layout": bool(rate >= 0.25),
        "quality_delta_pp": 0.0,  # unknown in mock
    }
    return tensor, meta


__all__ = ["apply_sparsity"]

