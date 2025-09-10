from __future__ import annotations

from typing import Dict


def measure_ept_stub(tokens: int = 1) -> Dict[str, float]:
    """Mock EpT calculator (no NVML)."""
    return {"ept_j_per_tok": float("nan"), "tokens": tokens}


__all__ = ["measure_ept_stub"]

