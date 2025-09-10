from __future__ import annotations

from typing import Dict


def collect_l2_section_stub() -> Dict[str, float]:
    """Mock L2 metrics collector when ncu is unavailable.

    Returns a fixed shape to keep schema consistent.
    """
    return {"l2_tex__t_sector_hit_rate.pct": float("nan")}


__all__ = ["collect_l2_section_stub"]

