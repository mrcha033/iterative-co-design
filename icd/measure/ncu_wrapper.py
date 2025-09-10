from __future__ import annotations

"""Nsight Compute (ncu) JSON section parser (CI-friendly).

If the file is missing or malformed, returns a dict with NaN values.
"""

import json
from typing import Dict


def parse_l2_hit_from_section_json(path: str) -> Dict[str, float | None]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Best-effort: look for a common L2 hit metric key
        for key in (
            "l2_tex__t_sector_hit_rate.pct",
            "lts__t_sectors_hit_rate.pct",
            "lts__t_sectors_srcunit_tex_op_read_lookup_hit_rate.pct",
        ):
            v = _find_key(data, key)
            if isinstance(v, (int, float)):
                return {"l2_hit_pct": float(v)}
    except Exception:
        pass
    return {"l2_hit_pct": float("nan")}


def _find_key(obj, target: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == target:
                return v
            r = _find_key(v, target)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _find_key(item, target)
            if r is not None:
                return r
    return None


__all__ = ["parse_l2_hit_from_section_json"]

