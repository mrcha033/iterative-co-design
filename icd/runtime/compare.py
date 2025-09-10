from __future__ import annotations

from typing import Any, Dict


def decide(baseline: Dict[str, Any], trial: Dict[str, Any], fixed_clock: bool, eps_J: float = 0.01) -> Dict[str, Any]:
    def pp(x):
        try:
            return float(x) if x is not None else None
        except Exception:
            return None

    dJ = pp(trial.get("acceptance", {}).get("delta_J"))
    lat0 = pp(baseline.get("latency_ms", {}).get("mean"))
    lat1 = pp(trial.get("latency_ms", {}).get("mean"))
    l20 = pp(baseline.get("l2_hit_pct"))
    l21 = pp(trial.get("l2_hit_pct"))
    scale = 2.0 if (fixed_clock is False) else 1.0
    cond_J = (dJ is not None) and (dJ <= -eps_J)
    cond_lat = (lat0 is not None and lat1 is not None) and ((lat1 - lat0) / lat0 <= -0.02 * scale)
    cond_l2 = (l20 is not None and l21 is not None) and ((l21 - l20) >= 2.0 * scale)
    accepted = bool(cond_J and (cond_lat or cond_l2))
    return {
        "delta": {
            "lat_rel": None if (lat0 is None or lat1 is None) else (lat1 - lat0) / lat0,
            "l2_pp": None if (l20 is None or l21 is None) else (l21 - l20),
            "J": dJ,
        },
        "accepted": accepted,
        "rolled_back": (not accepted),
        "incomplete": not ((lat0 is not None) and (lat1 is not None)),
    }

