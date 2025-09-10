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
    verdict = {
        "delta": {
            "lat_rel": None if (lat0 is None or lat1 is None) else (lat1 - lat0) / lat0,
            "l2_pp": None if (l20 is None or l21 is None) else (l21 - l20),
            "J": dJ,
        },
        "accepted": accepted,
        "rolled_back": (not accepted),
        "incomplete": not ((lat0 is not None) and (lat1 is not None)),
    }
    # Optional quality gate: apply only when both baseline and trial include quality
    base_q = baseline.get("quality")
    trial_q = trial.get("quality")
    if isinstance(base_q, dict) and isinstance(trial_q, dict):
        ppl_rel_max = 0.002  # +0.2%
        acc_drop_pp_max = 0.1  # -0.1pp
        qok = True
        if base_q.get("metric") == "perplexity" and trial_q.get("metric") == "perplexity":
            b = base_q.get("after"); t = trial_q.get("after")
            if b is not None and t is not None:
                rel = (t - b) / max(1e-9, b)
                qok &= (rel <= ppl_rel_max)
        if base_q.get("metric") == "accuracy" and trial_q.get("metric") == "accuracy":
            b = base_q.get("after"); t = trial_q.get("after")
            if b is not None and t is not None:
                dpp = (t - b) * 100.0
                qok &= (dpp >= -acc_drop_pp_max)
        verdict["quality_ok"] = bool(qok)
        verdict["accepted"] = bool(verdict["accepted"] and qok)
    return verdict
