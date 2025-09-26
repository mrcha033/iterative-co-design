from __future__ import annotations

from typing import Any, Dict

from icd.measure.gates import PRD_GATE_DEFAULTS


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
    ept0 = pp(baseline.get("ept_j_per_tok"))
    ept1 = pp(trial.get("ept_j_per_tok"))

    lat_rel = None
    if lat0 not in (None, 0.0) and lat1 is not None:
        lat_rel = (lat1 - lat0) / lat0

    l2_pp = None
    if l20 is not None and l21 is not None:
        delta = l21 - l20
        if max(abs(l20), abs(l21)) <= 1.0:
            delta *= 100.0
        l2_pp = delta

    ept_rel = None
    if ept0 not in (None, 0.0) and ept1 is not None:
        ept_rel = (ept1 - ept0) / ept0

    cond_J = (dJ is not None) and (dJ <= -eps_J)
    cond_lat = (lat_rel is not None) and (lat_rel <= PRD_GATE_DEFAULTS["iter.latency_rel_max"])
    cond_l2 = (l2_pp is not None) and (l2_pp >= PRD_GATE_DEFAULTS["iter.l2_pp_min"])
    cond_ept = (ept_rel is not None) and (ept_rel <= PRD_GATE_DEFAULTS["iter.ept_rel_max"])
    accepted = bool(cond_J and cond_lat and cond_l2 and cond_ept)

    missing = []
    if lat_rel is None:
        missing.append("latency_rel")
    if l2_pp is None:
        missing.append("l2_pp")
    if ept_rel is None:
        missing.append("ept_rel")
    verdict = {
        "delta": {
            "lat_rel": lat_rel,
            "l2_pp": l2_pp,
            "ept_rel": ept_rel,
            "J": dJ,
        },
        "accepted": accepted,
        "rolled_back": (not accepted),
        "incomplete": bool(missing),
        "missing": missing,
    }
    # Optional quality gate: apply only when both baseline and trial include quality
    base_q = baseline.get("quality")
    trial_q = trial.get("quality")
    if isinstance(base_q, dict) and isinstance(trial_q, dict):
        ppl_rel_max = 0.002  # +0.2%
        acc_drop_pp_max = PRD_GATE_DEFAULTS["quality.acc_drop_pp_max"]
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
