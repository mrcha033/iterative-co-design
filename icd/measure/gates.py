"""Gate evaluation utilities for metrics verdicts."""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Optional

__all__ = ["verdict", "make_pairwise_summary", "PRD_GATE_DEFAULTS"]


PRD_GATE_DEFAULTS = {
    "iter.latency_rel_max": -0.20,
    "iter.l2_pp_min": 10.0,
    "iter.ept_rel_max": -0.15,
    "quality.acc_drop_pp_max": 0.1,
}


_DEFAULT_THRESHOLDS = {
    "sst2.acc_min": 0.90,
    "wt103.ppl_factor": 1.01,
    **PRD_GATE_DEFAULTS,
}


def _float_or_nan(value: object) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return result


def verdict(
    metrics: Dict[str, object],
    *,
    dense_metrics: Optional[Dict[str, object]] = None,
    linear_metrics: Optional[Dict[str, object]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    th = dict(_DEFAULT_THRESHOLDS)
    if thresholds:
        th.update({k: float(v) for k, v in thresholds.items()})

    task = str(metrics.get("task", "")).lower()
    mode = str(metrics.get("mode", "")).lower()

    gates = metrics.setdefault("gates", {})
    thresholds_bucket = gates.setdefault("thresholds", {})
    thresholds_bucket.update(th)
    observed = gates.setdefault("observed", {})
    status = gates.setdefault("status", {})
    missing = gates.setdefault("missing", [])

    def apply_gate(name: str, value: Optional[float], predicate: Callable[[float], bool]) -> bool:
        if value is None:
            observed[name] = None
            if name not in missing:
                missing.append(name)
            status[name] = None
            return True
        try:
            val = float(value)
        except Exception:
            observed[name] = None
            if name not in missing:
                missing.append(name)
            status[name] = None
            return True
        if math.isnan(val):
            observed[name] = val
            if name not in missing:
                missing.append(name)
            status[name] = None
            return True
        observed[name] = val
        passed = bool(predicate(val))
        status[name] = passed
        return passed

    ok = True

    if task == "sst2":
        acc_val = metrics.get("accuracy")
        ok &= apply_gate("sst2.acc_min", acc_val, lambda v: v >= th["sst2.acc_min"])
    elif task in {"wt103", "wikitext", "wikitext-103"}:
        ppl_val = metrics.get("ppl")
        base = None
        if dense_metrics is not None:
            base = dense_metrics.get("ppl")
        ratio = None
        if ppl_val is not None and base not in (None, 0.0):
            try:
                ratio = float(ppl_val) / float(base)
            except Exception:
                ratio = None
        ok &= apply_gate("wt103.ppl_factor", ratio, lambda v: v <= th["wt103.ppl_factor"])

    if mode == "iterative":
        lat_val = metrics.get("latency_ms_mean")
        lat_lin = None
        if linear_metrics is not None:
            lat_lin = linear_metrics.get("latency_ms_mean")
        lat_rel = None
        if lat_val not in (None, 0.0) and lat_lin not in (None, 0.0):
            try:
                lat_rel = (float(lat_val) - float(lat_lin)) / float(lat_lin)
            except Exception:
                lat_rel = None
        ok &= apply_gate("iter.latency_rel", lat_rel, lambda v: v <= th["iter.latency_rel_max"])

        l2_iter = metrics.get("l2_hit_pct")
        l2_base = None if linear_metrics is None else linear_metrics.get("l2_hit_pct")
        l2_pp = None
        if l2_iter is not None and l2_base is not None:
            try:
                lhs = float(l2_iter)
                rhs = float(l2_base)
                delta = lhs - rhs
                if max(abs(lhs), abs(rhs)) <= 1.0:
                    delta *= 100.0
                l2_pp = delta
            except Exception:
                l2_pp = None
        ok &= apply_gate("iter.l2_pp", l2_pp, lambda v: v >= th["iter.l2_pp_min"])

        ept_iter = metrics.get("ept_j_per_tok")
        ept_base = None if linear_metrics is None else linear_metrics.get("ept_j_per_tok")
        ept_rel = None
        if ept_iter not in (None, 0.0) and ept_base not in (None, 0.0):
            try:
                ept_rel = (float(ept_iter) - float(ept_base)) / float(ept_base)
            except Exception:
                ept_rel = None
        ok &= apply_gate("iter.ept_rel", ept_rel, lambda v: v <= th["iter.ept_rel_max"])

        acc_iter = metrics.get("accuracy")
        acc_dense = None if dense_metrics is None else dense_metrics.get("accuracy")
        acc_drop_pp = None
        if acc_iter is not None and acc_dense is not None:
            try:
                acc_drop_pp = (float(acc_iter) - float(acc_dense)) * 100.0
            except Exception:
                acc_drop_pp = None
        ok &= apply_gate(
            "quality.acc_drop_pp",
            acc_drop_pp,
            lambda v: v >= -th["quality.acc_drop_pp_max"],
        )

    metrics["verdict"] = "pass" if ok else "fail"
    return metrics


def make_pairwise_summary(metrics_list: Iterable[Dict[str, object]]) -> list[Dict[str, object]]:
    metrics = list(metrics_list)
    by_mode = {str(m.get("mode", "")).lower(): m for m in metrics}
    dense = by_mode.get("dense")
    for m in metrics:
        mode = str(m.get("mode", "")).lower()
        if dense:
            if "latency_ms_mean" in m and "latency_ms_mean" in dense:
                m["delta_latency_vs_dense"] = float(m.get("latency_ms_mean", 0.0)) - float(dense.get("latency_ms_mean", 0.0))
            if "ppl" in m and "ppl" in dense:
                m["delta_ppl_vs_dense"] = float(m.get("ppl", 0.0)) - float(dense.get("ppl", 0.0))
            if "accuracy" in m and "accuracy" in dense:
                m["delta_accuracy_vs_dense"] = float(m.get("accuracy", 0.0)) - float(dense.get("accuracy", 0.0))
        if mode == "iterative" and "linear" in by_mode:
            linear = by_mode["linear"]
            if "latency_ms_mean" in m and "latency_ms_mean" in linear:
                m["delta_latency_vs_linear"] = float(m.get("latency_ms_mean", 0.0)) - float(linear.get("latency_ms_mean", 0.0))
            if "l2_hit_pct" in m and "l2_hit_pct" in linear:
                m["delta_l2_hit_vs_linear"] = float(m.get("l2_hit_pct", 0.0)) - float(linear.get("l2_hit_pct", 0.0))
            if "ept_j_per_tok" in m and "ept_j_per_tok" in linear:
                m["delta_ept_vs_linear"] = float(m.get("ept_j_per_tok", 0.0)) - float(linear.get("ept_j_per_tok", 0.0))
    return metrics
