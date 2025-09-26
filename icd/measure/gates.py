"""Gate evaluation utilities for metrics verdicts."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

__all__ = ["verdict", "make_pairwise_summary"]


_DEFAULT_THRESHOLDS = {
    "sst2.acc_min": 0.90,
    "wt103.ppl_factor": 1.01,
    "iter_vs_linear_latency_factor": 0.95,
}


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
    ok = True

    if task == "sst2":
        acc = float(metrics.get("accuracy") or 0.0)
        ok &= acc >= th["sst2.acc_min"]
    elif task in {"wt103", "wikitext", "wikitext-103"}:
        ppl = float(metrics.get("ppl") or 0.0)
        if dense_metrics and dense_metrics.get("ppl"):
            base = float(dense_metrics["ppl"]) or 1.0
            ok &= ppl <= base * th["wt103.ppl_factor"]
    if mode == "iterative" and linear_metrics and linear_metrics.get("latency_ms_mean"):
        lat = float(metrics.get("latency_ms_mean") or 0.0)
        lat_lin = float(linear_metrics["latency_ms_mean"]) or 1.0
        factor = th.setdefault("iter_vs_linear_latency_factor", 0.95)
        ok &= lat <= lat_lin * factor

    metrics["verdict"] = "pass" if ok else "fail"
    metrics.setdefault("gates", {}).update(th)
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
    return metrics
