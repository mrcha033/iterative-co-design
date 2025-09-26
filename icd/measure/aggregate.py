"""Aggregation helpers for metrics files."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterable, List

from .gates import make_pairwise_summary, verdict

__all__ = ["load_metrics", "write_pairwise_summary"]


def load_metrics(pattern: str = "metrics.*.json") -> List[dict]:
    results: List[dict] = []
    for path in glob.glob(pattern):
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def write_pairwise_summary(
    *,
    metrics_files: Iterable[str] | None = None,
    output_path: str | Path = "pairwise_summary.json",
) -> List[dict]:
    if metrics_files is not None:
        metrics = [json.load(open(str(p), "r", encoding="utf-8")) for p in metrics_files]
    else:
        metrics = load_metrics()

    by_mode = {str(m.get("mode", "")).lower(): m for m in metrics}
    dense = by_mode.get("dense")
    linear = by_mode.get("linear")

    for m in metrics:
        verdict(m, dense_metrics=dense, linear_metrics=linear)

    summary = make_pairwise_summary(metrics)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
