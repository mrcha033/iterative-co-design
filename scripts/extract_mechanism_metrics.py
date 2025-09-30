#!/usr/bin/env python3
"""Extract and summarize mechanistic metrics for Table 3.

This utility reads the `metrics.json` files produced by the mechanism
"deep dive" experiments and writes a CSV table that captures the
relationship between modularity, cache hit rate, and latency for each
pipeline mode.

Example
-------
python scripts/extract_mechanism_metrics.py \
    experiments/mechanism/mamba_deepdive/ \
    --output experiments/results/table3_mamba_deepdive.csv \
    --raw-output experiments/results/table3_mamba_deepdive_raw.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RunMetrics:
    """Container for the metrics captured from a single run."""

    mode: str
    run_id: str
    modularity: Optional[float]
    hit_rate: Optional[float]
    latency_ms: Optional[float]
    latency_p50: Optional[float]
    latency_p95: Optional[float]

    @classmethod
    def from_metrics_file(cls, metrics_path: Path, mode: str) -> "RunMetrics | None":
        if not metrics_path.exists():
            return None

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        solver_stats = metrics.get("solver_stats", {}) or {}
        modularity = solver_stats.get("Q_final") or solver_stats.get("Q_louvain")
        # Some runs persist modularity as nested dict {"value": x}
        if isinstance(modularity, dict):
            modularity = modularity.get("value")

        latency_block = metrics.get("latency_ms", {}) or {}
        if isinstance(latency_block, dict):
            latency_mean = latency_block.get("mean")
            latency_p50 = latency_block.get("p50")
            latency_p95 = latency_block.get("p95")
        else:
            latency_mean = latency_block or metrics.get("latency_ms_mean")
            latency_p50 = metrics.get("latency_ms_p50")
            latency_p95 = metrics.get("latency_ms_p95")

        hit_rate = metrics.get("l2_hit_pct")
        if isinstance(hit_rate, dict):
            hit_rate = hit_rate.get("mean")

        return cls(
            mode=mode,
            run_id=metrics_path.parent.name,
            modularity=_to_float(modularity),
            hit_rate=_to_float(hit_rate),
            latency_ms=_to_float(latency_mean),
            latency_p50=_to_float(latency_p50),
            latency_p95=_to_float(latency_p95),
        )


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def discover_runs(root: Path) -> Iterable[RunMetrics]:
    for mode_dir in sorted(root.iterdir()):
        if not mode_dir.is_dir():
            continue
        mode = mode_dir.name
        for run_dir in sorted(mode_dir.glob("run_*")):
            metrics_path = run_dir / "metrics.json"
            run_metrics = RunMetrics.from_metrics_file(metrics_path, mode)
            if run_metrics is not None:
                yield run_metrics


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("mode", dropna=True)
    summary_rows: List[Dict[str, float]] = []
    for mode, group in grouped:
        row = {
            "mode": mode,
            "runs": int(group.shape[0]),
        }
        for column in ("modularity", "hit_rate", "latency_ms"):
            series = group[column].dropna()
            if series.empty:
                row[f"{column}_mean"] = np.nan
                row[f"{column}_std"] = np.nan
            else:
                row[f"{column}_mean"] = float(series.mean())
                row[f"{column}_std"] = float(series.std(ddof=1)) if series.size > 1 else 0.0
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values("mode")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract mechanistic metrics")
    parser.add_argument(
        "root",
        type=Path,
        help="Directory produced by run_mechanism_deepdive.sh (contains mode/run_*/metrics.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mechanism_summary.csv"),
        help="Path to write aggregated summary CSV",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Optional path to dump per-run metrics",
    )
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"Mechanism directory not found: {args.root}")

    runs = list(discover_runs(args.root))
    if not runs:
        raise RuntimeError(
            f"No metrics.json files found under {args.root}. Did you run the deep dive script?"
        )

    raw_df = pd.DataFrame([r.__dict__ for r in runs])
    raw_df = raw_df.sort_values(["mode", "run_id"]).reset_index(drop=True)

    summary_df = summarize(raw_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output, index=False)

    if args.raw_output is not None:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(args.raw_output, index=False)

    print("✓ Extracted mechanistic metrics")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print(f"Saved summary to {args.output}")
    if args.raw_output is not None:
        print(f"Saved raw metrics to {args.raw_output}")


if __name__ == "__main__":
    main()
