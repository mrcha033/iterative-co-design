#!/usr/bin/env python3
"""Compare ICD latency metrics against TVM baseline metadata.

This utility bridges the documentation in ``docs/TVM_Evaluation.md`` which
references an explicit comparison script.  It accepts the JSON metrics emitted
by ``icd.cli`` runs and the ``metadata.json`` produced by
``scripts/run_autotvm.py`` (or the orchestrator's automatic TVM baseline) and
reports relative performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any, Dict, Tuple


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_icd_latency(metrics: Dict[str, Any]) -> Tuple[float | None, Dict[str, Any]]:
    latency = metrics.get("latency_ms")
    if isinstance(latency, dict):
        mean = latency.get("mean")
    else:
        mean = metrics.get("latency_ms_mean")
        latency = {
            "mean": mean,
            "p50": metrics.get("latency_ms_p50"),
            "p95": metrics.get("latency_ms_p95"),
            "ci95": metrics.get("latency_ms_ci95"),
        }
    return (float(mean) if mean is not None else None, latency or {})


def _extract_tvm_latency(metadata: Dict[str, Any]) -> Tuple[float | None, Dict[str, Any]]:
    latency = metadata.get("latency")
    if isinstance(latency, dict):
        mean = latency.get("mean")
        if mean is None:
            mean = latency.get("latency_ms_mean")
    else:
        mean = metadata.get("latency_ms_mean")
        latency = {
            "mean": mean,
            "p50": metadata.get("latency_ms_p50"),
            "p95": metadata.get("latency_ms_p95"),
        }
    return (float(mean) if mean is not None else None, latency or {})


def _format_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _write_output(path: pathlib.Path, result: Dict[str, Any]) -> None:
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "value"])
            for key, value in result.items():
                writer.writerow([key, json.dumps(value)])
    else:
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("icd_metrics", type=pathlib.Path, help="Path to ICD metrics.json")
    ap.add_argument("tvm_metadata", type=pathlib.Path, help="Path to TVM metadata.json")
    ap.add_argument("--output", type=pathlib.Path, help="Optional path to store comparison (JSON or CSV)")
    ap.add_argument("--quiet", action="store_true", help="Suppress stdout summary")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    metrics = _load_json(args.icd_metrics)
    metadata = _load_json(args.tvm_metadata)

    icd_mean, icd_detail = _extract_icd_latency(metrics)
    tvm_mean, tvm_detail = _extract_tvm_latency(metadata)

    comparison = {
        "icd_latency_ms": icd_mean,
        "icd_latency_detail": icd_detail,
        "tvm_latency_ms": tvm_mean,
        "tvm_latency_detail": tvm_detail,
        "speedup_vs_tvm": _format_ratio(tvm_mean, icd_mean),
        "delta_ms": (tvm_mean - icd_mean) if (tvm_mean is not None and icd_mean is not None) else None,
    }

    if "config" in metadata:
        comparison["tvm_config"] = metadata["config"]
    if "verification" in metadata:
        comparison["tvm_verification"] = metadata["verification"]

    if args.output:
        _write_output(args.output, comparison)

    if not args.quiet:
        print(json.dumps(comparison, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
