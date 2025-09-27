"""Bridge StableHLO verification metrics into runtime acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class BridgeResult:
    passed: bool
    metrics: dict[str, float]


def bridge_metrics(hlo_metrics: Mapping[str, float], runtime_metrics: dict[str, float]) -> BridgeResult:
    merged = dict(runtime_metrics)
    for key, value in hlo_metrics.items():
        merged[f"stablehlo.{key}"] = float(value)
    passed = all(v <= 0.0 for k, v in hlo_metrics.items() if k.endswith("_delta"))
    return BridgeResult(passed=passed, metrics=merged)
