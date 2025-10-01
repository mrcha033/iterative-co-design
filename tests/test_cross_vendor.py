from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import torch

from icd.measure.cross_vendor import CrossVendorProfiler, CrossVendorValidator


def _run_command(command: Sequence[str]) -> None:
    subprocess.check_call(command)


def test_cross_vendor_profiler_materialises_model(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 4)
    inputs = torch.randn(2, 4)

    captured: dict[str, Sequence[str]] = {}

    def fake_collector(
        command: Sequence[str],
        metrics: Sequence[str],
        env: MutableMapping[str, str] | None,
        cwd: str | None,
    ) -> Mapping[str, float]:
        captured["command"] = list(command)
        _run_command(command)
        return {
            "sm__cycles_active.avg.pct_of_peak_sustained_elapsed": 80.0,
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": 60.0,
            "lts__t_sectors_hit_rate.pct": 75.0,
            "tensor_precision_fu_utilization": 70.0,
        }

    profiler = CrossVendorProfiler("nvidia", nvidia_collector=fake_collector)
    result = profiler.collect_unified_metrics(model=model, inputs=inputs, output_dir=tmp_path, keep_artifacts=True, warmup=1, iterations=1)

    assert captured["command"][0].endswith("python") or captured["command"][0] == "python"
    assert result["metrics"]["gpu_utilization"] == 80.0
    assert result["metrics"]["memory_bandwidth_pct"] == 60.0
    assert result["metrics"]["cache_hit_rate"] == 75.0
    assert result["metrics"]["compute_efficiency"] == 70.0


def test_cross_vendor_validator_summary(tmp_path: Path) -> None:
    class DummyProfiler:
        def __init__(self, vendor: str, value: float) -> None:
            self._value = value
            self.vendor = vendor

        def collect_unified_metrics(self, **_: object) -> Mapping[str, Mapping[str, float]]:
            return {"metrics": {"gpu_utilization": self._value}}

    values = {"nvidia-a100": 1.0, "amd-mi100": 0.9}

    def factory(vendor: str) -> DummyProfiler:
        return DummyProfiler(vendor, values[vendor])

    validator = CrossVendorValidator(list(values.keys()), profiler_factory=factory)

    def builder(vendor: str, *_args: object) -> Sequence[str]:
        return ["python", "-c", "print('ok')"]

    results = validator.validate_model(command_builder=builder, output_root=tmp_path)
    summary = results["summary"]["gpu_utilization"]
    assert summary["baseline"] == "nvidia-a100"
    assert summary["values"]["nvidia-a100"] == 1.0
    assert summary["relative_to_baseline"]["amd-mi100"] == 0.9


def test_cross_vendor_validator_report(tmp_path: Path) -> None:
    runs = {
        "runs": {
            "nvidia": {"metrics": {"gpu_utilization": 1.0}},
            "amd": {"metrics": {"gpu_utilization": 0.9}},
        },
        "summary": {
            "gpu_utilization": {
                "values": {"nvidia": 1.0, "amd": 0.9},
                "baseline": "nvidia",
                "relative_to_baseline": {"nvidia": 1.0, "amd": 0.9},
            }
        },
    }

    validator = CrossVendorValidator(["nvidia", "amd"])
    output_file = tmp_path / "report.html"
    validator.generate_report(runs, output_file)

    content = output_file.read_text(encoding="utf-8")
    assert "Cross-Vendor Profiling Report" in content
    assert "nvidia" in content and "amd" in content
    assert json.loads(json.dumps("ok")) == "ok"  # sanity check to exercise json import
