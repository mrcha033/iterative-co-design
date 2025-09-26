import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest

from icd.errors import ConfigError, MeasureError
from icd.measure import latency as latency_mod
from icd.measure.l2_ncu import collect_l2_section_stub
from icd.measure.ncu_wrapper import parse_l2_hit_from_section_json
from icd.measure.nvml_logger import sample_power_series
from icd.measure.power import measure_ept_stub
from icd.measure.quality import eval_acc, eval_ppl
from icd.measure.report import html_escape, write_csv_report, write_html_report
from icd.runtime import compare
from icd.runtime.runner import prepare_runner_context, resolve_runner
from icd.runtime.runners import mock_inference
from icd.utils.imports import load_object


def sample_runner(mode: str, context: Dict[str, Any]) -> Dict[str, float]:
    context.setdefault("calls", 0)
    context["calls"] += 1
    base = 0.00005 if mode == "iterative" else 0.0001
    return {"tokens": 8, "l2_hit_pct": 0.9 if mode == "iterative" else 0.8, "latency": base}


def test_resolve_runner_handles_callable_and_dotted_path() -> None:
    assert resolve_runner({"runner": sample_runner}) is sample_runner

    resolved = resolve_runner({"runner": "icd.runtime.runners:mock_inference"})
    assert callable(resolved)

    with pytest.raises(ValueError):
        resolve_runner({"runner": "json"})  # missing attribute
    with pytest.raises(ValueError):
        resolve_runner({"runner": "json:__doc__"})


def test_prepare_runner_context_returns_copy() -> None:
    original = {"a": 1, "nested": {"b": 2}}
    ctx = prepare_runner_context(**original)
    assert ctx == original
    ctx["a"] = 5
    assert original["a"] == 1


def test_mock_inference_generates_expected_outputs(monkeypatch) -> None:
    # Patch time.sleep to avoid real delay
    monkeypatch.setattr("time.sleep", lambda _: None)
    ctx = {"tokens": 4, "provide_l2": True, "provide_ept": True}
    iter_metrics = mock_inference("iterative", ctx)
    lin_metrics = mock_inference("linear", ctx)
    assert iter_metrics["tokens"] == 4
    assert iter_metrics["l2_hit_pct"] > lin_metrics["l2_hit_pct"]
    assert "ept_j_per_tok" in iter_metrics


def test_compare_decide_applies_latency_and_quality_gates() -> None:
    baseline = {
        "latency_ms": {"mean": 100.0},
        "l2_hit_pct": 0.8,
        "ept_j_per_tok": 1.0,
        "quality": {"metric": "perplexity", "after": 10.0},
        "acceptance": {"delta_J": -0.01},
    }
    trial = {
        "latency_ms": {"mean": 70.0},
        "l2_hit_pct": 0.9,
        "ept_j_per_tok": 0.8,
        "quality": {"metric": "perplexity", "after": 10.01},
        "acceptance": {"delta_J": -0.02},
    }
    verdict = compare.decide(baseline, trial, fixed_clock=True, eps_J=0.01)
    assert verdict["accepted"] is True
    assert verdict["rolled_back"] is False
    assert verdict.get("missing") == []

    # Quality regression should flip acceptance
    trial_quality = trial.copy()
    trial_quality["quality"] = {"metric": "perplexity", "after": 11.0}
    verdict_bad = compare.decide(baseline, trial_quality, fixed_clock=True, eps_J=0.01)
    assert verdict_bad["accepted"] is False
    assert verdict_bad.get("quality_ok") is False


def test_measure_latency_helpers() -> None:
    calls = {"count": 0}

    def fn() -> None:
        calls["count"] += 1

    summary = latency_mod.measure_latency(fn, repeats=3, warmup=2)
    samples = latency_mod.measure_latency_samples(fn, repeats=4, warmup=1)
    assert calls["count"] == 2 + 3 + 1 + 4
    assert len(samples) == 4
    assert "mean_ms" in summary


def test_measure_report_writers(tmp_path: Path) -> None:
    metrics = {"latency": {"mean": 1.23}, "status": "ok", "note": "<unsafe>"}
    csv_path = write_csv_report(str(tmp_path), metrics)
    html_path = write_html_report(str(tmp_path), metrics)

    assert Path(csv_path).exists()
    assert Path(html_path).exists()

    html_text = Path(html_path).read_text(encoding="utf-8")
    assert "&lt;unsafe&gt;" in html_text
    assert html_escape("a & b") == "a &amp; b"


def test_parse_l2_hit_from_section_json(tmp_path: Path) -> None:
    doc = {"children": [{"metrics": {"lts__t_sectors_hit_rate.pct": 87.5}}]}
    path = tmp_path / "ncu.json"
    path.write_text(json.dumps(doc), encoding="utf-8")

    res = parse_l2_hit_from_section_json(str(path))
    assert res["l2_hit_pct"] == pytest.approx(87.5)

    missing = parse_l2_hit_from_section_json(str(path.with_name("missing.json")))
    val = missing["l2_hit_pct"]
    assert val is not None and math.isnan(val)


def test_sample_power_series_falls_back_without_nvml() -> None:
    series = sample_power_series(seconds=0.01, hz=2)
    assert len(series) >= 1
    assert math.isnan(series[0]["power_w"])


def test_sample_power_series_with_stubbed_nvml(monkeypatch) -> None:
    perf_values = iter([0.0, 0.1, 0.2, 0.3])
    power_values = iter([120000, 125000, 130000])

    def fake_perf_counter() -> float:
        return next(perf_values)

    def fake_sleep(_: float) -> None:
        return None

    fake_nvml = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda idx: object(),
        nvmlDeviceGetPowerUsage=lambda handle: next(power_values),
        nvmlShutdown=lambda: None,
    )

    monkeypatch.setattr("time.perf_counter", fake_perf_counter)
    monkeypatch.setattr("time.sleep", fake_sleep)
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    series = sample_power_series(seconds=1.0, hz=3)
    assert len(series) == 3
    assert series[0]["power_w"] == pytest.approx(120.0)


def test_measure_ept_and_l2_stub_shapes() -> None:
    ept = measure_ept_stub(tokens=4)
    assert math.isnan(ept["ept_j_per_tok"]) and ept["tokens"] == 4

    stub = collect_l2_section_stub()
    assert "l2_tex__t_sector_hit_rate.pct" in stub


def test_load_object_validates_module_and_attribute() -> None:
    fn = load_object("icd.runtime.runner:prepare_runner_context")
    assert fn is prepare_runner_context

    with pytest.raises(ValueError):
        load_object("")
    with pytest.raises(ValueError):
        load_object("json")
    with pytest.raises(ValueError):
        load_object("json:missing_attr")


def test_errors_are_regular_exceptions() -> None:
    with pytest.raises(ConfigError):
        raise ConfigError("bad config")
    with pytest.raises(MeasureError):
        raise MeasureError("measurement failure")

def test_eval_quality_stubs_return_none() -> None:
    assert eval_ppl("model") is None
    assert eval_acc("model") is None
