import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest

from icd.runtime.orchestrator import RunArtifacts, run, run_pair


class DummyModel:
    pass


def dummy_model_loader(*args, **kwargs):  # pragma: no cover - exercised via orchestrator
    return DummyModel(), ("example",)


def dummy_runner(mode: str, context: Dict[str, Any]) -> Dict[str, float]:
    context.setdefault("calls", 0)
    context["calls"] += 1
    return {"tokens": context.get("tokens", 16), "l2_hit_pct": 0.9, "ept_j_per_tok": 0.5}


def runner_tokens_only(mode: str, context: Dict[str, Any]) -> Dict[str, float]:
    context.setdefault("calls", 0)
    context["calls"] += 1
    tokens = int(context.get("tokens", 8))
    return {"tokens": tokens}


@pytest.fixture()
def orchestrator_config(tmp_path: Path):
    return {
        "report": {"out_dir": str(tmp_path / "run")},
        "pipeline": {
            "mode": "iterative",
            "runner": "tests.unit.test_runtime_orchestrator:dummy_runner",
            "runner_context": {"tokens": 16},
            "warmup_iter": 1,
            "repeats": 2,
        },
        "graph": {"source": "mock", "mock": {"d": 6, "blocks": 2, "noise": 0.0, "seed": 1}},
        "solver": {"time_budget_s": 0.01, "refine_steps": 2, "rng_seed": 0},
    }


def test_run_produces_artifacts(tmp_path: Path, monkeypatch, orchestrator_config: Dict[str, Any]) -> None:
    call_order: List[int] = []

    def fake_fit_permutation(W, time_budget_s=0.0, refine_steps=0, cfg=None, seed=0):
        size = W.shape[0]
        if not call_order:
            call_order.append(0)
            pi = list(range(size))
            stats = {"J": 5.0, "C": 1.0, "Q": 1.0, "improved": False}
        else:
            call_order.append(1)
            pi = list(reversed(range(size)))
            stats = {"J": 4.0, "C": 1.2, "Q": 1.1, "improved": True}
        return pi, stats

    monkeypatch.setattr("icd.runtime.orchestrator.fit_permutation", fake_fit_permutation)

    artifacts = run(orchestrator_config)

    assert isinstance(artifacts, RunArtifacts)
    assert Path(artifacts.config_lock_path).exists()
    assert Path(artifacts.metrics_path).exists()
    metrics = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))
    assert metrics["acceptance"]["accepted"] is True
    assert metrics["transform_meta"]["delta_layout"] is False
    assert (Path(artifacts.out_dir) / "report.csv").exists()
    assert (Path(artifacts.out_dir) / "report.html").exists()
    assert call_order == [0, 1]


def test_run_pair_generates_comparison(tmp_path: Path, monkeypatch) -> None:
    def fake_run(cfg: Dict[str, Any]) -> RunArtifacts:
        mode = cfg.get("pipeline", {}).get("mode")
        out_dir = Path(cfg.get("report", {}).get("out_dir", tmp_path / mode))
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "latency_ms": {"mean": 100.0 if mode == "linear" else 70.0, "p50": 0.0, "p95": 0.0, "ci95": 0.0},
            "l2_hit_pct": 0.82 if mode == "linear" else 0.88,
            "quality": {"metric": "perplexity", "after": 10.0},
            "acceptance": {"delta_J": -0.02},
            "env": {"fixed_clock": True},
        }
        metrics_path = out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        (out_dir / "config.lock.json").write_text("{}", encoding="utf-8")
        (out_dir / "run.log").write_text("{}", encoding="utf-8")
        return RunArtifacts(
            out_dir=str(out_dir),
            config_lock_path=str(out_dir / "config.lock.json"),
            run_log_path=str(out_dir / "run.log"),
            metrics_path=str(metrics_path),
        )

    monkeypatch.setattr("icd.runtime.orchestrator.run", fake_run)

    cfg = {
        "report": {"out_dir": str(tmp_path / "pair")},
        "pipeline": {"mode": "iterative"},
    }

    verdict = run_pair(cfg, str(tmp_path / "pair"))
    compare_path = Path(tmp_path / "pair" / "compare.json")
    assert compare_path.exists()
    compare_doc = json.loads(compare_path.read_text(encoding="utf-8"))
    assert verdict["accepted"] is True
    trial_metrics = json.loads(Path(tmp_path / "pair" / "iter" / "metrics.json").read_text(encoding="utf-8"))
    assert trial_metrics["acceptance"]["accepted"] is True


def test_run_with_transforms_cache_and_measurements(tmp_path: Path, monkeypatch) -> None:
    call_order: List[int] = []

    def fake_fit_permutation(W, time_budget_s=0.0, refine_steps=0, cfg=None, seed=0):
        size = W.shape[0]
        idx = len(call_order)
        call_order.append(idx)
        pi = list(range(size))
        stats = {"J": 5.0 - idx * 0.5, "C": 1.0 + idx, "Q": 1.0 + idx * 0.1, "improved": idx > 0}
        return pi, stats

    def fake_power_series(seconds: float, hz: int):
        return [
            {"t_s": 0.0, "power_w": 1.0},
            {"t_s": 0.25, "power_w": 1.2},
            {"t_s": 0.5, "power_w": 1.4},
        ]

    monkeypatch.setattr("icd.runtime.orchestrator.fit_permutation", fake_fit_permutation)
    monkeypatch.setattr("icd.measure.nvml_logger.sample_power_series", fake_power_series)

    cfg = {
        "report": {"out_dir": str(tmp_path / "run_transforms"), "formats": ["csv", "html"]},
        "pipeline": {
            "mode": "iterative",
            "runner": "tests.unit.test_runtime_orchestrator:runner_tokens_only",
            "runner_context": {"tokens": 32},
            "warmup_iter": 1,
            "repeats": 3,
            "fixed_clock": False,
        },
        "graph": {"source": "mock", "mock": {"d": 4, "blocks": 2, "noise": 0.0, "seed": 2}, "normalize": "sym"},
        "solver": {"time_budget_s": 0.01, "refine_steps": 1, "rng_seed": 1},
        "transform": {
            "sparsity": {"enable": True, "rate": 0.5},
            "quant": {"enable": True, "dtype": "int8"},
            "kv": {"enable": True, "block": 128},
        },
        "cache": {"enable": True, "cache_dir": str(tmp_path / "cache")},
        "measure": {"ncu_enable": True, "power_enable": True, "power_sample_hz": 4},
    }

    artifacts1 = run(cfg)
    metrics1 = json.loads(Path(artifacts1.metrics_path).read_text(encoding="utf-8"))
    assert Path(artifacts1.out_dir, "power.csv").exists()
    assert metrics1["transform_meta"]["delta_layout"] is True
    l2_val = metrics1["l2_hit_pct"]
    assert l2_val is not None and math.isnan(l2_val)  # derived from ncu stub
    assert metrics1["throughput_toks_s"] is not None

    # Second run hits cache; expect another invocation for re-permutation only
    cfg_second = json.loads(json.dumps(cfg))
    artifacts2 = run(cfg_second)
    metrics2 = json.loads(Path(artifacts2.metrics_path).read_text(encoding="utf-8"))
    assert metrics2["acceptance"]["delta_J"] <= 0
    assert len(call_order) == 3


def test_transform_stage_uses_loaded_model(tmp_path: Path, monkeypatch) -> None:
    recorded: Dict[str, Any] = {}

    def fake_apply_sparsity(model, **kwargs):
        recorded["model"] = model
        meta = {
            "delta_layout": True,
            "sparsity": {
                "type": kwargs.get("type"),
                "rate": kwargs.get("rate"),
                "method": kwargs.get("method", "l1_unstructured"),
            },
        }
        return model, meta

    monkeypatch.setattr("icd.adapters.sparsity.apply_sparsity", fake_apply_sparsity)

    cfg = {
        "report": {"out_dir": str(tmp_path / "transform_model")},
        "pipeline": {
            "mode": "iterative",
            "runner": "tests.unit.test_runtime_orchestrator:runner_tokens_only",
            "runner_context": {
                "tokens": 8,
                "model_loader": "tests.unit.test_runtime_orchestrator:dummy_model_loader",
            },
            "warmup_iter": 0,
            "repeats": 1,
        },
        "graph": {
            "source": "mock",
            "mock": {"d": 4, "blocks": 2, "noise": 0.0, "seed": 3},
            "normalize": "sym",
            "loader": "tests.unit.test_runtime_orchestrator:dummy_model_loader",
        },
        "solver": {"time_budget_s": 0.01, "refine_steps": 1, "rng_seed": 0},
        "transform": {"sparsity": {"enable": True, "rate": 0.5, "type": "unstructured"}},
    }

    run(cfg)

    model_obj = recorded.get("model")
    assert model_obj is not None
    assert model_obj.__class__.__name__ == "DummyModel"
