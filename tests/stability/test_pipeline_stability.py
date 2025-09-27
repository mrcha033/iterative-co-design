import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

from icd.runtime.orchestrator import run, run_pair

pytestmark = pytest.mark.stability


def _base_config(out_dir: Path) -> Dict[str, Any]:
    context = {
        "tokens": 24,
        "iter_delay_s": 0.004,
        "linear_delay_s": 0.006,
        "iter_l2_hit": 0.88,
        "linear_l2_hit": 0.80,
        "iter_ept": 1.05,
        "linear_ept": 0.95,
    }
    return {
        "report": {"out_dir": str(out_dir)},
        "pipeline": {
            "mode": "iterative",
            "runner": "icd.runtime.runners.mock_inference",
            "runner_context": context,
            "warmup_iter": 1,
            "repeats": 4,
            "fixed_clock": True,
        },
        "graph": {"source": "mock", "mock": {"d": 6, "blocks": 2, "noise": 0.0, "seed": 3}},
        "solver": {"time_budget_s": 0.01, "refine_steps": 4, "rng_seed": 0},
        "transform": {"sparsity": {"enable": True, "rate": 0.0}},
        "measure": {"ncu_enable": False, "power_enable": False},
    }


def test_missing_measurements_mark_acceptance_incomplete(tmp_path: Path, monkeypatch) -> None:
    cfg = _base_config(tmp_path / "missing_metrics")
    cfg["measure"].update({"ncu_enable": True, "power_enable": True, "power_sample_hz": 5})
    cfg["pipeline"]["runner_context"].update({"provide_l2": False, "provide_ept": False})

    calls: List[int] = []

    def fake_fit_permutation(W, time_budget_s=0.0, refine_steps=0, cfg=None, seed=0, clusters=None):
        size = W.shape[0]
        idx = len(calls)
        calls.append(idx)
        base_stats = {"J": 5.0, "C": 1.0, "Q": 1.0}
        improved_stats = {"J": 4.7, "C": 1.1, "Q": 1.0}
        stats = base_stats if idx == 0 else improved_stats
        return list(range(size)), stats

    monkeypatch.setattr("icd.runtime.orchestrator.fit_permutation", fake_fit_permutation)

    artifacts = run(cfg)
    metrics = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))

    assert metrics["acceptance"]["accepted"] is True
    assert metrics["acceptance"]["incomplete"] is True
    missing = set(metrics["acceptance"]["missing"])
    assert {"l2_hit_pct", "ept_j_per_tok"}.issubset(missing)

    gates_missing = set(metrics["gates"].get("missing", []))
    assert "iter.l2_pp" in gates_missing
    assert "iter.ept_rel" in gates_missing
    assert metrics["gates"]["status"].get("iter.l2_pp") is None
    assert metrics["gates"]["status"].get("iter.ept_rel") is None

    assert calls == [0, 1]


def test_retry_and_gate_failure_trigger_rollback(tmp_path: Path, monkeypatch) -> None:
    cfg = _base_config(tmp_path / "retry_rollback")
    cfg["pipeline"]["runner_context"].update({"provide_l2": True, "provide_ept": True})
    cfg["rollback"] = {"epsilon_J": 0.01, "retry_budget": 1}
    cfg["gates"] = {"iter.latency_rel": -0.5}

    calls: List[int] = []

    def fake_fit_permutation(W, time_budget_s=0.0, refine_steps=0, cfg=None, seed=0, clusters=None):
        size = W.shape[0]
        idx = len(calls)
        calls.append(idx)
        if idx == 0:
            stats = {"J": 5.0, "C": 1.0, "Q": 1.0}
        elif idx == 1:
            stats = {"J": 5.0, "C": 1.0, "Q": 1.0}
        else:
            stats = {"J": 5.2, "C": 1.0, "Q": 1.0}
        return list(range(size)), stats

    monkeypatch.setattr("icd.runtime.orchestrator.fit_permutation", fake_fit_permutation)

    out_dir = tmp_path / "pair_run"
    comparison = run_pair(cfg, str(out_dir))
    trial_metrics = json.loads((out_dir / "iter" / "metrics.json").read_text(encoding="utf-8"))

    assert comparison["accepted"] is False
    assert comparison["rolled_back"] is True

    acceptance = trial_metrics["acceptance"]
    assert acceptance["rolled_back"] is True
    assert acceptance["retry_budget"] == 1
    assert acceptance["missing"] == []
    assert acceptance["note"] == "complete"
    assert acceptance["delta_J"] > 0

    gates = trial_metrics["gates"]
    assert gates["status"].get("iter.latency_rel") is False
    assert "iter.latency_rel" in gates.get("thresholds", {})

    assert calls == [0, 1, 2]
