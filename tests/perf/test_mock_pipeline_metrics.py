import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

from icd.runtime.orchestrator import run

pytestmark = pytest.mark.perf


def _mock_config(out_dir: Path) -> Dict[str, Any]:
    context = {
        "tokens": 32,
        "iter_delay_s": 0.004,
        "linear_delay_s": 0.006,
        "iter_l2_hit": 0.91,
        "linear_l2_hit": 0.83,
        "iter_ept": 0.87,
        "linear_ept": 0.99,
    }
    return {
        "report": {"out_dir": str(out_dir)},
        "pipeline": {
            "mode": "iterative",
            "runner": "icd.runtime.runners.mock_inference",
            "runner_context": context,
            "warmup_iter": 1,
            "repeats": 5,
            "fixed_clock": True,
        },
        "graph": {"source": "mock", "mock": {"d": 6, "blocks": 2, "noise": 0.0, "seed": 2}},
        "solver": {"time_budget_s": 0.01, "refine_steps": 4, "rng_seed": 0},
        "transform": {"sparsity": {"enable": True, "rate": 0.0}},
        "measure": {"ncu_enable": False, "power_enable": False},
    }


def test_mock_pipeline_reports_latency_l2_and_ept(tmp_path: Path, monkeypatch) -> None:
    cfg = _mock_config(tmp_path / "perf_run")
    cfg["pipeline"]["runner_context"].update({"provide_l2": True, "provide_ept": True})

    calls: List[int] = []

    def fake_fit_permutation(W, time_budget_s=0.0, refine_steps=0, cfg=None, seed=0, clusters=None, method=None):
        size = W.shape[0]
        idx = len(calls)
        calls.append(idx)
        base_stats = {"J": 5.0, "C": 1.0, "Q": 1.0}
        improved_stats = {"J": 4.6, "C": 1.1, "Q": 1.0}
        stats = base_stats if idx == 0 else improved_stats
        permutation = list(range(size)) if idx == 0 else list(reversed(range(size)))
        return permutation, stats

    monkeypatch.setattr("icd.runtime.orchestrator.fit_permutation", fake_fit_permutation)

    artifacts = run(cfg)
    metrics_path = Path(artifacts.metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert metrics["acceptance"]["accepted"] is True
    assert metrics["acceptance"]["missing"] == []

    expected_delay_ms = cfg["pipeline"]["runner_context"]["iter_delay_s"] * 1000.0 * 0.98
    assert metrics["latency_ms"]["mean"] == pytest.approx(expected_delay_ms, rel=0.3)
    assert metrics["latency_ms"]["p95"] >= metrics["latency_ms"]["p50"]

    assert metrics["l2_hit_pct"] == pytest.approx(cfg["pipeline"]["runner_context"]["iter_l2_hit"])
    assert metrics["ept_j_per_tok"] == pytest.approx(cfg["pipeline"]["runner_context"]["iter_ept"])

    tokens = cfg["pipeline"]["runner_context"]["tokens"]
    expected_throughput = tokens * 1000.0 / metrics["latency_ms"]["mean"]
    assert metrics["throughput_toks_s"] == pytest.approx(expected_throughput, rel=1e-6)

    assert calls == [0, 1]
