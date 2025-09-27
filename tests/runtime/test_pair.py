import json
from pathlib import Path
from typing import Dict

import pytest

pytest.importorskip("torch")

from icd.runtime.orchestrator import RunArtifacts, run_pair


def _fake_run_factory(tmp_path: Path):
    def _fake_run(cfg: Dict[str, object]) -> RunArtifacts:
        pipeline = cfg.get("pipeline", {}) or {}
        solver_cfg = cfg.get("solver", {}) or {}
        mode = pipeline.get("mode", "linear")
        solver_method = solver_cfg.get("method")
        out_dir = Path(cfg.get("report", {}).get("out_dir", tmp_path / mode))
        out_dir.mkdir(parents=True, exist_ok=True)

        if mode == "iterative":
            latency = 70.0
        elif solver_method == "louvain":
            latency = 95.0
        elif solver_method == "memory_aware":
            latency = 90.0
        else:
            latency = 110.0

        metrics = {
            "latency_ms": {"mean": latency, "p50": latency, "p95": latency, "ci95": 0.0},
            "latency_ms_mean": latency,
            "l2_hit_pct": 0.85,
            "ept_j_per_tok": 0.45,
            "acceptance": {"delta_J": -0.05, "epsilon_J": 0.01},
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

    return _fake_run


def test_run_pair_handles_multiple_baselines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_run = _fake_run_factory(tmp_path)
    monkeypatch.setattr("icd.runtime.orchestrator.run", fake_run)

    cfg: Dict[str, object] = {
        "report": {"out_dir": str(tmp_path / "pair")},
        "pipeline": {"mode": "iterative"},
        "pair": {"baseline_methods": ["linear", "louvain", "memory_aware"]},
    }

    verdicts = run_pair(cfg, str(tmp_path / "pair"))

    assert set(verdicts.keys()) == {"linear", "louvain", "memory_aware"}
    assert Path(tmp_path / "pair" / "compare.json").exists()
    assert Path(tmp_path / "pair" / "compare_louvain.json").exists()
    assert Path(tmp_path / "pair" / "compare_memory_aware.json").exists()

    summary_path = Path(tmp_path / "pair" / "pairwise_summary.json")
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    modes = {entry["mode"]: entry for entry in summary}
    assert "linear" in modes
    assert "linear:louvain" in modes
    assert "linear:memory_aware" in modes
    assert "iterative" in modes
    assert modes["linear"]["baseline_method"] == "linear"
    assert modes["linear:louvain"]["baseline_method"] == "louvain"

    iter_metrics = json.loads(
        Path(tmp_path / "pair" / "iter" / "metrics.json").read_text(encoding="utf-8")
    )
    assert set(iter_metrics.get("comparisons", {}).keys()) == {
        "linear",
        "louvain",
        "memory_aware",
    }
    assert iter_metrics.get("baseline_methods") == ["linear", "louvain", "memory_aware"]

    for method in ["linear", "louvain", "memory_aware"]:
        metrics_path = (
            Path(tmp_path / "pair" / ("linear" if method == "linear" else f"baseline_{method}"))
            / "metrics.json"
        )
        doc = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert doc.get("baseline_method") == method
