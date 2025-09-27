import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

from icd.core.graph import CSRMatrix
from icd.runtime.orchestrator import RunArtifacts, run, run_pair


class DummyModel:
    pass


def dummy_model_loader(*args, **kwargs):  # pragma: no cover - exercised via orchestrator
    return DummyModel(), ("example",)


class ToyLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def correlation_model_loader(*args, **kwargs):  # pragma: no cover - orchestrator usage
    model = ToyLinear()
    example = torch.zeros(1, 4)
    return model, (example,)


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
        "transform": {"sparsity": {"enable": True, "rate": 0.0}},
    }


def test_run_produces_artifacts(tmp_path: Path, monkeypatch, orchestrator_config: Dict[str, Any]) -> None:
    call_order: List[int] = []

    def fake_fit_permutation(W, time_budget_s=0.0, refine_steps=0, cfg=None, seed=0, clusters=None, method=None):
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
    assert metrics["acceptance"]["incomplete"] is False
    assert metrics["acceptance"].get("missing") == []
    assert metrics["acceptance"]["retry_budget"] == 0
    assert metrics["acceptance"]["cache_notified"] is False
    assert metrics["acceptance"]["rca"]["status"] == "not_required"
    assert metrics["acceptance"]["active_perm"] == "perm_after.json"
    assert metrics["transform_meta"]["delta_layout"] is False
    assert (Path(artifacts.out_dir) / "report.csv").exists()
    assert (Path(artifacts.out_dir) / "report.html").exists()
    assert (Path(artifacts.out_dir) / "perm_active.json").exists()
    assert call_order == [0, 1]


def test_iterative_requires_transform_or_correlation(tmp_path: Path) -> None:
    cfg = {
        "report": {"out_dir": str(tmp_path / "guard")},
        "pipeline": {
            "mode": "iterative",
            "runner": "tests.unit.test_runtime_orchestrator:dummy_runner",
            "runner_context": {"tokens": 4},
        },
        "graph": {"source": "mock", "mock": {"d": 4, "blocks": 2, "noise": 0.0, "seed": 3}},
        "solver": {"time_budget_s": 0.01, "refine_steps": 1, "rng_seed": 0},
    }

    with pytest.raises(ValueError, match="pipeline.mode='iterative'"):
        run(cfg)


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

    verdicts = run_pair(cfg, str(tmp_path / "pair"))
    compare_path = Path(tmp_path / "pair" / "compare.json")
    assert compare_path.exists()
    compare_doc = json.loads(compare_path.read_text(encoding="utf-8"))
    assert verdicts["linear"]["accepted"] is True
    trial_metrics = json.loads(Path(tmp_path / "pair" / "iter" / "metrics.json").read_text(encoding="utf-8"))
    assert trial_metrics["acceptance"]["accepted"] is True
    assert "linear" in trial_metrics.get("comparisons", {})


def test_run_with_transforms_cache_and_measurements(tmp_path: Path, monkeypatch) -> None:
    call_order: List[int] = []

    def fake_fit_permutation(W, time_budget_s=0.0, refine_steps=0, cfg=None, seed=0, clusters=None, method=None):
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
    assert metrics1["acceptance"]["incomplete"] is True
    assert "l2_hit_pct" in metrics1["acceptance"]["note"]
    assert "l2_hit_pct" in metrics1["acceptance"].get("missing", [])
    assert metrics1["acceptance"]["active_perm"] == "perm_after.json"
    assert Path(artifacts1.out_dir, "perm_active.json").exists()

    # Second run hits cache; expect another invocation for re-permutation only
    cfg_second = json.loads(json.dumps(cfg))
    artifacts2 = run(cfg_second)
    metrics2 = json.loads(Path(artifacts2.metrics_path).read_text(encoding="utf-8"))
    assert metrics2["acceptance"]["delta_J"] <= 0
    assert len(call_order) == 3


def test_rollback_restores_perm_and_notifies_cache(tmp_path: Path, monkeypatch, orchestrator_config: Dict[str, Any]) -> None:
    cfg = json.loads(json.dumps(orchestrator_config))
    cfg["cache"] = {"enable": True, "cache_dir": str(tmp_path / "cache")}
    cfg["pipeline"]["no_measure"] = True

    calls: List[int] = []

    def fake_fit(W, time_budget_s=0.0, refine_steps=0, cfg=None, seed=0, clusters=None, method=None):
        size = W.shape[0]
        idx = len(calls)
        calls.append(idx)
        stats = {"J": 5.0, "C": 1.0, "Q": 1.0}
        if idx >= 1:
            stats = {"J": 5.0, "C": 1.1, "Q": 1.0}
        return list(range(size)), stats

    monkeypatch.setattr("icd.runtime.orchestrator.fit_permutation", fake_fit)

    artifacts = run(cfg)
    out_dir = Path(artifacts.out_dir)
    metrics = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))

    assert metrics["acceptance"]["rolled_back"] is True
    assert metrics["acceptance"]["rca"]["status"] == "pending"
    assert metrics["acceptance"]["active_perm"] == "perm_before.json"
    assert metrics["acceptance"]["cache_notified"] is True
    before = json.loads((out_dir / "perm_before.json").read_text(encoding="utf-8"))
    active = json.loads((out_dir / "perm_active.json").read_text(encoding="utf-8"))
    assert before["pi"] == active["pi"]
    cache_dir = Path(cfg["cache"]["cache_dir"])
    rollback_files = list(cache_dir.glob("*.rollback.json"))
    assert rollback_files, "expected rollback notice in cache"
def test_iterative_auto_enables_correlation(monkeypatch, tmp_path: Path) -> None:
    captured: Dict[str, Any] = {}

    def fake_collect(model, inputs, cfg):
        captured["collect"] = True
        matrix = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=torch.float32)
        return matrix, {"mode": "activation", "samples": cfg.samples}

    def fake_to_csr(matrix, cfg):
        return CSRMatrix(indptr=[0, 1, 2], indices=[1, 0], data=[0.5, 0.5], shape=(2, 2), meta={"source": "auto"})

    monkeypatch.setattr("icd.runtime.orchestrator.collect_correlations", fake_collect)
    monkeypatch.setattr("icd.runtime.orchestrator.correlation_to_csr", fake_to_csr)

    cfg = {
        "report": {"out_dir": str(tmp_path / "auto_corr")},
        "pipeline": {
            "mode": "iterative",
            "runner": "tests.unit.test_runtime_orchestrator:runner_tokens_only",
            "runner_context": {
                "tokens": 8,
                "model_loader": "tests.unit.test_runtime_orchestrator:correlation_model_loader",
            },
            "warmup_iter": 0,
            "repeats": 1,
        },
        "graph": {
            "source": "mock",
            "mock": {"d": 2, "blocks": 1, "noise": 0.0, "seed": 0},
            "loader": "tests.unit.test_runtime_orchestrator:correlation_model_loader",
        },
        "solver": {"time_budget_s": 0.01, "refine_steps": 1, "rng_seed": 0},
        "transform": {"sparsity": {"enable": True, "rate": 0.0}},
    }

    artifacts = run(cfg)
    metrics = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))

    assert captured.get("collect") is True
    assert metrics.get("correlation", {}).get("auto_enabled") is True
    triggers = metrics.get("transform_meta", {}).get("triggers", [])
    assert "CORR-auto" in triggers


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


def test_quant_pre_stage_updates_graph_and_w(tmp_path: Path, monkeypatch) -> None:
    class QuantModel(DummyModel):
        pass

    recorded: Dict[str, Any] = {}

    def loader(*args, **kwargs):
        model = QuantModel()
        recorded["loader_model"] = model
        return model, ("example",)

    def fake_apply_quant(model, dtype="int8", method="ptq-minmax"):
        recorded["quant_input"] = model
        quantized = QuantModel()
        quantized.quantized = True
        recorded["quant_output"] = quantized
        return quantized, {"delta_layout": True, "quant": {"dtype": dtype, "method": method}}

    class FakeW:
        def __init__(self, model):
            self._model = model
            self.shape = (1, 1)
            self.meta = {"source": "pytorch"}

        def nnz(self) -> int:
            return 1

    def fake_build_w(*, source: str, **kwargs):
        recorded["build_source"] = source
        recorded["build_model"] = kwargs.get("model")
        recorded["build_inputs"] = kwargs.get("example_inputs")
        return FakeW(kwargs.get("model"))

    def fake_save_w(path: str, W) -> None:
        recorded.setdefault("saved_w", path)

    def fake_fit_permutation(W, time_budget_s=0.0, refine_steps=0, cfg=None, seed=0, clusters=None, method=None):
        recorded["fit_w_model"] = getattr(W, "_model", None)
        return [0], {"J": 0.0, "C": 0.0, "Q": 0.0}

    monkeypatch.setattr("tests.unit.test_runtime_orchestrator.dummy_model_loader", loader)
    monkeypatch.setattr("icd.runtime.orchestrator.build_w", fake_build_w)
    monkeypatch.setattr("icd.runtime.orchestrator.save_w_npz", fake_save_w)
    monkeypatch.setattr("icd.runtime.orchestrator.fit_permutation", fake_fit_permutation)
    monkeypatch.setattr("icd.adapters.quant.apply_quant", fake_apply_quant)

    cfg = {
        "report": {"out_dir": str(tmp_path / "quant_pre")},
        "pipeline": {
            "mode": "iterative",
            "no_measure": True,
            "transform_stage": "pre",
            "post_transform_repermute": "never",
        },
        "graph": {
            "source": "pytorch",
            "loader": "tests.unit.test_runtime_orchestrator:dummy_model_loader",
        },
        "solver": {"time_budget_s": 0.01, "refine_steps": 1, "rng_seed": 0},
        "transform": {"quant": {"enable": True, "method": "ptq-minmax", "dtype": "int8"}},
    }

    run(cfg)

    assert recorded.get("quant_input") is recorded.get("loader_model")
    assert recorded.get("build_model") is recorded.get("quant_output")
    assert recorded.get("fit_w_model") is recorded.get("quant_output")
    assert recorded.get("build_inputs") == ("example",)

    metrics = json.loads((tmp_path / "quant_pre" / "metrics.json").read_text(encoding="utf-8"))
    tmeta = metrics.get("transform_meta", {})
    assert tmeta.get("delta_layout") is True
    assert any(meta.get("stage") == "pre" for meta in tmeta.get("metas", []))
    assert "Q" in tmeta.get("triggers", [])


def test_run_with_correlation_and_clustering(tmp_path: Path, monkeypatch) -> None:
    captured: Dict[str, Any] = {"clusters": []}

    def fake_collect(model, inputs, cfg):
        captured["collect_called"] = True
        matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        return matrix, {"mode": "activation", "samples": cfg.samples}

    def fake_to_csr(matrix, cfg):
        return CSRMatrix(indptr=[0, 1, 2], indices=[1, 0], data=[1.0, 1.0], shape=(2, 2), meta={"source": "test"})

    fake_clusters = [[0], [1]]

    def fake_cluster(W, cfg):
        captured["cluster_cfg"] = cfg
        cfg.last_meta = {
            "method": "spectral",
            "fallback_reason": "low_modularity",
            "runtime_s": 0.123,
            "modularity": 0.42,
        }
        return fake_clusters

    def fake_fit(W, time_budget_s=0.0, refine_steps=0, cfg=None, seed=0, clusters=None, method=None):
        captured["clusters"].append(clusters)
        return list(range(W.shape[0])), {
            "J": 1.0,
            "C": 0.0,
            "Q": 0.0,
            "clusters": len(clusters or []),
            "Q_cluster": 0.0,
            "Q_final": 0.0,
        }

    monkeypatch.setattr("icd.runtime.orchestrator.collect_correlations", fake_collect)
    monkeypatch.setattr("icd.runtime.orchestrator.correlation_to_csr", fake_to_csr)
    monkeypatch.setattr("icd.runtime.orchestrator.cluster_graph", fake_cluster)
    monkeypatch.setattr("icd.runtime.orchestrator.fit_permutation", fake_fit)

    cfg = {
        "report": {"out_dir": str(tmp_path / "corr")},
        "pipeline": {
            "mode": "iterative",
            "runner": "tests.unit.test_runtime_orchestrator:runner_tokens_only",
            "runner_context": {
                "tokens": 8,
                "model_loader": "tests.unit.test_runtime_orchestrator:correlation_model_loader",
            },
            "warmup_iter": 0,
            "repeats": 1,
        },
        "graph": {
            "source": "mock",
            "mock": {"d": 2, "blocks": 1, "noise": 0.0, "seed": 0},
            "correlation": {"enable": True, "samples": 1},
            "loader": "tests.unit.test_runtime_orchestrator:correlation_model_loader",
        },
        "solver": {
            "time_budget_s": 0.01,
            "refine_steps": 1,
            "rng_seed": 0,
            "clustering": {"method": "louvain", "rng_seed": 0},
        },
    }

    artifacts = run(cfg)
    metrics = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))

    assert captured.get("collect_called") is True
    assert captured["clusters"][-1] == fake_clusters
    assert metrics.get("correlation", {}).get("mode") == "activation"

    last_meta = captured["cluster_cfg"].last_meta
    assert last_meta["method"] == "spectral"
    assert last_meta["fallback_reason"] == "low_modularity"
    assert last_meta["modularity"] == pytest.approx(0.42)
    assert last_meta["runtime_s"] == pytest.approx(0.123)

    clustering_metrics = metrics.get("clustering", {})
    assert clustering_metrics.get("count") == len(fake_clusters)
    assert clustering_metrics.get("method") == "spectral"
    assert clustering_metrics.get("fallback_reason") == "low_modularity"
    assert clustering_metrics.get("modularity") == pytest.approx(0.42)
    assert clustering_metrics.get("runtime_s") == pytest.approx(0.123)


def test_reference_configs_pass_iterative_guard(tmp_path: Path, monkeypatch) -> None:
    def fake_collect(model, inputs, cfg):
        matrix = torch.tensor([[0.0, 0.4], [0.4, 0.0]], dtype=torch.float32)
        return matrix, {"mode": cfg.mode, "samples": cfg.samples}

    def fake_to_csr(matrix, cfg):
        return CSRMatrix(indptr=[0, 1, 2], indices=[1, 0], data=[1.0, 1.0], shape=(2, 2), meta={"source": "fake"})

    def fake_fit(W, time_budget_s=0.0, refine_steps=0, cfg=None, seed=0, clusters=None, method=None):
        size = W.shape[0]
        return list(range(size)), {"J": 1.0, "C": 0.0, "Q": 0.0, "clusters": len(clusters or [])}

    monkeypatch.setattr("icd.runtime.orchestrator.collect_correlations", fake_collect)
    monkeypatch.setattr("icd.runtime.orchestrator.correlation_to_csr", fake_to_csr)
    monkeypatch.setattr("icd.runtime.orchestrator.fit_permutation", fake_fit)

    repo_root = Path(__file__).resolve().parents[2]
    configs = repo_root / "configs"
    for name in ["bert.json", "mamba.json"]:
        cfg_doc = json.loads((configs / name).read_text(encoding="utf-8"))
        corr_cfg = cfg_doc.get("graph", {}).get("correlation")
        assert corr_cfg and corr_cfg.get("enable") is True

        out_dir = tmp_path / name.replace(".json", "")
        cfg_doc.setdefault("report", {})["out_dir"] = str(out_dir)

        pipeline = cfg_doc.setdefault("pipeline", {})
        pipeline.update(
            {
                "mode": "iterative",
                "runner": "tests.unit.test_runtime_orchestrator:runner_tokens_only",
                "runner_context": {
                    "tokens": 8,
                    "model_loader": "tests.unit.test_runtime_orchestrator:correlation_model_loader",
                },
                "warmup_iter": 0,
                "repeats": 1,
            }
        )

        gcfg = cfg_doc.setdefault("graph", {})
        gcfg["source"] = "mock"
        gcfg["mock"] = {"d": 2, "blocks": 1, "noise": 0.0, "seed": 0}
        gcfg["normalize"] = "sym"
        gcfg["loader"] = "tests.unit.test_runtime_orchestrator:correlation_model_loader"
        for key in ["loader_kwargs", "pytorch", "nnz_cap", "mock_kwargs"]:
            gcfg.pop(key, None)
        corr_cfg = gcfg.setdefault("correlation", corr_cfg or {})
        corr_cfg["enable"] = True
        corr_cfg.setdefault("samples", 1)

        solver = cfg_doc.setdefault("solver", {})
        solver.update(
            {
                "time_budget_s": 0.01,
                "refine_steps": 1,
                "rng_seed": 0,
                "clustering": {"enable": False},
            }
        )

        cfg_doc["measure"] = {"ncu_enable": False, "power_enable": False}

        artifacts = run(cfg_doc)
        metrics = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))
        assert metrics.get("correlation", {}).get("mode") == "activation"
