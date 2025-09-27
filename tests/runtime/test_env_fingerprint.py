from __future__ import annotations

import json
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

if "torch" not in sys.modules:  # pragma: no cover - exercised on CPU-only CI
    torch_stub = types.ModuleType("torch")

    class _TorchDType:  # minimal placeholder compatible with isinstance checks
        pass

    class _TorchTensor:  # noqa: D401 - simple sentinel class
        """Sentinel tensor type used for tests without torch."""

    class _TorchParameter:  # noqa: D401 - simple sentinel class
        """Sentinel parameter type used for tests without torch."""

    torch_stub.nn = types.SimpleNamespace(Parameter=_TorchParameter)
    torch_stub.Tensor = _TorchTensor
    torch_stub.dtype = _TorchDType
    torch_stub.float32 = _TorchDType()
    torch_stub.float64 = _TorchDType()
    torch_stub.float16 = _TorchDType()

    def _no_grad():
        def decorator(fn):
            return fn

        return decorator

    torch_stub.no_grad = _no_grad

    torch_stub.__spec__ = importlib.util.spec_from_loader("torch", loader=None)
    sys.modules["torch"] = torch_stub

from icd.runtime import orchestrator
from icd.runtime.orchestrator import run
from icd.utils.env import collect_env_fingerprint, load_env_fingerprint_schema


def _base_config(out_dir: Path, cache_dir: Path | None = None) -> dict:
    cfg: dict = {
        "pipeline": {"mode": "iterative", "no_measure": True},
        "graph": {"source": "mock", "mock": {"d": 8, "blocks": 2, "noise": 0.0, "seed": 0, "normalize": "sym"}},
        "solver": {"time_budget_s": 0.01, "refine_steps": 16, "k_blocks": 2, "rng_seed": 0},
        "transform": {"kv": {"enable": True, "block": 4, "drop": 0.0}},
        "report": {"out_dir": str(out_dir)},
        "measure": {},
    }
    if cache_dir is not None:
        cfg["cache"] = {"enable": True, "cache_dir": str(cache_dir)}
    return cfg


def _read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_collect_env_fingerprint_matches_schema():
    fingerprint = collect_env_fingerprint()
    schema = load_env_fingerprint_schema()

    # ``collect_env_fingerprint`` already validates but ensure schema stays aligned.
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(fingerprint, schema)

    assert "host" in fingerprint
    assert "gpu" in fingerprint
    assert isinstance(fingerprint.get("packages"), list)


def test_fingerprint_embedded_in_metrics_and_log(tmp_path):
    cfg = _base_config(tmp_path / "run")
    artifacts = run(cfg)

    metrics = _read_json(artifacts.metrics_path)
    env_doc = metrics.get("env", {}).get("fingerprint")
    assert env_doc is not None, "env fingerprint missing from metrics"

    schema = load_env_fingerprint_schema()
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(env_doc, schema)

    log_events = [json.loads(line) for line in Path(artifacts.run_log_path).read_text().splitlines() if line.strip()]
    env_events = [event for event in log_events if event.get("stage") == "ENV_FINGERPRINT"]
    assert env_events, "expected ENV_FINGERPRINT event in run.log"
    assert env_events[0]["meta"].get("fingerprint") == env_doc


def test_fingerprint_survives_cache_and_rollback(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cfg_first = _base_config(tmp_path / "run1", cache_dir=cache_dir)
    run(cfg_first)

    cfg_second = _base_config(tmp_path / "run2", cache_dir=cache_dir)

    def fake_acceptance(delta_J: float, epsilon_J: float, retry_budget: int) -> dict:
        return {"accepted": False, "rolled_back": True, "retry": False}

    monkeypatch.setattr(orchestrator, "evaluate_acceptance", fake_acceptance)

    artifacts = run(cfg_second)
    metrics = _read_json(artifacts.metrics_path)
    env_doc = metrics.get("env", {}).get("fingerprint")

    assert metrics.get("acceptance", {}).get("rolled_back") is True
    assert env_doc is not None

    schema = load_env_fingerprint_schema()
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(env_doc, schema)
