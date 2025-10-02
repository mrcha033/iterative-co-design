import json
from pathlib import Path
from typing import Any, Dict

import pytest

from icd.cli.main import main, parse_override, deep_update
from icd.errors import ConfigError


@pytest.fixture()
def base_config(tmp_path: Path) -> Path:
    cfg: Dict[str, Any] = {
        "pipeline": {"mode": "linear"},
        "graph": {"source": "mock", "mock": {"d": 4, "blocks": 2, "noise": 0.0, "seed": 0}},
        "solver": {"time_budget_s": 0.01, "refine_steps": 1, "rng_seed": 0},
        "report": {"out_dir": str(tmp_path / "out")},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_parse_override_creates_nested_dict() -> None:
    override = parse_override("pipeline.repeats=3")
    assert override == {"pipeline": {"repeats": 3}}

    override = parse_override("graph.mock.d=16")
    assert override["graph"]["mock"]["d"] == 16


def test_parse_override_falls_back_to_string() -> None:
    override = parse_override("pipeline.mode=iterative")
    assert override["pipeline"]["mode"] == "iterative"


def test_deep_update_merges_nested_maps() -> None:
    base = {"pipeline": {"mode": "linear", "repeats": 1}}
    updated = deep_update(base, {"pipeline": {"repeats": 4, "warmup_iter": 2}})
    assert updated["pipeline"]["repeats"] == 4
    assert updated["pipeline"]["warmup_iter"] == 2


def test_main_run_invokes_pipeline(tmp_path: Path, base_config: Path, monkeypatch, capsys) -> None:
    calls: Dict[str, Any] = {}

    def fake_run(cfg: Dict[str, Any]) -> None:
        calls["cfg"] = cfg

    monkeypatch.setattr("icd.cli.main.run_pipeline", fake_run)

    out_dir = tmp_path / "run"
    reuse_path = tmp_path / "perm.json"
    reuse_path.write_text("{}", encoding="utf-8")
    rc = main([
        "run",
        "-c",
        str(base_config),
        "--out",
        str(out_dir),
        "--override",
        "pipeline.repeats=2",
        "--no-measure",
        "--reuse-perm",
        str(reuse_path),
    ])

    assert rc == 0
    assert "cfg" in calls
    cfg = calls["cfg"]
    assert cfg["pipeline"]["repeats"] == 2
    assert cfg["pipeline"]["no_measure"] is True
    assert cfg["pipeline"]["reuse_perm"] == str(reuse_path)
    assert cfg["report"]["out_dir"] == str(out_dir)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_main_run_dry_run_reports_issues(tmp_path: Path, monkeypatch, capsys) -> None:
    cfg = {"graph": {}, "solver": {}, "report": {"out_dir": str(tmp_path / "out")}}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    rc = main(["run", "-c", str(path), "--out", str(tmp_path / "out"), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "Missing pipeline" in out


def test_main_run_dry_run_success(tmp_path: Path, base_config: Path, capsys) -> None:
    rc = main(["run", "-c", str(base_config), "--out", str(tmp_path / "out"), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Config OK" in out


def test_main_run_invalid_config_returns_error(tmp_path: Path, base_config: Path, monkeypatch, capsys) -> None:
    def fake_run(cfg: Dict[str, Any]) -> None:  # pragma: no cover - defensive
        raise AssertionError("run should not be called")

    monkeypatch.setattr("icd.cli.main.run_pipeline", fake_run)

    cfg = json.loads(base_config.read_text(encoding="utf-8"))
    cfg["pipeline"]["mode"] = "unsupported"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    rc = main(["run", "-c", str(path), "--out", str(tmp_path / "out")])
    err = capsys.readouterr().out
    assert rc == 2
    assert "pipeline.mode" in err


def test_main_pair_invokes_pair_runner(tmp_path: Path, base_config: Path, monkeypatch) -> None:
    called = {}

    def fake_pair(cfg: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
        called["cfg"] = cfg
        called["out"] = out_dir
        return {"accepted": True}

    monkeypatch.setattr("icd.cli.main.run_pipeline_pair", fake_pair)

    out_root = tmp_path / "pair"
    rc = main([
        "pair",
        "-c",
        str(base_config),
        "--out",
        str(out_root),
        "--override",
        "pipeline.repeats=3",
    ])
    assert rc == 0
    assert called["out"] == str(out_root)
    assert called["cfg"]["pipeline"]["repeats"] == 3


def test_main_print_schema_fallback(tmp_path: Path, base_config: Path, capsys) -> None:
    rc = main([
        "run",
        "-c",
        str(base_config),
        "--out",
        str(tmp_path / "out"),
        "--print-schema",
    ])
    output = capsys.readouterr().out
    assert rc == 0
    # Fallback schema prints pipeline + graph keys
    assert "pipeline" in output
    assert "graph" in output


def test_main_print_schema_real_file(tmp_path: Path, base_config: Path, capsys, monkeypatch) -> None:
    import io

    def fake_exists(path: str) -> bool:
        return path.endswith("run_config.schema.json")

    real_open = open

    def fake_open(path: str, mode: str = "r", encoding: str | None = None):
        if path.endswith("run_config.schema.json"):
            return io.StringIO('{"$schema": "fake"}')
        return real_open(path, mode, encoding=encoding)

    monkeypatch.setattr("icd.cli.main.os.path.exists", fake_exists)
    monkeypatch.setattr("builtins.open", fake_open)

    rc = main([
        "run",
        "-c",
        str(base_config),
        "--out",
        str(tmp_path / "out"),
        "--print-schema",
    ])
    output = capsys.readouterr().out
    assert rc == 0
    assert "$schema" in output

def test_main_pair_print_schema(tmp_path: Path, base_config: Path, capsys) -> None:
    rc = main([
        "pair",
        "-c",
        str(base_config),
        "--out",
        str(tmp_path / "out"),
        "--print-schema",
    ])
    output = capsys.readouterr().out
    assert rc == 0
    assert "pipeline" in output


def test_main_pair_dry_run_reports_issues(tmp_path: Path, capsys) -> None:
    cfg = {"graph": {}, "solver": {}, "report": {"out": ""}}
    path = tmp_path / "pair_bad.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    rc = main([
        "pair",
        "-c",
        str(path),
        "--out",
        str(tmp_path / "out"),
        "--dry-run",
    ])
    output = capsys.readouterr().out
    assert rc == 2
    assert "Missing pipeline" in output


def test_main_pair_dry_run_success(tmp_path: Path, base_config: Path, capsys) -> None:
    rc = main([
        "pair",
        "-c",
        str(base_config),
        "--out",
        str(tmp_path / "out"),
        "--dry-run",
    ])
    output = capsys.readouterr().out
    assert rc == 0
    assert "Config OK" in output


def test_main_run_dry_run_trace_graph_errors(tmp_path: Path, capsys) -> None:
    cfg = {
        "pipeline": {"mode": "linear"},
        "graph": {"source": "trace", "normalize": "bad", "trace": ""},
        "solver": {"time_budget_s": 0.1},
        "report": {"out_dir": str(tmp_path / "out")},
    }
    path = tmp_path / "trace_bad.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    rc = main(["run", "-c", str(path), "--out", str(tmp_path / "out"), "--dry-run"])
    output = capsys.readouterr().out
    assert rc == 2
    assert "graph.trace" in output
    assert "graph.normalize" in output


def test_main_run_dry_run_measure_and_cache_types(tmp_path: Path, capsys) -> None:
    cfg = {
        "pipeline": {"mode": "linear"},
        "graph": {"source": "mock", "mock": {"d": 4, "blocks": 2, "noise": 0.0, "seed": 0}},
        "solver": {"time_budget_s": 0.1},
        "report": {"out_dir": str(tmp_path / "out")},
        "measure": "invalid",
        "cache": "invalid",
    }
    path = tmp_path / "measure_bad.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    rc = main(["run", "-c", str(path), "--out", str(tmp_path / "out"), "--dry-run"])
    output = capsys.readouterr().out
    assert rc == 2
    assert "measure must be" in output
    assert "cache must be" in output

def test_main_run_reports_many_issues(tmp_path: Path, capsys) -> None:
    cfg = {
        "pipeline": {
            "mode": "diag",
            "runner": 123,
            "runner_context": "nope",
            "repeats": 0,
            "warmup_iter": -1,
            "fixed_clock": "yes",
        },
        "graph": {"source": "mock", "normalize": "bad", "mock": {}},
        "solver": {"time_budget_s": "fast", "refine_steps": -5},
        "report": {"out_dir": "", "formats": ["pdf"]},
        "measure": {
            "ncu_enable": "maybe",
            "power_sample_hz": -1,
            "tvm_enable": "yes",
            "tvm_trials": "ten",
            "tvm_repeats": 0,
            "tvm_warmup": "zero",
            "tvm_target": 5,
            "tvm_log": 0,
            "tvm_artifacts_dir": False,
        },
        "cache": {"enable": True},
    }
    path = tmp_path / "invalid_many.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    rc = main(["run", "-c", str(path), "--out", str(tmp_path / "out"), "--dry-run"])
    output = capsys.readouterr().out
    assert rc == 2
    assert "pipeline.mode" in output
    assert "pipeline.runner" in output
    assert "graph.normalize" in output
    assert "measure.ncu_enable" in output
    assert "cache.cache_dir" in output


def test_main_calibrate_stub(capsys) -> None:
    rc = main(["calibrate"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Calibration CLI is stubbed" in out


def test_main_validate_invokes_validation(tmp_path: Path, monkeypatch) -> None:
    class FakeResult:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    captured = {}

    def fake_run_full_validation(cfg):
        captured["cfg"] = cfg
        return FakeResult(returncode=3)

    monkeypatch.setattr("icd.cli.main.run_full_validation", fake_run_full_validation)

    rc = main([
        "validate",
        "--out",
        str(tmp_path / "val"),
        "--device",
        "cpu",
        "--models",
        "bert",
        "--num-permutations",
        "2",
        "--quick",
        "--skip-matrix",
    ])
    assert rc == 3
    cfg = captured["cfg"]
    assert cfg.device == "cpu"
    assert cfg.models == ["bert"]
    assert cfg.quick is True and cfg.skip_matrix is True


def test_main_run_handles_config_error(tmp_path: Path, base_config: Path, monkeypatch, capsys) -> None:
    def raise_config_error(cfg):
        raise ConfigError("boom")

    monkeypatch.setattr("icd.cli.main.run_pipeline", raise_config_error)

    rc = main([
        "run",
        "-c",
        str(base_config),
        "--out",
        str(tmp_path / "out"),
    ])
    out = capsys.readouterr().out
    assert rc == 2
    assert "ConfigError" in out
