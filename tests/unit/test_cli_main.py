import json
from pathlib import Path
from typing import Any, Dict

import pytest

from icd.cli.main import main, parse_override, deep_update


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
    rc = main([
        "run",
        "-c",
        str(base_config),
        "--out",
        str(out_dir),
        "--override",
        "pipeline.repeats=2",
        "--no-measure",
    ])

    assert rc == 0
    assert "cfg" in calls
    cfg = calls["cfg"]
    assert cfg["pipeline"]["repeats"] == 2
    assert cfg["pipeline"]["no_measure"] is True
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
    ])
    assert rc == 0
    assert called["out"] == str(out_root)


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
        "graph": {"source": "trace", "normalize": "bad", "trace": ""},
        "solver": {"time_budget_s": "fast", "refine_steps": -5},
        "report": {"out_dir": "", "formats": ["pdf"]},
        "measure": "bad",
        "cache": "bad",
    }
    path = tmp_path / "invalid_many.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    rc = main(["run", "-c", str(path), "--out", str(tmp_path / "out"), "--dry-run"])
    output = capsys.readouterr().out
    assert rc == 2
    assert "pipeline.mode" in output
    assert "pipeline.runner" in output
    assert "graph.trace" in output
    assert "measure must be" in output
