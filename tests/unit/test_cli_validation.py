import json
import subprocess
import sys
from pathlib import Path


def test_cli_dry_run_ok(tmp_path: Path):
    # should exit 0
    rc = subprocess.call([
        sys.executable,
        "-m",
        "icd.cli.main",
        "run",
        "-c",
        "configs/mock.json",
        "--out",
        str(tmp_path / "out"),
        "--dry-run",
    ])
    assert rc == 0


def test_cli_dry_run_bad(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"graph": {"source": "mock"}, "solver": {}, "report": {"out_dir": str(tmp_path / 'o')}}))
    rc = subprocess.call([
        sys.executable,
        "-m",
        "icd.cli.main",
        "run",
        "-c",
        str(bad),
        "--out",
        str(tmp_path / "out"),
        "--dry-run",
    ])
    assert rc == 2

