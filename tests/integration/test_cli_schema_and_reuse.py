import json
import subprocess
import sys
from pathlib import Path


def test_print_schema(tmp_path: Path):
    # Should print schema and exit 0
    out = subprocess.check_output([
        sys.executable,
        "-m",
        "icd.cli.main",
        "run",
        "--print-schema",
        "-c",
        "configs/mock.json",
        "--out",
        str(tmp_path / "o"),
    ])
    s = out.decode("utf-8")
    assert "pipeline" in s and "graph" in s


def test_reuse_perm_directory(tmp_path: Path):
    base = tmp_path / "base"
    trial = tmp_path / "trial"
    # First run produces a baseline with perm_before.json
    subprocess.check_call([
        sys.executable,
        "-m",
        "icd.cli.main",
        "run",
        "-c",
        "configs/mock.json",
        "--override",
        "pipeline.mode=linear",
        "--out",
        str(base),
    ])
    assert (base / "perm_before.json").exists()
    # Second run reuses the baseline perm and writes artifacts
    subprocess.check_call([
        sys.executable,
        "-m",
        "icd.cli.main",
        "run",
        "-c",
        "configs/mock.json",
        "--reuse-perm",
        str(base),
        "--out",
        str(trial),
    ])
    assert (trial / "metrics.json").exists()
    # Sanity: metrics JSON parses
    json.loads((trial / "metrics.json").read_text())

