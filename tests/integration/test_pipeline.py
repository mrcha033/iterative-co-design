import json
import subprocess
import sys
from pathlib import Path


def _run(mode: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "icd.cli.main",
            "run",
            "-c",
            "configs/mock.json",
            "--override",
            f"pipeline.mode={mode}",
            "--out",
            str(out),
        ]
    )


def test_artifacts(tmp_path: Path):
    lin = tmp_path / "linear"
    itr = tmp_path / "iter"
    _run("linear", lin)
    _run("iterative", itr)
    for d in [lin, itr]:
        for f in ["metrics.json", "config.lock.json", "run.log", "report.csv", "report.html"]:
            assert (d / f).exists()
    m0 = json.loads((lin / "metrics.json").read_text())
    m1 = json.loads((itr / "metrics.json").read_text())
    assert m1["latency_ms"]["mean"] < m0["latency_ms"]["mean"] * 0.99  # smoke gate

