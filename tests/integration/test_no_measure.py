import subprocess
import sys
from pathlib import Path


def test_no_measure_skips_reports(tmp_path: Path):
    out = tmp_path / "no_measure"
    cmd = [
        sys.executable,
        "-m",
        "icd.cli.main",
        "run",
        "-c",
        "configs/mock.json",
        "--no-measure",
        "--out",
        str(out),
    ]
    subprocess.check_call(cmd)
    # metrics and config exist, but no report files
    assert (out / "metrics.json").exists()
    assert (out / "config.lock.json").exists()
    assert not (out / "report.html").exists()
    assert not (out / "report.csv").exists()

