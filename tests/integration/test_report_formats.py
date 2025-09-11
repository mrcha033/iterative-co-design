import subprocess
import sys
from pathlib import Path


def test_report_formats_html_only(tmp_path: Path):
    out = tmp_path / "fmt_html"
    cmd = [
        sys.executable,
        "-m",
        "icd.cli.main",
        "run",
        "-c",
        "configs/mock.json",
        "--override",
        "report.formats=[\"html\"]",
        "--out",
        str(out),
    ]
    subprocess.check_call(cmd)
    assert (out / "report.html").exists()
    assert not (out / "report.csv").exists()


def test_report_formats_default_both(tmp_path: Path):
    out = tmp_path / "fmt_both"
    cmd = [
        sys.executable,
        "-m",
        "icd.cli.main",
        "run",
        "-c",
        "configs/mock.json",
        "--out",
        str(out),
    ]
    subprocess.check_call(cmd)
    assert (out / "report.html").exists()
    assert (out / "report.csv").exists()

