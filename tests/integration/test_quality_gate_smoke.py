import json
import subprocess
import sys
from pathlib import Path


def test_quality_field_exists_when_enabled(tmp_path: Path):
    out = tmp_path / "pair_q"
    cfg = "configs/mock.json"
    subprocess.check_call([
        sys.executable,
        "-m",
        "icd.cli.main",
        "pair",
        "-c",
        cfg,
        "--out",
        str(out),
        "--override",
        "eval.enable=true",
    ])
    it = json.loads((out / "iter" / "metrics.json").read_text())
    assert "quality" in it
