import json
import subprocess
import sys
from pathlib import Path


def test_pair_compare_and_acceptance_update(tmp_path: Path):
    out = tmp_path / "pair01"
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
    ])
    # compare.json exists and has required fields
    compare_path = out / "compare.json"
    assert compare_path.exists()
    verdict = json.loads(compare_path.read_text())
    assert "accepted" in verdict and "delta" in verdict
    # iter metrics acceptance should be updated with verdict
    iter_metrics = json.loads((out / "iter" / "metrics.json").read_text())
    acc = iter_metrics.get("acceptance", {})
    assert "accepted" in acc and isinstance(acc.get("accepted"), bool)
    assert "rolled_back" in acc
