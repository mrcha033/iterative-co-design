import json
import sys
from pathlib import Path
import subprocess


def _run(mode: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        sys.executable, "-m", "icd.cli.main", "run",
        "-c", "configs/mock.json",
        "--override", f"pipeline.mode={mode}",
        "--override", "measure.ncu_enable=false",
        "--override", "measure.power_enable=false",
        "--override", "pipeline.runner_context={\"tokens\":256,\"provide_l2\":false,\"provide_ept\":false}",
        "--out", str(out),
    ])


def test_metrics_have_nulls_when_disabled(tmp_path: Path):
    out = tmp_path / "ci"
    _run("iterative", out)
    metrics = json.loads((out / "metrics.json").read_text())
    assert "l2_hit_pct" in metrics and metrics["l2_hit_pct"] is None
    assert "ept_j_per_tok" in metrics and metrics["ept_j_per_tok"] is None
