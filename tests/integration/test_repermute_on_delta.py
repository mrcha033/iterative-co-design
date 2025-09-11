import json
import subprocess
import sys
from pathlib import Path


def test_repermute_on_delta_in_linear_mode(tmp_path: Path):
    out = tmp_path / "linear_delta"
    # linear mode but enable sparsity and repermute_on_delta=true should create perm_after.json
    cmd = [
        sys.executable,
        "-m",
        "icd.cli.main",
        "run",
        "-c",
        "configs/mock.json",
        "--override",
        "pipeline.mode=linear",
        "--override",
        "pipeline.repermute_on_delta=true",
        "--override",
        "transform.sparsity.enable=true",
        "--override",
        "transform.sparsity.rate=0.5",
        "--out",
        str(out),
    ]
    subprocess.check_call(cmd)
    assert (out / "perm_after.json").exists()
    # transform_meta should indicate delta_layout true and include trigger 'S'
    metrics = json.loads((out / "metrics.json").read_text())
    tmeta = metrics.get("transform_meta", {})
    assert tmeta.get("delta_layout") is True
    assert "S" in (tmeta.get("triggers") or [])

