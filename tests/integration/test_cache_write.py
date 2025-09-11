import json
import subprocess
import sys
from pathlib import Path


def test_cache_writes_artifacts(tmp_path: Path):
    out = tmp_path / "cache_run"
    cache_dir = tmp_path / ".icd_cache"
    cmd = [
        sys.executable,
        "-m",
        "icd.cli.main",
        "run",
        "-c",
        "configs/mock.json",
        "--override",
        "cache.enable=true",
        "--override",
        f"cache.cache_dir={json.dumps(str(cache_dir))}",
        "--out",
        str(out),
    ]
    subprocess.check_call(cmd)
    # cache dir should contain two files with .perm_before.json and .stats_before.json suffixes
    files = {p.name for p in cache_dir.glob("*")}
    assert any(name.endswith(".perm_before.json") for name in files)
    assert any(name.endswith(".stats_before.json") for name in files)

