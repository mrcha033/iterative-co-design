import json
from pathlib import Path

from icd.core.graph import build_w


def test_w_meta_and_ops(tmp_path: Path):
    # Build mock W (meta files only written for pytorch, so validate CSR stats here)
    W = build_w(source="mock", D=64, blocks=4, noise=0.02, seed=1)
    assert W.nnz() > 0
    assert all(v >= 0 for v in W.data)

