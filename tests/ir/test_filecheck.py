import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MLIR_DIR = ROOT / "mlir"


def _run_opt(args, mlir_path: Path) -> str:
    cmd = [sys.executable, "-m", "scripts.icd_mlir_opt", *args, str(mlir_path)]
    out = subprocess.check_output(cmd)
    return out.decode("utf-8")


def _assert_check(output: str, checks: list[str], not_checks: list[str]):
    for pat in checks:
        assert re.search(pat, output), f"Missing CHECK: {pat}\nOutput:\n{output}"
    for pat in not_checks:
        assert not re.search(pat, output), f"Unexpected CHECK-NOT: {pat}\nOutput:\n{output}"


def test_attach_basic():
    p = MLIR_DIR / "attach_basic.mlir"
    out = _run_opt(["--icd-attach-metadata"], p)
    _assert_check(out, [r"icd\.layout_tag = \"icd/v1\"", r"icd\.layout_perm"], [])


def test_attach_idempotent():
    p = MLIR_DIR / "attach_idempotent.mlir"
    out = _run_opt(["--icd-attach-metadata", "--icd-attach-metadata"], p)
    _assert_check(out, [r"icd\.layout_tag = \"icd/v1\""], [r"icd\.layout_tag = \"icd/v1\".*icd\.layout_tag"])


def test_verify_perm_rank():
    p = MLIR_DIR / "verify_perm_rank.mlir"
    out = _run_opt(["--icd-attach-metadata", "--icd-verify"], p)
    _assert_check(out, [r"icd\.layout_perm", r"icd\.metrics"], [])


def test_no_tensors_noop():
    p = MLIR_DIR / "no_tensors_noop.mlir"
    out = _run_opt(["--icd-attach-metadata", "--icd-verify"], p)
    _assert_check(out, [r"module"], [])


def test_attach_verify_mix():
    p = MLIR_DIR / "attach_verify_mix.mlir"
    out = _run_opt(["--icd-attach-metadata", "--icd-verify"], p)
    _assert_check(out, [r"icd\.layout_perm", r"icd\.metrics"], [])

