from pathlib import Path

from scripts.nsight_stub import run_stub


def test_run_stub_writes_payload(tmp_path: Path) -> None:
    out = tmp_path / "nsight.json"
    payload = run_stub(out, ["kernel_a", "kernel_b"])
    assert out.exists()
    assert payload["summary"]["launches"] == 2
