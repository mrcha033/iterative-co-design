from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from scripts import repro_ablation


def test_run_ablation_invokes_expected_commands(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    collected: dict[str, object] = {}

    def fake_runner(cmd: repro_ablation.Command) -> None:
        commands.append(list(cmd))

    def fake_collector(run_dirs: Sequence[Path], archive: Path) -> None:
        collected["run_dirs"] = list(run_dirs)
        collected["archive"] = archive

    out_root = tmp_path / "runs"
    issued = repro_ablation.run_ablation(
        config=Path("configs/bert.json"),
        out_root=out_root,
        sparsity=[0.3],
        precision=["fp8", "int8"],
        sequence_length=[256],
        runner=fake_runner,
        collector=fake_collector,
    )

    assert len(commands) == 2, "Expected one command per ablation combination"
    assert [cmd[:3] for cmd in commands] == [[sys.executable, "-m", "icd.cli.main"]] * 2

    for run_dir, cmd in issued:
        assert run_dir.exists()
        assert f"--override" in cmd
        assert any(token == "--override" and "transform.sparsity.rate=0.3" in cmd[idx + 1] for idx, token in enumerate(cmd[:-1]))
        assert any("transform.quant.dtype=" in token for token in cmd)
        assert any("sequence_length=256" in token for token in cmd)

    expected_dirs = [entry[0] for entry in issued]
    assert collected["run_dirs"] == expected_dirs
    assert collected["archive"] == out_root / "ablation_artifacts.zip"

    combo_names = [path.name for path in expected_dirs]
    assert combo_names == ["s0p3_pfp8_seq256", "s0p3_pint8_seq256"]

