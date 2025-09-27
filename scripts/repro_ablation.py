"""Repro ablation harness for sparsity/precision/sequence sweeps."""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

Command = Sequence[str]
Runner = Callable[[Command], None]
Collector = Callable[[Sequence[Path], Path], None]


def _float_label(value: float) -> str:
    text = ("{:.3g}".format(value)).rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def build_combo_name(sparsity: float, precision: str, sequence_length: int) -> str:
    return f"s{_float_label(sparsity)}_p{precision.lower()}_seq{sequence_length}"


def build_overrides(
    sparsity: float,
    precision: str,
    sequence_length: int,
) -> List[str]:
    base = [
        "transform.sparsity.enable=true",
        f"transform.sparsity.rate={sparsity}",
        "transform.quant.enable=true",
        f"transform.quant.dtype={precision}",
        f"pipeline.runner_context.model_loader_kwargs.sequence_length={sequence_length}",
        f"graph.loader_kwargs.sequence_length={sequence_length}",
        f"graph.pytorch.max_len={sequence_length}",
        f"quality.max_length={sequence_length}",
    ]
    return base


def default_runner(cmd: Command) -> None:
    subprocess.run(cmd, check=True)


def default_collector(run_dirs: Sequence[Path], archive: Path) -> None:
    if not run_dirs:
        return
    collector_cmd = [
        sys.executable,
        "scripts/collect_artifacts.py",
        *[str(p) for p in run_dirs],
        "-o",
        str(archive),
    ]
    subprocess.run(collector_cmd, check=True)


def run_ablation(
    *,
    config: Path,
    out_root: Path,
    sparsity: Iterable[float],
    precision: Iterable[str],
    sequence_length: Iterable[int],
    runner: Runner = default_runner,
    collector: Collector = default_collector,
    dry_run: bool = False,
) -> List[Tuple[Path, Command]]:
    out_root.mkdir(parents=True, exist_ok=True)
    issued: List[Tuple[Path, Command]] = []
    run_dirs: List[Path] = []

    for s_val, p_val, seq_len in itertools.product(sparsity, precision, sequence_length):
        combo_name = build_combo_name(s_val, p_val, seq_len)
        run_dir = out_root / combo_name
        run_dir.mkdir(parents=True, exist_ok=True)
        overrides = build_overrides(s_val, p_val, seq_len)
        cmd: List[str] = [
            sys.executable,
            "-m",
            "icd.cli.main",
            "run",
            "-c",
            str(config),
            "--out",
            str(run_dir),
        ]
        for override in overrides:
            cmd.extend(["--override", override])
        issued.append((run_dir, tuple(cmd)))
        if not dry_run:
            runner(cmd)
        run_dirs.append(run_dir)

    archive = out_root / "ablation_artifacts.zip"
    if not dry_run:
        collector(run_dirs, archive)
    return issued


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run reproducibility ablation sweeps")
    ap.add_argument("--config", default="configs/bert.json", help="Base configuration path")
    ap.add_argument("--out-root", default="runs/ablation", help="Root directory for ablation runs")
    ap.add_argument("--sparsity", nargs="*", type=float, default=[0.3, 0.5, 0.7], help="Sparsity rates to sweep")
    ap.add_argument("--precision", nargs="*", default=["fp8", "int8"], help="Quantization precisions to sweep")
    ap.add_argument(
        "--sequence-length",
        nargs="*",
        type=int,
        default=[256, 1024],
        help="Sequence lengths to sweep",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = Path(args.config)
    out_root = Path(args.out_root)
    issued = run_ablation(
        config=config,
        out_root=out_root,
        sparsity=args.sparsity,
        precision=args.precision,
        sequence_length=args.sequence_length,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        for run_dir, cmd in issued:
            print(f"[DRY-RUN] {run_dir}: {' '.join(cmd)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
