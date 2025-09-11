#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


ARTIFACT_FILES = {
    "metrics.json",
    "report.html",
    "report.csv",
    "ncu.json",
    "power.csv",
    "config.lock.json",
    "run.log",
}


def add_dir(z: ZipFile, run_dir: Path, rel_root: Path) -> None:
    for name in ARTIFACT_FILES:
        p = run_dir / name
        if p.exists():
            arc = str((rel_root / run_dir.name / name).as_posix())
            z.write(str(p), arcname=arc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="Run directories to collect")
    ap.add_argument("-o", "--out", required=True, help="Output zip path")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out, "w", ZIP_DEFLATED) as z:
        for r in args.runs:
            rp = Path(r)
            if rp.exists() and rp.is_dir():
                add_dir(z, rp, Path("runs"))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

