#!/usr/bin/env python3
"""Verify environment lock by checking CUDA clocks and seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify(manifest: dict[str, object]) -> bool:
    clocks = manifest.get("gpu", {}).get("clocks", {})
    seeds = manifest.get("seeds", {})
    return bool(clocks) and "training" in seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    if not verify(data):
        raise SystemExit("environment manifest incomplete")


if __name__ == "__main__":
    main()
