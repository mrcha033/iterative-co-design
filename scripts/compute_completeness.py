#!/usr/bin/env python3
"""Compute completeness score from a manifest of gate outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compute_score(manifest: dict[str, object]) -> float:
    coverage = manifest.get("coverage", {})
    latency = float(coverage.get("latency", 0.0))
    quality = float(coverage.get("quality", 0.0))
    energy = float(coverage.get("energy", 0.0))
    return 0.4 * latency + 0.3 * quality + 0.3 * energy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    print(f"{compute_score(data):.3f}")


if __name__ == "__main__":
    main()
