#!/usr/bin/env python3
"""CLI utility to record activation capture metadata for IASP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch

from icd.graph.correlation import CorrelationConfig, collect_correlations


def _dummy_inputs(feature_dim: int, batch: int) -> Iterable[torch.Tensor]:
    torch.manual_seed(0)
    for _ in range(batch):
        yield torch.randn(4, feature_dim)


def run_capture(out_dir: Path, feature_dim: int, samples: int) -> dict:
    model = torch.nn.Sequential(torch.nn.Linear(feature_dim, feature_dim), torch.nn.ReLU())
    cfg = CorrelationConfig(samples=samples, layers=["0"], transfer_batch_size=2, whiten=True)
    matrix, meta = collect_correlations(model, list(_dummy_inputs(feature_dim, samples)), cfg=cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "1.0",
        "captures": [
            {
                "layer": entry["name"],
                "samples": entry["count"],
                "path": str(out_dir / f"{entry['name']}.pt"),
                "dtype": meta["dtype"],
                "device": entry["storage_device"],
            }
            for entry in meta["layers"]
        ],
    }
    torch.save(matrix, out_dir / "correlation.pt")
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="output directory")
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    manifest = run_capture(args.out, args.feature_dim, args.samples)
    json.dump(manifest, fp=sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
