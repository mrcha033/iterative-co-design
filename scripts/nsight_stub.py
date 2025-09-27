#!/usr/bin/env python3
"""Stub Nsight invocation for CI environments without GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def run_stub(output: Path, kernels: Iterable[str]) -> dict:
    payload = {
        "kernels": list(kernels),
        "summary": {
            "launches": len(list(kernels)),
            "status": "stubbed",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("kernels", nargs="*", default=["icd::stub_kernel"])
    args = parser.parse_args()
    payload = run_stub(args.out, args.kernels)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
