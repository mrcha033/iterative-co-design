"""Utility to assert deterministic CUDA environment configuration."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

REQUIRED_DRIVER_MAJOR = 535
REQUIRED_CUDA_RUNTIME = "12.2"


def _nvidia_smi() -> dict | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query", "--format=json"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def main() -> int:
    runtime_version = torch.version.cuda or "unknown"
    if runtime_version != REQUIRED_CUDA_RUNTIME:
        print(
            f"Expected CUDA runtime {REQUIRED_CUDA_RUNTIME}, got {runtime_version}",
            file=sys.stderr,
        )
        return 1

    payload = _nvidia_smi()
    if not payload:
        print("nvidia-smi not available; cannot verify driver version", file=sys.stderr)
        return 1

    driver_version = payload.get("driver_version", "0.0")
    major = int(driver_version.split(".")[0])
    if major < REQUIRED_DRIVER_MAJOR:
        print(
            f"Driver version {driver_version} < required major {REQUIRED_DRIVER_MAJOR}",
            file=sys.stderr,
        )
        return 1

    fingerprint = {
        "driver_version": driver_version,
        "cuda_runtime": runtime_version,
        "torch": torch.__version__,
    }
    out_path = Path("logs/cuda_fingerprint.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    print(f"CUDA environment fingerprint written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
