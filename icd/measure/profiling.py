"""Profiling helpers for latency, Nsight Compute, and NVML."""

from __future__ import annotations

import contextlib
import math
import os
import re
import subprocess
import time
from typing import Iterable, Optional

try:  # optional NVTX
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # optional NVML
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore

__all__ = [
    "nvtx_range",
    "NVMLPowerLogger",
    "run_with_ncu",
    "energy_per_token_j",
]


@contextlib.contextmanager
def nvtx_range(name: str = "ICD_MEASURE"):
    """Lightweight NVTX context that degrades gracefully when unavailable."""

    if torch is None or not hasattr(torch.cuda, "nvtx"):
        yield
        return

    pushed = False
    try:
        try:
            torch.cuda.nvtx.range_push(name)  # type: ignore[attr-defined]
            pushed = True
        except Exception:
            pushed = False
        yield
    finally:
        if pushed:
            try:
                torch.cuda.nvtx.range_pop()  # type: ignore[attr-defined]
            except Exception:
                pass


class NVMLPowerLogger:
    """Context manager that samples NVML power readings."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._handle = None
        self.samples: list[tuple[float, float]] = []
        self._start = 0.0
        self._available = pynvml is not None

    def __enter__(self):
        if not self._available:
            return self
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._start = time.perf_counter()
        except Exception:
            self._available = False
            self._handle = None
        return self

    def tick(self) -> None:
        if not self._available or self._handle is None:
            return
        try:
            power_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            t = time.perf_counter() - self._start
            self.samples.append((t, float(power_w)))
        except Exception:
            pass

    def energy_j(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        energy = 0.0
        for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]):
            dt = max(0.0, t1 - t0)
            energy += 0.5 * (p0 + p1) * dt
        return energy

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        self._handle = None
        return False


def run_with_ncu(
    command: Iterable[str],
    metrics: Optional[Iterable[str]] = None,
    *,
    env: Optional[dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> dict[str, float]:
    """Run a command with Nsight Compute and parse metrics.

    Parameters
    ----------
    command: Sequence of strings representing the target command (e.g. Python invocation).
    metrics: Optional sequence of metric names; defaults to L2 hit rate.
    """

    metrics = tuple(metrics or ("lts__t_sectors_hit_rate.pct",))
    binary = os.environ.get("NCU_PATH", "ncu")
    args = [
        binary,
        "--target-processes",
        "all",
        "--page",
        "raw",
        "--csv",
        "--metrics",
        ",".join(metrics),
        "--",
    ] + list(command)

    try:
        output = subprocess.check_output(args, text=True, env=env, cwd=cwd)
    except Exception:
        return {}

    result: dict[str, float] = {}
    pattern = re.compile(r"([\w\.]+),[^,]*,([\d\.eE\+-]+)$")
    for line in output.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        key, value = match.groups()
        if key in metrics:
            try:
                result[key] = float(value)
            except ValueError:
                continue
    if "lts__t_sectors_hit_rate.pct" in result:
        result["l2_hit_rate_pct"] = result["lts__t_sectors_hit_rate.pct"]
    return result


def energy_per_token_j(energy_joules: float, tokens: int) -> float:
    if tokens <= 0:
        return float("nan")
    return energy_joules / tokens
