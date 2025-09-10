from __future__ import annotations

from time import perf_counter
from typing import Callable, Dict, List


def measure_latency(fn: Callable[[], None], repeats: int = 1000, warmup: int = 50) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    t0 = perf_counter()
    for _ in range(repeats):
        fn()
    t1 = perf_counter()
    dur = (t1 - t0) * 1000.0
    mean = dur / max(repeats, 1)
    return {"mean_ms": mean}


def measure_latency_samples(fn: Callable[[], None], repeats: int = 1000, warmup: int = 50) -> List[float]:
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        t1 = perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return samples


__all__ = ["measure_latency", "measure_latency_samples"]
