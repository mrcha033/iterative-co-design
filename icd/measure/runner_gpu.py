"""Built-in measurement harness for PyTorch models."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import torch

from .profiling import nvtx_range

__all__ = ["BenchmarkConfig", "benchmark_inference"]


@dataclass
class BenchmarkConfig:
    repeats: int = 200
    warmup: int = 20
    sync: bool = True
    use_cuda_events: bool = True
    device: str | None = None
    tokens_per_batch: int | None = None


def _ensure_tuple(inputs: Any) -> Tuple[Any, ...]:
    if isinstance(inputs, tuple):
        return inputs
    return (inputs,)


def _select_device(model: torch.nn.Module, cfg: BenchmarkConfig) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    try:
        return next(model.parameters()).device  # type: ignore[attr-defined]
    except StopIteration:
        return torch.device("cpu")


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_inference(
    model: torch.nn.Module,
    example_inputs: Any,
    cfg: BenchmarkConfig | None = None,
) -> Dict[str, Any]:
    """Benchmark a model forward pass using CPU timers or CUDA events."""

    cfg = cfg or BenchmarkConfig()
    device = _select_device(model, cfg)
    inputs = tuple(inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in _ensure_tuple(example_inputs))
    model = model.to(device)
    model.eval()

    repeats = max(1, int(cfg.repeats))
    warmup = max(0, int(cfg.warmup))

    use_cuda_events = device.type == "cuda" and cfg.use_cuda_events and torch.cuda.is_available()

    event_start = event_end = None
    if use_cuda_events:
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)

    latencies: list[float] = []

    with torch.inference_mode():
        for _ in range(warmup):
            with nvtx_range("ICD_BENCHMARK_WARMUP"):
                _ = model(*inputs)
        for _ in range(repeats):
            if use_cuda_events and event_start and event_end:
                event_start.record()
                with nvtx_range("ICD_BENCHMARK_FORWARD"):
                    _ = model(*inputs)
                event_end.record()
                if cfg.sync:
                    torch.cuda.synchronize(device)
                lat = event_start.elapsed_time(event_end)
            else:
                if cfg.sync:
                    _maybe_sync(device)
                start = time.perf_counter()
                with nvtx_range("ICD_BENCHMARK_FORWARD"):
                    _ = model(*inputs)
                if cfg.sync and device.type == "cuda":
                    torch.cuda.synchronize(device)
                lat = (time.perf_counter() - start) * 1000.0
            latencies.append(float(lat))

    latencies.sort()
    mean = statistics.fmean(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
    stdev = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
    ci95 = 1.96 * (stdev / (len(latencies) ** 0.5)) if len(latencies) > 1 else 0.0

    result: Dict[str, Any] = {
        "latency_ms": latencies,
        "latency_ms_mean": mean,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "latency_ms_ci95": ci95,
        "device": str(device),
        "repeats": repeats,
        "warmup": warmup,
    }
    if cfg.tokens_per_batch:
        total_tokens = cfg.tokens_per_batch * repeats
        result["tokens"] = total_tokens
        result["throughput_toks_s"] = (cfg.tokens_per_batch * 1000.0) / mean if mean > 0 else float("nan")
    return result

