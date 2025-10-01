"""Latency measurement utilities used across the measurement stack.

The public API intentionally mirrors the documentation in
``docs/API_Reference.md`` where :class:`LatencyMeasurer` is referenced.
Historically this module only exposed two helper functions; however the docs
and downstream tooling expect the richer class interface.  This module now
provides both the legacy helpers and the documented class so that existing
callers continue to work while the documentation remains accurate.
"""

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

try:  # pragma: no cover - torch is optional in CI
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def measure_latency(fn: Callable[[], None], repeats: int = 1000, warmup: int = 50) -> Dict[str, float]:
    """Return the mean latency (in milliseconds) for ``fn`` over ``repeats`` runs."""

    for _ in range(max(0, warmup)):
        fn()
    t0 = time.perf_counter()
    for _ in range(max(1, repeats)):
        fn()
    t1 = time.perf_counter()
    dur = (t1 - t0) * 1000.0
    mean = dur / max(repeats, 1)
    return {"mean_ms": mean}


def measure_latency_samples(fn: Callable[[], None], repeats: int = 1000, warmup: int = 50) -> List[float]:
    """Collect per-iteration latency samples (in milliseconds)."""

    for _ in range(max(0, warmup)):
        fn()
    samples: List[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return samples


def _percentile(sorted_samples: Sequence[float], pct: float) -> float:
    if not sorted_samples:
        return float("nan")
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    k = (len(sorted_samples) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_samples[int(k)])
    d0 = sorted_samples[f] * (c - k)
    d1 = sorted_samples[c] * (k - f)
    return float(d0 + d1)


def _iqr_bounds(samples: Sequence[float]) -> Tuple[float, float]:
    q1 = _percentile(samples, 0.25)
    q3 = _percentile(samples, 0.75)
    if math.isnan(q1) or math.isnan(q3):
        return float("nan"), float("nan")
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _summarize(samples: Sequence[float]) -> Dict[str, Any]:
    if not samples:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "ci95": (float("nan"), float("nan")),
            "outliers": 0,
            "raw_samples": [],
        }

    sorted_samples = sorted(float(s) for s in samples)
    mean = statistics.fmean(sorted_samples)
    std = statistics.pstdev(sorted_samples) if len(sorted_samples) > 1 else 0.0
    p50 = _percentile(sorted_samples, 0.5)
    p95 = _percentile(sorted_samples, 0.95)
    p99 = _percentile(sorted_samples, 0.99)
    if len(sorted_samples) > 1:
        half_width = 1.96 * (std / math.sqrt(len(sorted_samples)))
        ci95 = (mean - half_width, mean + half_width)
    else:
        ci95 = (mean, mean)
    lower, upper = _iqr_bounds(sorted_samples)
    if math.isnan(lower) or math.isnan(upper):
        outliers = 0
    else:
        outliers = sum(1 for s in sorted_samples if s < lower or s > upper)

    return {
        "mean": mean,
        "std": std,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "ci95": ci95,
        "outliers": outliers,
        "raw_samples": list(sorted_samples),
    }


def _move_to_device(obj: Any, device: "torch.device") -> Any:  # pragma: no cover - torch optional
    if torch is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach().to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_move_to_device(v, device) for v in obj]
        return type(obj)(converted)
    return obj


def _ensure_callable_inputs(inputs: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any] | None]:
    if isinstance(inputs, dict):
        return tuple(), {str(k): v for k, v in inputs.items()}
    if isinstance(inputs, tuple):
        return inputs, None
    if isinstance(inputs, list):
        return tuple(inputs), None
    return (inputs,), None


@dataclass
class LatencyMeasurer:
    """High level latency measurement helper.

    Parameters mirror the documentation: ``warmup_iter`` controls the number of
    warmup runs, ``repeats`` is the sample size, and ``fixed_clock`` toggles
    whether GPU synchronisation is enforced before timing measurements.
    """

    warmup_iter: int = 50
    repeats: int = 1000
    fixed_clock: bool = True
    sync_gpu: bool = True

    def _summarize(self, samples: Sequence[float]) -> Dict[str, Any]:
        summary = _summarize(samples)
        summary["warmup_iter"] = self.warmup_iter
        summary["repeats"] = self.repeats
        summary["fixed_clock"] = self.fixed_clock
        return summary

    def measure_callable(self, fn: Callable[[], None]) -> Dict[str, Any]:
        """Measure latency for an arbitrary callable."""

        samples = measure_latency_samples(fn, repeats=self.repeats, warmup=self.warmup_iter)
        return self._summarize(samples)

    def measure(self, model: Any, inputs: Any, device: str | None = None) -> Dict[str, Any]:
        """Measure latency of a PyTorch ``model`` on ``inputs``.

        Parameters
        ----------
        model:
            ``torch.nn.Module`` to evaluate.
        inputs:
            Positional (``tuple``/``list``) or keyword (``dict``) inputs passed to
            the model during measurement.
        device:
            Optional device string; defaults to ``"cuda"`` when available.
        """

        if torch is None:  # pragma: no cover - optional dependency
            raise RuntimeError("PyTorch is required for LatencyMeasurer.measure")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_device = torch.device(device)

        model = model.to(torch_device)
        model.eval()

        positional, keyword = _ensure_callable_inputs(_move_to_device(inputs, torch_device))

        def _sync() -> None:
            if not self.sync_gpu:
                return
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)

        @torch.inference_mode()  # type: ignore[attr-defined]
        def _forward() -> None:  # pragma: no cover - timing heavy for unit tests
            if keyword is None:
                model(*positional)
            else:
                model(**keyword)

        for _ in range(max(0, self.warmup_iter)):
            _forward()

        samples: List[float] = []
        for _ in range(max(1, self.repeats)):
            if self.fixed_clock:
                _sync()
            start = time.perf_counter()
            _forward()
            if self.fixed_clock:
                _sync()
            samples.append((time.perf_counter() - start) * 1000.0)

        summary = self._summarize(samples)
        summary["device"] = str(torch_device)
        return summary


__all__ = [
    "LatencyMeasurer",
    "measure_latency",
    "measure_latency_samples",
]
