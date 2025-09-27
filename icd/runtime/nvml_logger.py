"""NVML power logging with graceful fallbacks."""

from __future__ import annotations

import contextlib
from typing import List

try:  # pragma: no cover - optional dependency
    import pynvml
except Exception:  # pragma: no cover - fallback when NVML missing
    pynvml = None  # type: ignore[assignment]


class NVMLPowerLogger(contextlib.AbstractContextManager):
    def __init__(self, device_index: int = 0, interval_s: float = 0.1) -> None:
        self.device_index = int(device_index)
        self.interval_s = float(interval_s)
        self._readings: List[float] = []
        self._running = False

    def __enter__(self) -> "NVMLPowerLogger":
        if pynvml is None:
            return self
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._running = True
        except Exception:
            self._running = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._running = False
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def tick(self) -> None:
        if not self._running or pynvml is None:
            return
        power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
        self._readings.append(power_mw / 1000.0)

    def energy_j(self) -> float:
        if not self._readings:
            return 0.0
        return sum(self._readings) * self.interval_s

