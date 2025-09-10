from __future__ import annotations

"""NVML power sampling stub with graceful fallback.

If pynvml is not available or NVML init fails, functions return NaN values.
"""

from typing import Dict, List


def sample_power_series(seconds: float = 1.0, hz: int = 10) -> List[Dict[str, float]]:
    try:
        import time
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        out = []
        steps = max(1, int(seconds * hz))
        for _ in range(steps):
            t = time.perf_counter()
            mw = float(pynvml.nvmlDeviceGetPowerUsage(h))  # milliwatts
            out.append({"t_s": t, "power_w": mw / 1000.0})
            time.sleep(1.0 / hz)
        pynvml.nvmlShutdown()
        return out
    except Exception:
        # fallback one-sample NaN
        return [{"t_s": 0.0, "power_w": float("nan") }]


__all__ = ["sample_power_series"]

