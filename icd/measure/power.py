"""Real NVIDIA Management Library (NVML) power monitoring.

This module provides production-grade GPU power monitoring with background
sampling, energy calculation, and energy-per-token metrics.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

__all__ = [
    "PowerMonitor",
    "is_nvml_available",
    "measure_power",
    "measure_ept_stub",
    "compute_energy_per_token",
]

logger = logging.getLogger(__name__)

# Try to import pynvml
_NVML_AVAILABLE = False
_NVML_IMPORT_ERROR = None

try:
    import pynvml

    _NVML_AVAILABLE = True
except ImportError as e:
    _NVML_IMPORT_ERROR = e
    logger.warning("pynvml not available, power monitoring will be disabled")


def is_nvml_available() -> bool:
    """Check if NVML is available."""
    return _NVML_AVAILABLE


@dataclass
class PowerSample:
    """Single power measurement sample."""

    timestamp: float  # Time in seconds since epoch
    power_watts: float  # Power draw in watts
    temperature_c: Optional[float] = None  # GPU temperature in Celsius
    gpu_util_pct: Optional[float] = None  # GPU utilization percentage


class PowerMonitor:
    """Background GPU power monitor using NVML."""

    def __init__(
        self,
        device_id: int = 0,
        sample_hz: float = 10.0,
        max_samples: int = 10000,
        collect_temperature: bool = True,
        collect_utilization: bool = True,
    ):
        """Initialize power monitor.

        Args:
            device_id: CUDA device ID to monitor.
            sample_hz: Sampling frequency in Hz.
            max_samples: Maximum samples to store (prevents memory bloat).
            collect_temperature: Whether to collect GPU temperature.
            collect_utilization: Whether to collect GPU utilization.
        """
        if not _NVML_AVAILABLE:
            raise RuntimeError(
                "pynvml not available. Install with: pip install nvidia-ml-py3"
            ) from _NVML_IMPORT_ERROR

        self.device_id = device_id
        self.sample_hz = sample_hz
        self.sample_interval = 1.0 / sample_hz
        self.max_samples = max_samples
        self.collect_temperature = collect_temperature
        self.collect_utilization = collect_utilization

        # Initialize NVML
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            # Get device name
            device_name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(device_name, bytes):
                device_name = device_name.decode("utf-8")
            logger.info(f"Power monitor initialized for device {device_id}: {device_name}")

        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to initialize NVML for device {device_id}: {e}")

        # Sampling state
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.samples: Deque[PowerSample] = deque(maxlen=max_samples)
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None

    def start(self) -> None:
        """Start background power sampling."""
        if self.running:
            logger.warning("Power monitor already running")
            return

        self.running = True
        self._start_time = time.perf_counter()
        self.samples.clear()

        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

        logger.info(f"Power monitoring started at {self.sample_hz} Hz")

    def stop(self) -> Dict[str, Any]:
        """Stop sampling and return statistics.

        Returns:
            Dictionary with power statistics:
                - mean_watts: Mean power draw
                - p50_watts: Median power draw
                - p95_watts: 95th percentile power
                - total_energy_joules: Total energy consumed
                - duration_seconds: Monitoring duration
                - num_samples: Number of samples collected
                - samples: List of (timestamp, watts) tuples
        """
        if not self.running:
            logger.warning("Power monitor not running")
            return self._create_null_result()

        self.running = False
        self._stop_time = time.perf_counter()

        if self.thread:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Power monitoring thread did not terminate cleanly")

        return self._compute_statistics()

    def _sample_loop(self) -> None:
        """Background sampling loop."""
        next_sample_time = time.perf_counter()

        while self.running:
            try:
                # Get power in milliwatts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                power_watts = power_mw / 1000.0

                # Optional metrics
                temperature_c = None
                gpu_util_pct = None

                if self.collect_temperature:
                    try:
                        temperature_c = pynvml.nvmlDeviceGetTemperature(
                            self.handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    except pynvml.NVMLError:
                        pass

                if self.collect_utilization:
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                        gpu_util_pct = util.gpu
                    except pynvml.NVMLError:
                        pass

                # Record sample
                timestamp = time.perf_counter()
                sample = PowerSample(
                    timestamp=timestamp,
                    power_watts=power_watts,
                    temperature_c=temperature_c,
                    gpu_util_pct=gpu_util_pct,
                )
                self.samples.append(sample)

                # Sleep until next sample
                next_sample_time += self.sample_interval
                sleep_duration = next_sample_time - time.perf_counter()
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

            except pynvml.NVMLError as e:
                logger.error(f"NVML error during sampling: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error during power sampling: {e}")
                break

    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute statistics from collected samples."""
        if not self.samples:
            return self._create_null_result()

        samples_list = list(self.samples)
        power_values = [s.power_watts for s in samples_list]

        # Basic stats
        import statistics

        mean_watts = statistics.mean(power_values)
        median_watts = statistics.median(power_values)

        # Percentiles
        power_sorted = sorted(power_values)
        p95_idx = int(0.95 * len(power_sorted))
        p95_watts = power_sorted[p95_idx] if p95_idx < len(power_sorted) else power_sorted[-1]

        # Energy calculation (trapezoidal integration)
        total_energy_joules = 0.0
        for i in range(1, len(samples_list)):
            dt = samples_list[i].timestamp - samples_list[i - 1].timestamp
            avg_power = (samples_list[i].power_watts + samples_list[i - 1].power_watts) / 2.0
            total_energy_joules += avg_power * dt

        # Duration
        duration_seconds = samples_list[-1].timestamp - samples_list[0].timestamp

        # Optional: temperature and utilization stats
        temperatures = [s.temperature_c for s in samples_list if s.temperature_c is not None]
        utilizations = [s.gpu_util_pct for s in samples_list if s.gpu_util_pct is not None]

        result = {
            "mean_watts": mean_watts,
            "median_watts": median_watts,
            "p95_watts": p95_watts,
            "total_energy_joules": total_energy_joules,
            "duration_seconds": duration_seconds,
            "num_samples": len(samples_list),
            "samples": [(s.timestamp, s.power_watts) for s in samples_list],
            "status": "ok",
        }

        if temperatures:
            result["mean_temperature_c"] = statistics.mean(temperatures)
            result["max_temperature_c"] = max(temperatures)

        if utilizations:
            result["mean_gpu_util_pct"] = statistics.mean(utilizations)

        return result

    def _create_null_result(self) -> Dict[str, Any]:
        """Create null result when no data available."""
        return {
            "mean_watts": None,
            "median_watts": None,
            "p95_watts": None,
            "total_energy_joules": None,
            "duration_seconds": None,
            "num_samples": 0,
            "samples": [],
            "status": "no_data",
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    def __del__(self):
        """Cleanup NVML on destruction."""
        try:
            if self.running:
                self.stop()
            pynvml.nvmlShutdown()
        except Exception:
            pass


def measure_power(
    enabled: bool = True,
    device_id: int = 0,
    sample_hz: float = 10.0,
) -> Optional[PowerMonitor]:
    """Create and start a power monitor (convenience function).

    Args:
        enabled: Whether power monitoring is enabled.
        device_id: CUDA device ID.
        sample_hz: Sampling frequency in Hz.

    Returns:
        PowerMonitor instance (already started) or None if disabled/unavailable.

    Example:
        >>> monitor = measure_power(enabled=True)
        >>> # ... run workload ...
        >>> stats = monitor.stop()
        >>> print(f"Energy: {stats['total_energy_joules']:.2f} J")
    """
    if not enabled:
        logger.debug("Power monitoring disabled")
        return None

    if not is_nvml_available():
        logger.warning("NVML not available, power monitoring disabled")
        return None

    try:
        monitor = PowerMonitor(device_id=device_id, sample_hz=sample_hz)
        monitor.start()
        return monitor
    except Exception as e:
        logger.error(f"Failed to start power monitoring: {e}")
        return None


def compute_energy_per_token(
    power_stats: Dict[str, Any], num_tokens: int
) -> Optional[float]:
    """Compute energy-per-token (EpT) metric.

    Args:
        power_stats: Dictionary from PowerMonitor.stop()
        num_tokens: Number of tokens processed

    Returns:
        Energy per token in joules, or None if unavailable.
    """
    if not power_stats or power_stats.get("status") != "ok":
        return None

    energy_j = power_stats.get("total_energy_joules")
    if energy_j is None or num_tokens <= 0:
        return None

    return energy_j / num_tokens


def measure_ept_stub(
    tokens: int,
    duration_s: Optional[float] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Emit a schema-compatible EpT payload when NVML is unavailable.

    Downstream reporting expects EpT data even when we cannot access the GPU
    power sensors (e.g. in CI).  The stub mirrors the keys produced by the real
    measurement pipeline so consumers can rely on a stable contract while still
    being able to differentiate stubbed runs via the ``status`` field.
    """

    result: Dict[str, Any] = {
        "status": "stub",
        "tokens": int(tokens),
        "duration_s": float(duration_s) if duration_s is not None else None,
        "ept_j_per_tok": float("nan"),
        "power_stats": {
            "status": "unavailable",
            "mean_watts": float("nan"),
            "total_energy_joules": float("nan"),
            "num_samples": 0,
        },
    }

    if note is not None:
        result["note"] = note

    return result
