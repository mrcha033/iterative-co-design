"""
Profiling modules for precise benchmarking and hardware metric collection.

This package provides specialized profilers for:
- Latency measurement with torch.cuda.Event
- Hardware profiling with NVIDIA Nsight Compute
- System calibration and baseline verification
"""

from .latency import LatencyProfiler, benchmark_model_latency
from .ncu import NsightComputeProfiler, collect_hardware_metrics
from .calibration import SystemCalibrator, validate_system_performance

__all__ = [
    'LatencyProfiler',
    'benchmark_model_latency', 
    'NsightComputeProfiler',
    'collect_hardware_metrics',
    'SystemCalibrator',
    'validate_system_performance'
]