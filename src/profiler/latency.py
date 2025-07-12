"""
Precise latency profiling using torch.cuda.Event for accurate timing.

This module implements high-precision latency measurement with proper
warmup, synchronization, and statistical analysis as required by the PRD.
"""
import logging
import time
import warnings
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class LatencyConfig:
    """Configuration for latency profiling."""
    warmup_runs: int = 10
    measurement_runs: int = 5
    use_cuda_events: bool = True
    cuda_sync: bool = True
    enable_autograd: bool = False
    memory_cleanup: bool = True
    statistical_validation: bool = True
    outlier_threshold: float = 2.0  # Standard deviations for outlier detection


@dataclass
class LatencyResults:
    """Results from latency profiling."""
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    cv_percent: float  # Coefficient of variation
    raw_times: List[float]
    warmup_times: List[float]
    outliers_removed: int
    measurement_runs: int
    device: str
    timing_method: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'mean_latency_ms': self.mean_latency_ms,
            'std_latency_ms': self.std_latency_ms,
            'min_latency_ms': self.min_latency_ms,
            'max_latency_ms': self.max_latency_ms,
            'median_latency_ms': self.median_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'cv_percent': self.cv_percent,
            'outliers_removed': self.outliers_removed,
            'measurement_runs': self.measurement_runs,
            'device': self.device,
            'timing_method': self.timing_method,
            'raw_times': self.raw_times
        }


class LatencyProfiler:
    """
    High-precision latency profiler using torch.cuda.Event.
    
    This profiler implements the exact timing methodology described in the PRD,
    with proper warmup, synchronization, and statistical validation.
    """
    
    def __init__(self, config: Optional[LatencyConfig] = None):
        """
        Initialize latency profiler.
        
        Args:
            config: Profiling configuration
        """
        self.config = config if config is not None else LatencyConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"LatencyProfiler initialized for device: {self.device}")
        
    def _validate_config(self) -> None:
        """Validate profiling configuration."""
        if self.config.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        
        if self.config.measurement_runs <= 0:
            raise ValueError("measurement_runs must be positive")
        
        if self.device == 'cpu' and self.config.use_cuda_events:
            warnings.warn("CUDA events not available on CPU, falling back to time.perf_counter()")
            self.config.use_cuda_events = False
            
        if self.config.outlier_threshold <= 0:
            raise ValueError("outlier_threshold must be positive")
    
    def profile_function(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> LatencyResults:
        """
        Profile a function's execution time.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            LatencyResults with detailed timing statistics
        """
        # Determine device from model parameters if available
        device = self._infer_device(args, kwargs)
        
        # Memory cleanup before profiling
        if self.config.memory_cleanup and device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Warmup runs
        warmup_times = self._run_warmup(func, device, *args, **kwargs)
        
        # Measurement runs
        measurement_times = self._run_measurements(func, device, *args, **kwargs)
        
        # Statistical analysis
        results = self._analyze_results(measurement_times, warmup_times, device)
        
        return results
    
    def _infer_device(self, args, kwargs) -> str:
        """Infer device from function arguments."""
        # Check for model parameters
        for arg in args:
            if isinstance(arg, nn.Module):
                try:
                    param = next(arg.parameters())
                    return param.device.type
                except StopIteration:
                    continue
        
        # Check for tensor arguments
        for arg in args:
            if isinstance(arg, torch.Tensor):
                return arg.device.type
        
        # Check kwargs
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                return value.device.type
            elif isinstance(value, nn.Module):
                try:
                    param = next(value.parameters())
                    return param.device.type
                except StopIteration:
                    continue
        
        return self.device
    
    def _run_warmup(
        self,
        func: Callable,
        device: str,
        *args,
        **kwargs
    ) -> List[float]:
        """Run warmup iterations."""
        warmup_times = []
        
        logger.debug(f"Running {self.config.warmup_runs} warmup iterations")
        
        for i in range(self.config.warmup_runs):
            try:
                elapsed_ms = self._time_single_execution(func, device, *args, **kwargs)
                warmup_times.append(elapsed_ms)
            except Exception as e:
                logger.warning(f"Warmup run {i} failed: {e}")
                continue
        
        if warmup_times:
            logger.debug(f"Warmup completed: {np.mean(warmup_times):.2f}±{np.std(warmup_times):.2f} ms")
        
        return warmup_times
    
    def _run_measurements(
        self,
        func: Callable,
        device: str,
        *args,
        **kwargs
    ) -> List[float]:
        """Run measurement iterations."""
        measurement_times = []
        
        logger.debug(f"Running {self.config.measurement_runs} measurement iterations")
        
        for i in range(self.config.measurement_runs):
            try:
                elapsed_ms = self._time_single_execution(func, device, *args, **kwargs)
                measurement_times.append(elapsed_ms)
            except Exception as e:
                logger.error(f"Measurement run {i} failed: {e}")
                continue
        
        if not measurement_times:
            raise RuntimeError("All measurement runs failed")
        
        logger.debug(f"Measurements completed: {len(measurement_times)} successful runs")
        
        return measurement_times
    
    def _time_single_execution(
        self,
        func: Callable,
        device: str,
        *args,
        **kwargs
    ) -> float:
        """Time a single function execution."""
        if device == 'cuda' and self.config.use_cuda_events:
            return self._time_with_cuda_events(func, *args, **kwargs)
        else:
            return self._time_with_perf_counter(func, device, *args, **kwargs)
    
    def _time_with_cuda_events(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> float:
        """Time execution using CUDA events."""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Synchronize before timing
        if self.config.cuda_sync:
            torch.cuda.synchronize()
        
        # Start timing
        start_event.record()
        
        # Execute function
        if self.config.enable_autograd:
            result = func(*args, **kwargs)
        else:
            with torch.no_grad():
                result = func(*args, **kwargs)
        
        # End timing
        end_event.record()
        
        # Synchronize and get elapsed time
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        
        return elapsed_ms
    
    def _time_with_perf_counter(
        self,
        func: Callable,
        device: str,
        *args,
        **kwargs
    ) -> float:
        """Time execution using time.perf_counter()."""
        # Synchronize before timing if on CUDA
        if device == 'cuda' and self.config.cuda_sync:
            torch.cuda.synchronize()
        
        # Start timing
        start_time = time.perf_counter()
        
        # Execute function
        if self.config.enable_autograd:
            result = func(*args, **kwargs)
        else:
            with torch.no_grad():
                result = func(*args, **kwargs)
        
        # Synchronize after execution if on CUDA
        if device == 'cuda' and self.config.cuda_sync:
            torch.cuda.synchronize()
        
        # End timing
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000.0
        return elapsed_ms
    
    def _analyze_results(
        self,
        measurement_times: List[float],
        warmup_times: List[float],
        device: str
    ) -> LatencyResults:
        """Analyze timing results and compute statistics."""
        times = np.array(measurement_times)
        
        # Remove outliers if enabled
        outliers_removed = 0
        if self.config.statistical_validation and len(times) > 3:
            mean_time = np.mean(times)
            std_time = np.std(times)
            
            # Remove outliers beyond threshold standard deviations
            outlier_mask = np.abs(times - mean_time) <= (self.config.outlier_threshold * std_time)
            filtered_times = times[outlier_mask]
            
            if len(filtered_times) >= 3:  # Keep at least 3 measurements
                outliers_removed = len(times) - len(filtered_times)
                times = filtered_times
                
                if outliers_removed > 0:
                    logger.debug(f"Removed {outliers_removed} outlier measurements")
        
        # Compute statistics
        mean_latency = float(np.mean(times))
        std_latency = float(np.std(times))
        min_latency = float(np.min(times))
        max_latency = float(np.max(times))
        median_latency = float(np.median(times))
        
        # Percentiles
        p95_latency = float(np.percentile(times, 95)) if len(times) > 1 else mean_latency
        p99_latency = float(np.percentile(times, 99)) if len(times) > 1 else mean_latency
        
        # Coefficient of variation
        cv_percent = (std_latency / mean_latency * 100) if mean_latency > 0 else 0.0
        
        # Timing method
        timing_method = 'cuda_events' if device == 'cuda' and self.config.use_cuda_events else 'perf_counter'
        
        results = LatencyResults(
            mean_latency_ms=mean_latency,
            std_latency_ms=std_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            cv_percent=cv_percent,
            raw_times=times.tolist(),
            warmup_times=warmup_times,
            outliers_removed=outliers_removed,
            measurement_runs=len(times),
            device=device,
            timing_method=timing_method
        )
        
        # Log results
        logger.info(f"Latency profiling complete: {mean_latency:.2f}±{std_latency:.2f} ms "
                   f"(CV: {cv_percent:.1f}%, runs: {len(times)})")
        
        # Warning for high variability
        if cv_percent > 10.0:
            logger.warning(f"High timing variability detected (CV: {cv_percent:.1f}%). "
                          "Consider increasing warmup runs or checking system load.")
        
        return results


def benchmark_model_latency(
    model: nn.Module,
    inputs: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
    config: Optional[LatencyConfig] = None
) -> LatencyResults:
    """
    Benchmark model inference latency.
    
    This is a convenient wrapper function for profiling model inference.
    
    Args:
        model: Model to benchmark
        inputs: Input tensors for the model
        config: Profiling configuration
        
    Returns:
        LatencyResults with detailed timing statistics
    """
    profiler = LatencyProfiler(config)
    
    # Ensure model is in eval mode
    model.eval()
    
    # Define the function to profile
    def model_forward():
        if isinstance(inputs, dict):
            return model(**inputs)
        elif isinstance(inputs, (list, tuple)):
            return model(*inputs)
        else:
            return model(inputs)
    
    return profiler.profile_function(model_forward)


# Utility functions for common profiling scenarios
def profile_layer_latency(
    layer: nn.Module,
    input_shape: tuple,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
    config: Optional[LatencyConfig] = None
) -> LatencyResults:
    """
    Profile latency of a single layer.
    
    Args:
        layer: Layer to profile
        input_shape: Shape of input tensor
        device: Device to use
        dtype: Data type for input tensor
        config: Profiling configuration
        
    Returns:
        LatencyResults with detailed timing statistics
    """
    layer = layer.to(device)
    layer.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device, dtype=dtype)
    
    return benchmark_model_latency(layer, dummy_input, config)


def compare_model_latencies(
    models: Dict[str, nn.Module],
    inputs: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
    config: Optional[LatencyConfig] = None
) -> Dict[str, LatencyResults]:
    """
    Compare latencies of multiple models.
    
    Args:
        models: Dictionary of model name to model
        inputs: Input tensors (same for all models)
        config: Profiling configuration
        
    Returns:
        Dictionary of model name to LatencyResults
    """
    results = {}
    
    for name, model in models.items():
        logger.info(f"Profiling model: {name}")
        try:
            results[name] = benchmark_model_latency(model, inputs, config)
        except Exception as e:
            logger.error(f"Failed to profile model {name}: {e}")
            continue
    
    return results


def analyze_latency_improvements(
    baseline: LatencyResults,
    optimized: LatencyResults
) -> Dict[str, float]:
    """
    Analyze latency improvements between baseline and optimized models.
    
    Args:
        baseline: Baseline latency results
        optimized: Optimized model latency results
        
    Returns:
        Dictionary with improvement metrics
    """
    improvement_ms = baseline.mean_latency_ms - optimized.mean_latency_ms
    improvement_pct = (improvement_ms / baseline.mean_latency_ms) * 100
    speedup_factor = baseline.mean_latency_ms / optimized.mean_latency_ms
    
    # Statistical significance (simple t-test approximation)
    baseline_times = np.array(baseline.raw_times)
    optimized_times = np.array(optimized.raw_times)
    
    pooled_std = np.sqrt(
        ((len(baseline_times) - 1) * baseline.std_latency_ms**2 + 
         (len(optimized_times) - 1) * optimized.std_latency_ms**2) /
        (len(baseline_times) + len(optimized_times) - 2)
    )
    
    t_stat = improvement_ms / (pooled_std * np.sqrt(1/len(baseline_times) + 1/len(optimized_times)))
    
    return {
        'improvement_ms': improvement_ms,
        'improvement_pct': improvement_pct,
        'speedup_factor': speedup_factor,
        'baseline_mean_ms': baseline.mean_latency_ms,
        'optimized_mean_ms': optimized.mean_latency_ms,
        'baseline_std_ms': baseline.std_latency_ms,
        'optimized_std_ms': optimized.std_latency_ms,
        't_statistic': t_stat,
        'statistically_significant': abs(t_stat) > 2.0  # Rough significance threshold
    }