"""
Profiling utilities for hardware performance measurement.

This module provides utilities for collecting hardware performance metrics
using NVIDIA Nsight Compute and PyTorch profiler.
"""
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import warnings

import torch
import torch.nn as nn
import numpy as np


logger = logging.getLogger(__name__)


class HardwareProfiler:
    """Hardware profiler for collecting performance metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize hardware profiler.
        
        Args:
            config: Profiling configuration
        """
        self.config = config
        self.enabled = config.get('enabled', False)
        self.tool = config.get('tool', 'nsight_compute')
        self.metrics = config.get('metrics', [])
        
        # Check tool availability
        if self.enabled:
            self._check_tool_availability()
    
    def _check_tool_availability(self) -> None:
        """Check if profiling tools are available."""
        if self.tool == 'nsight_compute':
            try:
                result = subprocess.run(['ncu', '--version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    warnings.warn("NVIDIA Nsight Compute (ncu) not found in PATH")
                    self.enabled = False
                else:
                    logger.info(f"Found Nsight Compute: {result.stdout.strip()}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                warnings.warn("NVIDIA Nsight Compute (ncu) not available")
                self.enabled = False
        
        elif self.tool == 'pytorch_profiler':
            # PyTorch profiler is always available
            pass
        
        else:
            warnings.warn(f"Unknown profiling tool: {self.tool}")
            self.enabled = False
    
    def profile_model(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Profile model execution and collect hardware metrics.
        
        Args:
            model: Model to profile
            inputs: Input tensors
            output_dir: Directory to save profiling results
            
        Returns:
            Dictionary containing profiling results
        """
        if not self.enabled:
            logger.warning("Profiling disabled or tools unavailable")
            return {}
        
        if self.tool == 'nsight_compute':
            return self._profile_with_nsight_compute(model, inputs, output_dir)
        elif self.tool == 'pytorch_profiler':
            return self._profile_with_pytorch_profiler(model, inputs, output_dir)
        else:
            logger.error(f"Unsupported profiling tool: {self.tool}")
            return {}
    
    def _profile_with_nsight_compute(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Profile using NVIDIA Nsight Compute."""
        logger.info("Starting profiling with NVIDIA Nsight Compute...")
        
        # Create temporary script for profiling
        profile_script = self._create_profile_script(model, inputs)
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(profile_script)
                script_path = f.name
            
            # Create output file for ncu results
            if output_dir:
                ncu_output = output_dir / 'ncu_profile.csv'
            else:
                ncu_output = Path(tempfile.mktemp(suffix='_ncu.csv'))
            
            # Construct ncu command
            ncu_cmd = [
                'ncu',
                '--csv',
                '--log-file', str(ncu_output),
                '--metrics', ','.join(self.metrics),
                '--target-processes', 'all',
                'python', script_path
            ]
            
            # Run profiling
            logger.info(f"Running command: {' '.join(ncu_cmd)}")
            result = subprocess.run(
                ncu_cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Nsight Compute failed: {result.stderr}")
                return {'error': result.stderr}
            
            # Parse results
            profile_results = self._parse_ncu_results(ncu_output)
            
            return profile_results
            
        except subprocess.TimeoutExpired:
            logger.error("Nsight Compute profiling timed out")
            return {'error': 'Profiling timeout'}
        except Exception as e:
            logger.error(f"Profiling failed: {e}")
            return {'error': str(e)}
        finally:
            # Cleanup temporary files
            if 'script_path' in locals():
                try:
                    os.unlink(script_path)
                except OSError:
                    pass
    
    def _create_profile_script(self, model: nn.Module, inputs: torch.Tensor) -> str:
        """Create a Python script for profiling."""
        script = f"""
import torch
import torch.nn as nn
import pickle
import sys

# Load model and inputs (this would need proper serialization in practice)
def run_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simplified model execution for profiling
    # In practice, you'd serialize and load the actual model
    x = torch.randn({list(inputs.shape)}, device=device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            # Simulate model forward pass
            y = torch.matmul(x, x.transpose(-2, -1))
            torch.cuda.synchronize()
    
    # Profile target
    with torch.no_grad():
        torch.cuda.synchronize()
        for _ in range(5):
            y = torch.matmul(x, x.transpose(-2, -1))
            torch.cuda.synchronize()

if __name__ == '__main__':
    run_model()
"""
        return script
    
    def _parse_ncu_results(self, csv_path: Path) -> Dict[str, Any]:
        """Parse Nsight Compute CSV results."""
        results = {
            'tool': 'nsight_compute',
            'metrics': {},
            'raw_file': str(csv_path)
        }
        
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            # Extract key metrics
            for metric in self.metrics:
                if metric in df.columns:
                    values = df[metric].dropna()
                    if len(values) > 0:
                        results['metrics'][metric] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()) if len(values) > 1 else 0.0,
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'values': values.tolist()
                        }
            
            # Extract L2 cache hit rate if available
            l2_metric = 'lts__t_sector_hit_rate.pct'
            if l2_metric in results['metrics']:
                results['l2_cache_hit_rate'] = results['metrics'][l2_metric]['mean']
            
            # Extract DRAM bandwidth if available
            dram_read_metric = 'dram__bytes_read.sum'
            dram_write_metric = 'dram__bytes_write.sum'
            
            if dram_read_metric in results['metrics'] and dram_write_metric in results['metrics']:
                read_bw = results['metrics'][dram_read_metric]['mean']
                write_bw = results['metrics'][dram_write_metric]['mean']
                results['dram_bandwidth_gb_s'] = (read_bw + write_bw) / 1e9
            
        except Exception as e:
            logger.error(f"Failed to parse ncu results: {e}")
            results['parse_error'] = str(e)
        
        return results
    
    def _profile_with_pytorch_profiler(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Profile using PyTorch profiler."""
        logger.info("Starting profiling with PyTorch profiler...")
        
        results = {
            'tool': 'pytorch_profiler',
            'metrics': {}
        }
        
        try:
            from torch.profiler import profile, record_function, ProfilerActivity
            
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            
            with profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                with record_function("model_inference"):
                    model.eval()
                    with torch.no_grad():
                        _ = model(inputs)
            
            # Extract key metrics
            events = prof.key_averages()
            
            cuda_events = [e for e in events if e.device_type == 'cuda']
            if cuda_events:
                total_cuda_time = sum(e.cuda_time_total for e in cuda_events)
                total_self_cuda_time = sum(e.self_cuda_time_total for e in cuda_events)
                
                results['metrics']['total_cuda_time_us'] = total_cuda_time
                results['metrics']['self_cuda_time_us'] = total_self_cuda_time
                results['metrics']['cuda_memory_usage'] = sum(e.cuda_memory_usage for e in cuda_events if e.cuda_memory_usage > 0)
            
            cpu_events = [e for e in events if e.device_type == 'cpu']
            if cpu_events:
                total_cpu_time = sum(e.cpu_time_total for e in cpu_events)
                results['metrics']['total_cpu_time_us'] = total_cpu_time
            
            # Save detailed profile if output directory provided
            if output_dir:
                trace_path = output_dir / 'pytorch_trace.json'
                prof.export_chrome_trace(str(trace_path))
                results['trace_file'] = str(trace_path)
                
                # Also save key averages
                averages_path = output_dir / 'pytorch_averages.txt'
                with open(averages_path, 'w') as f:
                    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                results['averages_file'] = str(averages_path)
        
        except Exception as e:
            logger.error(f"PyTorch profiling failed: {e}")
            results['error'] = str(e)
        
        return results


class BenchmarkRunner:
    """Utility for running performance benchmarks."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize benchmark runner.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.warmup_runs = config.get('warmup_runs', 10)
        self.num_runs = config.get('num_runs', 5)
        self.use_cuda_events = config.get('use_cuda_events', True)
        self.cuda_sync = config.get('cuda_sync', True)
        
    def benchmark_function(
        self, 
        func: Callable,
        *args,
        **kwargs
    ) -> Dict[str, float]:
        """
        Benchmark a function execution.
        
        Args:
            func: Function to benchmark
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Dictionary with timing statistics
        """
        device = 'cuda' if torch.cuda.is_available() and next(iter(kwargs.get('model', [None]).parameters()), torch.tensor(0)).is_cuda else 'cpu'
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                _ = func(*args, **kwargs)
                if device == 'cuda' and self.cuda_sync:
                    torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"Warmup run failed: {e}")
        
        # Benchmark runs
        times = []
        
        for run in range(self.num_runs):
            if device == 'cuda' and self.use_cuda_events:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                if self.cuda_sync:
                    torch.cuda.synchronize()
                
                start_event.record()
                try:
                    _ = func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Benchmark run {run} failed: {e}")
                    continue
                    
                end_event.record()
                torch.cuda.synchronize()
                
                elapsed_ms = start_event.elapsed_time(end_event)
                times.append(elapsed_ms)
                
            else:
                if device == 'cuda' and self.cuda_sync:
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                try:
                    _ = func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Benchmark run {run} failed: {e}")
                    continue
                    
                if device == 'cuda' and self.cuda_sync:
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000
                times.append(elapsed_ms)
        
        if not times:
            logger.error("All benchmark runs failed")
            return {'error': 'All runs failed'}
        
        return {
            'mean_latency_ms': float(np.mean(times)),
            'std_latency_ms': float(np.std(times)) if len(times) > 1 else 0.0,
            'min_latency_ms': float(np.min(times)),
            'max_latency_ms': float(np.max(times)),
            'median_latency_ms': float(np.median(times)),
            'p95_latency_ms': float(np.percentile(times, 95)) if len(times) > 1 else times[0],
            'times': times,
            'num_runs': len(times),
            'warmup_runs': self.warmup_runs,
            'device': device
        }
    
    def benchmark_model(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        include_profiling: bool = False,
        profiler: Optional[HardwareProfiler] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Benchmark model inference.
        
        Args:
            model: Model to benchmark
            inputs: Input tensors
            include_profiling: Whether to include hardware profiling
            profiler: Hardware profiler instance
            output_dir: Output directory for profiling results
            
        Returns:
            Dictionary with benchmark and profiling results
        """
        model.eval()
        
        def model_forward():
            with torch.no_grad():
                return model(inputs)
        
        # Run benchmark
        benchmark_results = self.benchmark_function(model_forward)
        
        results = {
            'benchmark': benchmark_results,
            'model_info': {
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'device': next(model.parameters()).device.type,
                'input_shape': list(inputs.shape),
                'input_dtype': str(inputs.dtype)
            }
        }
        
        # Add profiling if requested
        if include_profiling and profiler and profiler.enabled:
            profile_results = profiler.profile_model(model, inputs, output_dir)
            results['profiling'] = profile_results
        
        return results