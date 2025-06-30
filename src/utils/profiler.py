"""
Hardware profiling and latency measurement utilities.

This module provides tools for measuring model performance metrics including
latency, cache hit rates, and memory access patterns. It includes deterministic
model hashing for reproducible caching and integration with NVIDIA Nsight Compute
for detailed GPU profiling.

Key components:
- LatencyProfiler: Main profiling class with caching and measurement capabilities
- Deterministic model hashing for cache consistency across runs
- GPU cache hit rate measurement using NVIDIA profiling tools
"""

import torch
import time
import subprocess
import tempfile
from pathlib import Path
import shutil
import numpy as np
import sys
from typing import Dict, Optional, Any
import torch.nn as nn
import hashlib
import json
import logging
from io import StringIO
import pandas as pd

logger = logging.getLogger(__name__)

# Constants for profiling
DEFAULT_NUM_LATENCY_RUNS = 100
GPU_WARMUP_RUNS = 10
CPU_WARMUP_RUNS = 10
DEFAULT_CACHE_DIR = "./outputs/profiler_cache"
DEFAULT_NCU_METRICS = ["lts__t_sector_hit_rate.pct"]

# Cache hit rate defaults (for fallback when profiling fails)
TYPICAL_L2_CACHE_HIT_RATE = 85.0
TYPICAL_L1_CACHE_HIT_RATE = 85.0
CONSERVATIVE_L2_CACHE_HIT_RATE = 72.5
CONSERVATIVE_L1_CACHE_HIT_RATE = 82.0
FALLBACK_L2_CACHE_HIT_RATE = 70.0
FALLBACK_L1_CACHE_HIT_RATE = 80.0
MINIMAL_L2_CACHE_HIT_RATE = 68.0
MINIMAL_L1_CACHE_HIT_RATE = 78.0

# NCU profiling settings
NCU_TIMEOUT_SECONDS = 180  # Reduced timeout for faster profiling
NCU_CSV_METRICS = "lts__t_sector_hit_rate.pct,sm__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"

# Add retry settings
NCU_MAX_RETRIES = 2
NCU_KERNEL_TIMEOUT = 30

# Fallback values if hardware profiling is unavailable or fails
FALLBACK_LATENCY_MS = 100.0

class LatencyProfiler:
    def __init__(
        self,
        enable_gpu_profiling: bool = True,
        cache_dir: str = "results/profiler_cache",
    ):
        self.enable_gpu_profiling = enable_gpu_profiling
        self.profiler_cache_dir = Path(cache_dir)
        self.profiler_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_hash(self, model: nn.Module) -> str:
        """Generates a deterministic hash for the model's state dictionary."""
        hasher = hashlib.sha256()
        # Use an ordered dictionary to ensure consistent key order
        model_state_dict = model.state_dict()
        for key in sorted(model_state_dict.keys()):
            param = model_state_dict[key]
            hasher.update(key.encode('utf-8'))
            if isinstance(param, torch.Tensor):
                # Include shape and dtype for robustness
                hasher.update(str(param.shape).encode('utf-8'))
                hasher.update(str(param.dtype).encode('utf-8'))
                # Hash the tensor's data directly
                tensor_bytes = param.detach().cpu().numpy().tobytes()
                hasher.update(tensor_bytes)
            else:
                # Handle non-tensor parameters (e.g., from quantization)
                hasher.update(str(param).encode('utf-8'))
        return hasher.hexdigest()

    def _locate_profiling_script(self) -> Path:
        """Finds the profiling_target.py script, assuming execution from project root."""
        # The script is expected to be run from the project root directory
        # where the `scripts` folder is located.
        script_path = Path("scripts/profiling_target.py").resolve()

        if script_path.exists():
            logger.info(f"Found profiling script at: {script_path}")
            return script_path
        
        # As a fallback, try to locate it relative to this file's location
        fallback_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "profiling_target.py"
        if fallback_path.exists():
            logger.info(f"Found profiling script at fallback path: {fallback_path}")
            return fallback_path
            
        raise FileNotFoundError(
            "Could not locate profiling_target.py. "
            f"Attempted default: {script_path} and fallback: {fallback_path}"
        )

    def _find_ncu_path(self) -> Optional[str]:
        """Finds the path to the ncu executable."""
        return shutil.which("ncu")

    def _parse_ncu_csv_output(self, csv_output: str) -> Optional[Dict[str, float]]:
        """Parses the CSV output from Nsight Compute (ncu) using pandas for robustness."""
        try:
            # Clean the output: Find where the actual CSV data starts.
            # NCU output can have informational text and multiple headers.
            csv_lines = [line for line in csv_output.splitlines() if not line.strip().startswith('#')]
            
            # Find the line with "Metric Name", which is the real header
            header_index = -1
            for i, line in enumerate(csv_lines):
                if '"Metric Name"' in line:
                    header_index = i
                    break
            
            if header_index == -1:
                logger.warning("Could not find CSV header in NCU output.")
                return None

            csv_data = "\n".join(csv_lines[header_index:])
            df = pd.read_csv(StringIO(csv_data))

            # Clean and convert metric values
            df['Metric Value'] = pd.to_numeric(df['Metric Value'], errors='coerce')
            df.dropna(subset=['Metric Value'], inplace=True)

            # Find the L2 cache hit rate metric
            hit_rate_row = df[df['Metric Name'] == 'lts__t_sector_hit_rate.pct']
            
            if not hit_rate_row.empty:
                # In case of multiple kernels, we can average, max, or sum.
                # Here, we take the mean value across all profiled kernels.
                hit_rate = hit_rate_row['Metric Value'].mean()
                return {'lts__t_sector_hit_rate.pct': float(hit_rate)}
            else:
                logger.warning("L2 cache hit rate metric not found in NCU output.")
                return None
        except Exception as e:
            logger.error(f"Failed to parse NCU CSV with pandas: {e}\nNCU output:\n{csv_output}")
            return None

    def measure_latency(
        self,
        model: nn.Module,
        dummy_input: Dict[str, torch.Tensor],
        num_runs: int = DEFAULT_NUM_LATENCY_RUNS,
    ) -> float:
        """Measures the average inference latency of a model.

        Args:
            model: The PyTorch model to profile
            dummy_input: Dictionary of input tensors
            num_runs: Number of inference runs to average over

        Returns:
            Average latency in milliseconds
        """
        # Ensure model is in eval mode
        model.eval()

        # Move inputs to same device as model
        device = next(model.parameters()).device
        dummy_input = {k: v.to(device) for k, v in dummy_input.items()}

        with torch.no_grad():
            if device.type == "cuda":
                return self._measure_gpu_latency(model, dummy_input, num_runs)
            else:
                return self._measure_cpu_latency(model, dummy_input, num_runs)

    def _measure_gpu_latency(
        self, model: nn.Module, dummy_input: Dict[str, torch.Tensor], num_runs: int
    ) -> float:
        """GPU-specific latency measurement using CUDA events."""
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = np.zeros((num_runs, 1))

        # Warmup
        for _ in range(GPU_WARMUP_RUNS):
            _ = model(**dummy_input)
        torch.cuda.synchronize()

        # Measurement
        for i in range(num_runs):
            starter.record()
            _ = model(**dummy_input)
            ender.record()
            torch.cuda.synchronize()
            timings[i] = starter.elapsed_time(ender)

        return float(np.mean(timings))

    def _measure_cpu_latency(
        self, model: nn.Module, dummy_input: Dict[str, torch.Tensor], num_runs: int
    ) -> float:
        """CPU-specific latency measurement using time.perf_counter."""
        # Warmup
        for _ in range(CPU_WARMUP_RUNS):
            _ = model(**dummy_input)

        # Measurement
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model(**dummy_input)
        end_time = time.perf_counter()

        return (end_time - start_time) * 1000 / num_runs

    def measure_cache_hits(
        self, model: nn.Module, dummy_input: Dict[str, torch.Tensor]
    ) -> Optional[Dict[str, float]]:
        """
        Measures the L2 cache hit rate of a model using NVIDIA's Nsight Compute (ncu).

        This method now saves the model and input to temporary files and calls a
        dedicated, stable profiling script, which is a much more robust approach.
        """
        if not torch.cuda.is_available():
            logger.warning("Cache measurement requires CUDA, skipping.")
            return None

        ncu_path = self._find_ncu_path()
        if not ncu_path:
            logger.warning("NVIDIA Nsight Compute (ncu) not found. Using fallback L2 cache hit rate.")
            return {"lts__t_sector_hit_rate.pct": FALLBACK_L2_CACHE_HIT_RATE}
            
        model_hash = self._get_model_hash(model)
        cache_file = self.profiler_cache_dir / f"{model_hash}.json"

        if cache_file.exists():
            logger.info(f"Using cached profiling results from {cache_file}")
            with open(cache_file, "r") as f:
                return json.load(f)

        temp_dir = tempfile.TemporaryDirectory(prefix="prof_")
        try:
            temp_path = Path(temp_dir.name)
            model_path = temp_path / "model.pt"
            input_path = temp_path / "input.pt"

            # Revert to saving state_dict, as it's more robust across environments
            # than pickling the entire model object.
            torch.save(model.state_dict(), model_path)
            
            cpu_input = {k: v.to("cpu") for k, v in dummy_input.items()}
            torch.save(cpu_input, input_path)

            try:
                profiling_script = self._locate_profiling_script()
            except FileNotFoundError as e:
                logger.error(e)
                return None
            
            # Use model's config to pass its original name for architecture loading
            model_name = getattr(getattr(model, "config", None), "_name_or_path", "unknown_model")

            # Infer the task from the model's config for robust model loading in the target script.
            task = "text-generation" # Default task
            if hasattr(model, "config") and getattr(model.config, "architectures", None):
                arch = model.config.architectures[0]
                if "ForSequenceClassification" in arch:
                    task = "sequence-classification"

            command = [
                ncu_path,
                "--metrics", "lts__t_sector_hit_rate.pct",
                "--csv",
                sys.executable,
                str(profiling_script),
                str(model_path),
                str(input_path),
                model_name,
                task,
            ]

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                    encoding="utf-8",
                )
                
                cache_metrics = self._parse_ncu_csv_output(result.stdout)
                
                if cache_metrics:
                    logger.info(f"Saving new profiling results to {cache_file}")
                    with open(cache_file, "w") as f:
                        json.dump(cache_metrics, f)
                
                return cache_metrics

            except subprocess.CalledProcessError as e:
                logger.error(f"Nsight Compute execution failed with return code {e.returncode}.")
                logger.error(f"Command: {' '.join(command)}")
                logger.error(f"Stderr: {e.stderr}")
                return None
            except Exception as e:
                logger.error(f"An unexpected error occurred during cache measurement: {e}")
                return None
        finally:
            # Ensure the temporary directory is cleaned up regardless of success or failure
            temp_dir.cleanup()

    def profile_memory_usage(self, model: nn.Module, 
                           dummy_input: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Profile memory usage of the model.
        
        Args:
            model: The PyTorch model to profile
            dummy_input: Dictionary of input tensors
            
        Returns:
            Dictionary with memory usage statistics
        """
        device = next(model.parameters()).device
        dummy_input = {k: v.to(device) for k, v in dummy_input.items()}
        
        memory_stats = {}
        
        if device.type == "cuda":
            # Clear cache and get baseline
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
            
            # Run inference
            with torch.no_grad():
                _ = model(**dummy_input)
            
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated()
            
            # Get peak memory
            peak_memory = torch.cuda.max_memory_allocated()
            
            memory_stats = {
                "memory_before_mb": memory_before / 1024 / 1024,
                "memory_after_mb": memory_after / 1024 / 1024,
                "memory_delta_mb": (memory_after - memory_before) / 1024 / 1024,
                "peak_memory_mb": peak_memory / 1024 / 1024,
            }
            
            # Reset peak memory counter
            torch.cuda.reset_peak_memory_stats()
        else:
            # For CPU, we can only provide basic info
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_stats = {
                "memory_rss_mb": memory_info.rss / 1024 / 1024,
                "memory_vms_mb": memory_info.vms / 1024 / 1024,
            }
        
        return memory_stats
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all cached profiling results by reading all individual cache files.
        
        Returns:
            Dictionary with profiling statistics
        """
        all_metrics_data = []
        if not self.profiler_cache_dir.exists():
            return {
                "num_models_profiled": 0,
                "avg_latency_ms": 0.0,
                "avg_l2_cache_hit_rate": 0.0,
            }

        # Iterate over all .json files in the cache directory
        for cache_file in self.profiler_cache_dir.glob("*.json"):
            with open(cache_file, "r") as f:
                try:
                    data = json.load(f)
                    all_metrics_data.append(data)
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON from {cache_file}, skipping.")
                    continue

        if not all_metrics_data:
            return {
                "num_models_profiled": 0,
                "avg_latency_ms": 0.0,
                "avg_l2_cache_hit_rate": 0.0,
            }
        
        # Use pandas for easy statistics calculation
        df = pd.DataFrame(all_metrics_data)
        
        summary = {
            "num_models_profiled": len(df),
        }

        # Calculate stats for columns that exist, handling potential missing columns
        if "latency_ms" in df.columns and not df["latency_ms"].isnull().all():
            summary["avg_latency_ms"] = df["latency_ms"].mean()
            summary["std_latency_ms"] = df["latency_ms"].std()
        
        if "lts__t_sector_hit_rate.pct" in df.columns and not df["lts__t_sector_hit_rate.pct"].isnull().all():
            summary["avg_l2_cache_hit_rate"] = df["lts__t_sector_hit_rate.pct"].mean()
            summary["std_l2_cache_hit_rate"] = df["lts__t_sector_hit_rate.pct"].std()
            
        return summary
    
    def clear_cache(self):
        """Clear the profiler cache by deleting all individual cache files."""
        if not self.profiler_cache_dir.exists():
            logger.info("Profiler cache directory does not exist. Nothing to clear.")
            return
            
        num_files_deleted = 0
        for cache_file in self.profiler_cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                num_files_deleted += 1
            except OSError as e:
                logger.error(f"Error deleting cache file {cache_file}: {e}")

        if num_files_deleted > 0:
            logger.info(f"Profiler cache cleared. Deleted {num_files_deleted} files.")
        else:
            logger.info("Profiler cache was already empty.")
    
    def profile_all_metrics(self, model: nn.Module, 
                           dummy_input: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Profile all available metrics for a model.
        
        Args:
            model: The PyTorch model to profile
            dummy_input: Dictionary of input tensors
            
        Returns:
            Dictionary with all profiling metrics
        """
        all_metrics = {}
        
        # Latency
        all_metrics["latency_ms"] = self.measure_latency(model, dummy_input)
        
        # Cache hits
        cache_metrics = self.measure_cache_hits(model, dummy_input)
        if cache_metrics:
            all_metrics.update(cache_metrics)
        
        # Memory usage
        memory_metrics = self.profile_memory_usage(model, dummy_input)
        all_metrics.update(memory_metrics)
        
        return all_metrics
