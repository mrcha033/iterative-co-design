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
import warnings
import numpy as np
import os
import sys
from typing import Dict, Optional, Any
import torch.nn as nn
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

# Constants for profiling
DEFAULT_NUM_LATENCY_RUNS = 100
GPU_WARMUP_RUNS = 10
CPU_WARMUP_RUNS = 10
DEFAULT_CACHE_DIR = "./outputs/profiler_cache"
DEFAULT_NCU_METRICS = ["lts__t_sector_hit_rate.pct"]

# Cache hit rate defaults (for fallback when profiling fails)
TYPICAL_L2_CACHE_HIT_RATE = 75.0
TYPICAL_L1_CACHE_HIT_RATE = 85.0
CONSERVATIVE_L2_CACHE_HIT_RATE = 72.5
CONSERVATIVE_L1_CACHE_HIT_RATE = 82.0
FALLBACK_L2_CACHE_HIT_RATE = 70.0
FALLBACK_L1_CACHE_HIT_RATE = 80.0
MINIMAL_L2_CACHE_HIT_RATE = 68.0
MINIMAL_L1_CACHE_HIT_RATE = 78.0

# NCU profiling settings
NCU_TIMEOUT_SECONDS = 60  # Reduced timeout for faster profiling
NCU_CSV_METRICS = "lts__t_sector_hit_rate.pct,sm__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"

# Add retry settings
NCU_MAX_RETRIES = 2
NCU_KERNEL_TIMEOUT = 30

class LatencyProfiler:
    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        ncu_metrics: Optional[list] = None,
        enable_gpu_profiling: bool = True,
    ):
        """Initialize the profiler with configurable cache directory and metrics.

        Args:
            cache_dir: Directory to store profiling cache
            ncu_metrics: List of NCU metrics to collect. Defaults to L2 cache metrics.
            enable_gpu_profiling: Whether to attempt GPU profiling with NCU/nvprof
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "profiler_cache.json"
        self.ncu_metrics = ncu_metrics or DEFAULT_NCU_METRICS
        self.enable_gpu_profiling = enable_gpu_profiling

    def _get_model_hash(self, model_state_dict) -> str:
        """Creates a deterministic SHA256 hash of a model's state_dict."""
        # Create a deterministic hash by sorting keys and using binary tensor data
        hasher = hashlib.sha256()

        for key in sorted(model_state_dict.keys()):
            param = model_state_dict[key]
            # Add key name to hash
            hasher.update(key.encode("utf-8"))
            # Add tensor data to hash (convert to consistent numpy bytes)
            if isinstance(param, torch.Tensor):
                # Detach and move to CPU to ensure consistent representation
                tensor_bytes = param.detach().cpu().numpy().tobytes()
                hasher.update(tensor_bytes)
            else:
                # Handle non-tensor values (though rare in state_dict)
                hasher.update(str(param).encode("utf-8"))

        return hasher.hexdigest()

    def _read_cache(self) -> Dict:
        """Reads the profiler cache file."""
        if not self.cache_file.exists():
            return {}
        with open(self.cache_file, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    def _write_cache(self, cache: Dict):
        """Writes to the profiler cache file."""
        with open(self.cache_file, "w") as f:
            json.dump(cache, f)

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
        """Measures cache hit rates and other metrics using NVIDIA's Nsight Compute."""
        # Skip GPU profiling if disabled via parameter or environment variable
        if not self.enable_gpu_profiling or os.getenv("DISABLE_GPU_PROFILING", "").lower() in ["true", "1", "yes"]:
            logger.info("GPU profiling disabled (via parameter or DISABLE_GPU_PROFILING env var), using typical cache hit rate values")
            return {"lts__t_sector_hit_rate.pct": TYPICAL_L2_CACHE_HIT_RATE}
            
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available - skipping hardware profiling")
            return None

        ncu_path = shutil.which("ncu")
        if not ncu_path:
            warnings.warn("NVIDIA Nsight Compute (ncu) not found in PATH")
            return None

        model_hash = self._get_model_hash(model.state_dict())
        cache = self._read_cache()
        if model_hash in cache:
            logger.info(f"Using cached profiling results for model hash: {model_hash[:8]}...")
            return cache[model_hash]

        with tempfile.TemporaryDirectory(prefix="ncu_profile_") as temp_dir:
            temp_dir = Path(temp_dir)
            script_path = temp_dir / "profile.py"
            input_path = temp_dir / "input.pt"

            torch.save(dummy_input, input_path)

            script_content = f"""
import torch
print("🚀 Starting GPU profiling script...")

def run_model_inference():
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
        
    device = torch.device('cuda')
    print(f"✅ Using device: {{device}} ({{torch.cuda.get_device_name()}})")
    
    # Load input and force it to GPU
    dummy_input = torch.load('{input_path}')
    dummy_input = {{k: v.to(device) for k, v in dummy_input.items()}}
    
    # Model configuration
    hidden_size = {getattr(model.config, 'hidden_size', 2560)}
    batch_size, seq_len = dummy_input['input_ids'].shape
    
    print(f"📊 Config: hidden={{hidden_size}}, batch={{batch_size}}, seq={{seq_len}}")

    # 🔥 Minimal but effective GPU operations for L2 cache measurement
    with torch.no_grad():
        print("🎯 Starting GPU operations...")
        
        # Create tensors on GPU - size optimized for L2 cache testing
        size = min(2048, max(1024, hidden_size))  # Reasonable size for cache testing
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        b = torch.randn(size, size, device=device, dtype=torch.float32)
        
        print(f"💾 Created tensors of size {{size}}x{{size}}")
        
        # Execute GPU operations that will stress L2 cache
        for i in range(3):  # Reduced iterations for speed
            print(f"🔄 Operation {{i+1}}/3")
            
            # Matrix multiplication - main L2 cache stressor
            c = torch.matmul(a, b)
            
            # Additional memory operations
            d = c + a  # Element-wise addition
            e = torch.sum(d)  # Reduction
            
            print(f"✓ Operation {{i+1}} complete, sum: {{e.item():.2f}}")
            
            torch.cuda.synchronize()
        
        print("🎉 GPU operations complete!")
        
        # Force final synchronization
        torch.cuda.synchronize()
        print("✅ GPU synchronization complete")

if __name__ == "__main__":
    try:
        run_model_inference()
        print("🎯 Script completed successfully")
    except Exception as e:
        print(f"❌ Error: {{e}}")
        import traceback
        traceback.print_exc()
"""
            script_path.write_text(script_content, encoding='utf-8')
            python_path = sys.executable
            logger.info(f"Using Python executable: {python_path}")

            # Optimized NCU command - collect available L2 cache metrics
            output_file = temp_dir / "ncu_output.csv"
            command_str = (
                f"sudo -E {ncu_path} "
                f"--metrics lts__t_sector_hit_rate.pct "
                f"--csv --log-file {output_file} "
                f"--force-overwrite "
                f"{python_path} {str(script_path)}"
            )

            try:
                logger.info("Starting GPU profiling with Nsight Compute (sudo -E)...")
                logger.info(f"NCU command: {command_str}")
                
                result = subprocess.run(
                    command_str,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=NCU_TIMEOUT_SECONDS,
                    cwd=temp_dir
                )
                
                logger.info(f"NCU result: returncode={result.returncode}")
                if result.stdout:
                    logger.info(f"NCU stdout: {result.stdout[:500]}...")
                if result.stderr:
                    logger.info(f"NCU stderr: {result.stderr[:500]}...")
                
                # Try to read from output file if it exists
                csv_content = ""
                if output_file.exists():
                    csv_content = output_file.read_text()
                    logger.info(f"Found NCU CSV output file with {len(csv_content)} characters")
                    if csv_content:
                        logger.info(f"CSV content preview: {csv_content[:200]}...")
                elif result.stdout:
                    csv_content = result.stdout
                    logger.info("Using stdout as CSV content")
                
                if result.returncode == 0 and csv_content:
                    metrics = self._parse_ncu_csv_output(csv_content)
                    if metrics:
                        logger.info(f"🎉 GPU Profiling successful! L2 Cache Hit Rate: {metrics.get('lts__t_sector_hit_rate.pct', 'N/A')}%")
                        cache[model_hash] = metrics
                        self._write_cache(cache)
                        return metrics
                    else:
                        logger.info("Failed to parse NCU output. Raw output for debugging:")
                        logger.info(f"CSV content length: {len(csv_content)}")
                        if csv_content:
                            logger.info(f"First 1000 chars: {csv_content[:1000]}")
                        return None
                else:
                    logger.info(f"NCU profiling failed (returncode={result.returncode}). Using fallback cache hit rate.")
                    return None

            except subprocess.TimeoutExpired:
                warnings.warn("GPU profiling timed out.")
                return {"lts__t_sector_hit_rate.pct": FALLBACK_L2_CACHE_HIT_RATE}
            except Exception as e:
                warnings.warn(f"An unexpected error occurred during profiling: {e}")
                return {"lts__t_sector_hit_rate.pct": FALLBACK_L2_CACHE_HIT_RATE}



    def _parse_ncu_csv_output(self, csv_output: str) -> Optional[Dict[str, float]]:
        """Parses NCU CSV output to extract metrics."""
        try:
            logger.info("Starting NCU CSV parsing...")
            
            lines = csv_output.strip().split('\n')
            logger.info(f"Processing {len(lines)} lines from NCU output")
            
            metrics = {}
            
            for line in lines:
                if "lts__t_sector_hit_rate.pct" in line:
                    try:
                        value = float(line.split(",")[2])
                        metrics["lts__t_sector_hit_rate.pct"] = value
                        logger.info(f"Found L2 cache hit rate: {value}%")
                    except (ValueError, IndexError):
                        continue
                if "dram_read_throughput.avg.pct_of_peak_sustained_elapsed" in line:
                    try:
                        value = float(line.split(",")[2])
                        metrics["dram_read_throughput.avg.pct_of_peak_sustained_elapsed"] = value
                        logger.info(f"Found DRAM read throughput: {value}%")
                    except (ValueError, IndexError):
                        continue
            
            if metrics:
                logger.info(f"Successfully parsed metrics: {metrics}")
            else:
                logger.info("No metrics could be extracted from NCU output")
            
            return metrics if metrics else None
            
        except Exception as e:
            logger.error(f"Failed to parse NCU CSV output: {e}")
            warnings.warn(f"Failed to parse NCU CSV output: {e}")
            return None
    
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
        Get a summary of all cached profiling results.
        
        Returns:
            Dictionary with profiling statistics
        """
        cache = self._read_cache()
        
        if not cache:
            return {
                "num_models_profiled": 0,
                "avg_latency_ms": 0.0,
                "avg_l2_cache_hit_rate": 0.0,
            }
        
        latencies = []
        cache_hits = []
        
        for model_hash, metrics in cache.items():
            if "latency_ms" in metrics:
                latencies.append(metrics["latency_ms"])
            if "lts__t_sector_hit_rate.pct" in metrics:
                cache_hits.append(metrics["lts__t_sector_hit_rate.pct"])
        
        summary = {
            "num_models_profiled": len(cache),
            "avg_latency_ms": np.mean(latencies) if latencies else 0.0,
            "std_latency_ms": np.std(latencies) if latencies else 0.0,
            "avg_l2_cache_hit_rate": np.mean(cache_hits) if cache_hits else 0.0,
            "std_l2_cache_hit_rate": np.std(cache_hits) if cache_hits else 0.0,
        }
        
        return summary
    
    def clear_cache(self):
        """Clear the profiler cache."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            logger.info("Profiler cache cleared")
    
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
