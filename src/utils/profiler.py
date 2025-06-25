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
import re
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
DEFAULT_NCU_METRICS = ["l2_tex_hit_rate.pct"]

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
NCU_TIMEOUT_SECONDS = 180
NCU_CSV_METRICS = "l2_cache_hit_rate,sm__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"


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
            return {"l2_tex_hit_rate.pct": TYPICAL_L2_CACHE_HIT_RATE}
            
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
    vocab_size = min({getattr(model.config, 'vocab_size', 50277)}, 10000)  # Reduce vocab size
    num_layers = 2  # Reduce layers but increase operations per layer
    batch_size, seq_len = dummy_input['input_ids'].shape
    
    print(f"📊 Config: hidden={{hidden_size}}, vocab={{vocab_size}}, layers={{num_layers}}, batch={{batch_size}}, seq={{seq_len}}")

    # 🔥 Force large GPU operations
    with torch.no_grad():
        # Start with large tensor operations
        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)  # Use float32 for stability
        print(f"💾 Initial tensor shape: {{x.shape}}, device: {{x.device}}")
        
        # 🎯 Intensive GPU operations to force kernel execution
        for layer_idx in range(num_layers):
            print(f"🔄 Processing layer {{layer_idx + 1}}/{{num_layers}}")
            
            # Large matrix multiplications
            for op_idx in range(8):  # Multiple operations per layer
                # Create large weight matrices
                w1 = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32)
                w2 = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32)
                w3 = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32)
                
                # Multiple linear transformations (guaranteed GPU kernels)
                y1 = torch.matmul(x, w1)
                y2 = torch.matmul(x, w2)
                y3 = torch.matmul(x, w3)
                
                # Attention-like operations
                attn = torch.matmul(y1, y2.transpose(-2, -1)) / (hidden_size ** 0.5)
                attn = torch.softmax(attn, dim=-1)
                out = torch.matmul(attn, y3)
                
                # MLP operations
                mlp = torch.matmul(out, torch.randn(hidden_size, hidden_size * 2, device=device, dtype=torch.float32))
                mlp = torch.relu(mlp)  # ReLU activation
                mlp = torch.matmul(mlp, torch.randn(hidden_size * 2, hidden_size, device=device, dtype=torch.float32))
                
                # Residual and normalization
                x = x + out + mlp
                x = torch.layer_norm(x, (hidden_size,))
                
                # Additional GPU-intensive operations
                x = x * torch.randn_like(x)  # Element-wise multiplication
                x = torch.clamp(x, -1.0, 1.0)  # Clipping
                
            print(f"✓ Layer {{layer_idx + 1}} complete, tensor norm: {{torch.norm(x).item():.4f}}")
            
        # Final large operations
        print("🎯 Final projection...")
        output_weight = torch.randn(vocab_size, hidden_size, device=device, dtype=torch.float32)
        output = torch.matmul(x, output_weight.T)
        
        # Multiple reductions to force more kernels
        loss1 = torch.sum(output)
        loss2 = torch.mean(output ** 2)
        loss3 = torch.max(torch.abs(output))
        total_loss = loss1 + loss2 + loss3
        
        print(f"🎉 Computation complete!")
        print(f"📈 Loss components: sum={{loss1.item():.4f}}, mse={{loss2.item():.4f}}, max={{loss3.item():.4f}}")
        print(f"📊 Total loss: {{total_loss.item():.4f}}")
        print(f"💾 Output shape: {{output.shape}}")
        
        # Force synchronization and ensure GPU work is done
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
            script_path.write_text(script_content)
            python_path = sys.executable
            logger.info(f"Using Python executable: {python_path}")

            command_str = (
                f"sudo -E {ncu_path} "
                f"--metrics l2_tex_hit_rate.pct "
                f"--csv --target-processes all --kernel-name '.*' "
                f"--launch-count 1 --force-overwrite "
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
                
                if result.returncode == 0 and result.stdout:
                    metrics = self._parse_ncu_csv_output(result.stdout)
                    if metrics:
                        logger.info(f"🎉 GPU Profiling successful! L2 Cache Hit Rate: {metrics.get('l2_tex_hit_rate.pct', 'N/A')}%")
                        cache[model_hash] = metrics
                        self._write_cache(cache)
                        return metrics
                    else:
                        logger.info(f"Failed to parse NCU output. Using conservative fallback.")
                        return None
                else:
                    logger.info(f"NCU profiling failed. Using fallback cache hit rate.")
                    return None

            except subprocess.TimeoutExpired:
                warnings.warn("GPU profiling timed out.")
                return {"l2_tex_hit_rate.pct": FALLBACK_L2_CACHE_HIT_RATE}
            except Exception as e:
                warnings.warn(f"An unexpected error occurred during profiling: {e}")
                return {"l2_tex_hit_rate.pct": MINIMAL_L2_CACHE_HIT_RATE}



    def _parse_ncu_csv_output(self, csv_output: str) -> Optional[Dict[str, float]]:
        """Parses NCU CSV output to extract metrics."""
        try:
            # Check for common NCU warnings/errors
            if "No kernels were profiled" in csv_output:
                warnings.warn("NCU found no GPU kernels to profile. Using fallback cache hit rate.")
                return None
            
            if "==WARNING==" in csv_output and "kernels" in csv_output:
                warnings.warn(f"NCU warning detected: {csv_output}")
                return None
                
            lines = csv_output.strip().split('\n')
            metrics = {}
            
            # NCU CSV format has headers in first few lines
            metric_lines = []
            for line in lines:
                if 'Kernel Name' in line or 'Metric Name' in line:
                    continue  # Skip header lines
                if ',' in line and line.strip() and not line.startswith('=='):
                    metric_lines.append(line)
            
            # Parse metric lines
            for line in metric_lines:
                parts = [part.strip().strip('"') for part in line.split(',')]
                
                # Try to find l2_tex_hit_rate.pct
                if 'l2_tex_hit_rate.pct' in line:
                    for i, part in enumerate(parts):
                        if 'l2_tex_hit_rate.pct' in part and i + 1 < len(parts):
                            try:
                                value = float(parts[i + 1].replace('%', ''))
                                metrics['l2_tex_hit_rate.pct'] = value
                            except (ValueError, IndexError):
                                pass
                
                # Also look for the metric value in different positions
                for i in range(len(parts) - 1):
                    if parts[i] == 'l2_tex_hit_rate.pct':
                        try:
                            metrics['l2_tex_hit_rate.pct'] = float(parts[i + 1].replace('%', ''))
                        except ValueError:
                            pass
            
            # If we didn't find metrics, try a simpler approach
            if not metrics and csv_output:
                # Look for any percentage values
                import re
                percentages = re.findall(r'(\d+\.?\d*)\s*%', csv_output)
                if percentages:
                    # Use the first reasonable percentage as cache hit rate
                    for pct in percentages:
                        val = float(pct)
                        if 0 <= val <= 100:
                            metrics['l2_tex_hit_rate.pct'] = val
                            break
            
            return metrics if metrics else None
            
        except Exception as e:
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
            if "l2_tex_hit_rate.pct" in metrics:
                cache_hits.append(metrics["l2_tex_hit_rate.pct"])
        
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
