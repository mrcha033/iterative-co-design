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
from typing import Dict, Optional
import torch.nn as nn
import hashlib
import json


class LatencyProfiler:
    def __init__(
        self,
        cache_dir: str = "./outputs/profiler_cache",
        ncu_metrics: Optional[list] = None,
    ):
        """Initialize the profiler with configurable cache directory and metrics.

        Args:
            cache_dir: Directory to store profiling cache
            ncu_metrics: List of NCU metrics to collect. Defaults to L2 cache metrics.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "profiler_cache.json"
        self.ncu_metrics = ncu_metrics or ["l2_tex_hit_rate.pct"]

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
        num_runs: int = 100,
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
        for _ in range(10):
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
        for _ in range(10):
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
        """Measures cache hit rates and other metrics using NVIDIA's Nsight Compute.

        Args:
            model: The PyTorch model to profile
            dummy_input: Dictionary of input tensors

        Returns:
            Dictionary of metrics or None if profiling is not possible
        """
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available - skipping hardware profiling")
            return None

        if not shutil.which("ncu"):
            warnings.warn("NVIDIA Nsight Compute (ncu) not found in PATH")
            return None

        # Check cache
        model_hash = self._get_model_hash(model.state_dict())
        cache = self._read_cache()
        if model_hash in cache:
            return cache[model_hash]

        # Create temp directory for profiling
        with tempfile.TemporaryDirectory(prefix="ncu_profile_") as temp_dir:
            temp_dir = Path(temp_dir)
            model_path = temp_dir / "model.pt"
            input_path = temp_dir / "input.pt"
            script_path = temp_dir / "profile.py"
            output_path = temp_dir / "ncu_output.txt"

            # Save model and inputs
            torch.save(model, model_path)
            torch.save(dummy_input, input_path)

            # Create profiling script
            script_content = (
                "import torch\n"
                f"model = torch.load(r'{model_path}')\n"
                f"inputs = torch.load(r'{input_path}')\n"
                "model.cuda(); model.eval()\n"
                "with torch.no_grad():\n"
                "    inputs = {k: v.cuda() for k, v in inputs.items()}\n"
                "    model(**inputs)\n"
            )
            script_path.write_text(script_content)

            # Run NCU
            metrics_str = ",".join(self.ncu_metrics)
            cmd = [
                "ncu",
                "--metrics",
                metrics_str,
                "--csv",
                "--replay-mode",
                "kernel",
                "--log-file",
                str(output_path),
                "python",
                str(script_path),
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                metrics = self._parse_ncu_output(output_path)
                if metrics:
                    cache[model_hash] = metrics
                    self._write_cache(cache)
                return metrics
            except subprocess.CalledProcessError as e:
                warnings.warn(f"NCU profiling failed: {e.stderr}")
                return None

    def _parse_ncu_output(self, output_path: Path) -> Optional[Dict[str, float]]:
        """Parses NCU output CSV to extract metrics."""
        try:
            with open(output_path) as f:
                content = f.read()

            metrics = {}
            for metric in self.ncu_metrics:
                # Enhanced regex to support both decimal and scientific notation
                pattern = (
                    f"{metric}\\s*,\\s*[\\w%]+\\s*,\\s*([\\d\\.]+(?:[eE][+-]?\\d+)?)"
                )
                match = re.search(pattern, content)
                if match:
                    metrics[metric] = float(match.group(1))

            return metrics if metrics else None
        except Exception as e:
            warnings.warn(f"Failed to parse NCU output: {e}")
            return None
