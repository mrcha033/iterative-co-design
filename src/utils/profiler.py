import torch
import time
import re
import subprocess
import tempfile
import os
from pathlib import Path
import shutil
from omegaconf import OmegaConf
import warnings
import numpy as np
from typing import Dict
import torch.nn as nn
import hashlib
import json

PROFILER_CACHE_PATH = Path("./results/.profiler_cache.json")

def _get_model_hash(model_state_dict) -> str:
    """Creates a SHA256 hash of a model's state_dict."""
    s = str(model_state_dict)
    return hashlib.sha256(s.encode()).hexdigest()

def _read_profiler_cache() -> Dict:
    """Reads the profiler cache file."""
    if not PROFILER_CACHE_PATH.exists():
        return {}
    with open(PROFILER_CACHE_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def _write_profiler_cache(cache: Dict):
    """Writes to the profiler cache file."""
    PROFILER_CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(PROFILER_CACHE_PATH, "w") as f:
        json.dump(cache, f)

def measure_latency(model: nn.Module, dummy_input: Dict[str, torch.Tensor], num_runs: int = 100) -> float:
    """
    Measures the average inference latency of a model.

    This function warms up the model and then measures the average wall-clock
    time over a number of runs for a given dummy input. It supports both
    CPU and GPU execution.

    Args:
        model: The PyTorch model to profile.
        dummy_input: A dictionary of tensors to be used as input for the model.
        num_runs: The number of inference runs to average over.

    Returns:
        The average latency in milliseconds (ms).
    """
    is_cuda = next(model.parameters()).is_cuda
    if is_cuda:
        # GPU latency measurement
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = np.zeros((num_runs, 1))

        # Warm-up
        for _ in range(10):
            _ = model(**dummy_input)

        # Measurement
        for i in range(num_runs):
            starter.record()
            _ = model(**dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time

        avg_latency_ms = np.sum(timings) / num_runs
    else:
        # CPU latency measurement
        # Warm-up
        for _ in range(10):
            _ = model(**dummy_input)
        
        # Measurement
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(**dummy_input)
        end_time = time.time()
        
        avg_latency_ms = (end_time - start_time) * 1000 / num_runs
        
    return avg_latency_ms

def measure_cache_hits(model: nn.Module, dummy_input: Dict[str, torch.Tensor], temp_dir: str = "./results/tmp") -> float:
    """
    Measures the L2 cache hit rate using NVIDIA's Nsight Compute (ncu).

    This function serializes the model and a dummy input, then invokes `ncu`
    as a subprocess to profile a single inference run. It parses the ncu
    output to extract the L2 cache hit rate. A warning is issued if `ncu` is
    not found in the system's PATH.

    A caching mechanism is used to avoid re-profiling identical models. The cache
    is stored at `results/.profiler_cache.json`.

    Args:
        model: The PyTorch model to profile.
        dummy_input: A dictionary of tensors to be used as input.
        temp_dir: A temporary directory to store the serialized model and input.

    Returns:
        The L2 cache hit rate as a percentage, or 0.0 if ncu is not available.
    """
    # Check cache first
    model_hash = _get_model_hash(model.state_dict())
    cache = _read_profiler_cache()
    if model_hash in cache:
        print(f"Found cached L2 hit rate for model hash {model_hash[:8]}: {cache[model_hash]:.2f}%")
        return cache[model_hash]
        
    # Check if ncu is in the PATH
    if not shutil.which("ncu"):
        warnings.warn("`ncu` (NVIDIA Nsight Compute) not found in PATH. Skipping cache hit measurement.")
        return 0.0

    if not torch.cuda.is_available():
        warnings.warn("Warning: CUDA not available. Skipping cache hit measurement.")
        return 0.0

    # Create a temporary directory for profiling artifacts
    profiling_dir = Path(tempfile.mkdtemp(prefix="ncu_profiling_"))
    model_path = profiling_dir / "model.pth"
    input_path = profiling_dir / "input.pth"
    script_path = profiling_dir / "profile_script.py"
    output_log = profiling_dir / "ncu_output.txt"

    # Save model state and dummy input
    torch.save(model.state_dict(), model_path)
    torch.save(dummy_input, input_path)

    # Get the model's class to reconstruct it in the script
    model_class_name = model.__class__.__name__
    # This assumes the model is defined in a discoverable way, e.g., in src.models.wrapper
    # This might need to be made more robust if models are defined elsewhere.
    model_module_name = model.__class__.__module__

    # Create the minimal script for profiling
    script_content = f"""
import torch
from {model_module_name} import {model_class_name}
from src.models.utils import get_model_from_config # Assuming this helper exists

# This is a simplification. We need a way to load the model architecture.
# We assume the model wrapper can be instantiated from a config-like object.
# This part is tricky and depends heavily on the model wrapper implementation.
# For now, let's assume a simple instantiation, which might fail for complex models.
# A more robust solution would pass the hydra config for the model.
# This part of the original code was flawed, and this is a corrected, but still potentially fragile, implementation.

# Let's try to load the model without config first, assuming it has a simple init
# This will likely fail and needs to be adjusted based on the actual model wrapper
try:
    model_config = torch.load('{model_path.parent / "config.pt"}') # Hypothetical config
    model = get_model_from_config(model_config)
except FileNotFoundError:
    # Fallback for models that can be instantiated without a config
    # This is a major assumption.
    print("Warning: Model config not found, attempting to instantiate model directly. This may fail.")
    from transformers import AutoModelForCausalLM, AutoConfig
    model_name = "state-spaces/mamba-130m-hf" # This is a placeholder
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)

model.load_state_dict(torch.load('{model_path}'))
dummy_input = torch.load('{input_path}')
model.cuda()
model.eval()
with torch.no_grad():
    model(**dummy_input)
"""
    
    # This part of the logic is fundamentally hard because the profiler script
    # is a separate process and needs to reconstruct the model architecture.
    # The original implementation was incorrect. A truly robust implementation
    # requires saving the full model or its configuration.
    
    # Let's pivot to a simpler, more robust approach taken by other profiling tools:
    # Save the WHOLE model, not just the state_dict.
    torch.save(model, model_path) # Save the entire model object

    script_content = f"""
import torch
model = torch.load('{model_path}')
dummy_input = torch.load('{input_path}')
model.cuda()
model.eval()
with torch.no_grad():
    # The dummy input is already on the correct device from the main process
    dummy_input = {{k: v.cuda() for k, v in dummy_input.items()}}
    model(**dummy_input)
"""
    with open(script_path, "w") as f:
        f.write(script_content)

    # Construct the ncu command
    cmd = [
        "ncu",
        "--metrics", "l2_tex_hit_rate.pct",
        "--replay-mode", "kernel",
        "--log-file", str(output_log),
        "python",
        str(script_path),
    ]
    
    print(f"Running NCU command: {' '.join(cmd)}")
    hit_rate = 0.0
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=profiling_dir)
        
        with open(output_log, 'r') as f:
            output = f.read()

        match = re.search(r"l2_tex_hit_rate\.pct\s+,\s+pct,\s+([\d\.]+)", output)

        if match:
            hit_rate = float(match.group(1))
            print(f"NCU Profiler found L2 Cache Hit Rate: {hit_rate:.2f}%")
            # Save to cache on success
            cache[model_hash] = hit_rate
            _write_profiler_cache(cache)
        else:
            warnings.warn("Could not parse L2 cache hit rate from ncu output. "
                               "Please check if ncu ran correctly. Caching as 0.0")
            cache[model_hash] = 0.0 # Cache failure to avoid retries
            _write_profiler_cache(cache)

    except FileNotFoundError:
        warnings.warn("ncu command not found. Please ensure NVIDIA Nsight Compute is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        warnings.warn(f"Nsight Compute execution failed. Caching as 0.0. Stderr: {e.stderr}")
        cache[model_hash] = 0.0 # Cache failure to avoid retries
        _write_profiler_cache(cache)
    finally:
        shutil.rmtree(profiling_dir)
        
    return hit_rate 