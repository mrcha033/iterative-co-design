import torch
import time
import re
import subprocess
import tempfile
import os
from pathlib import Path
import shutil

def measure_latency(model: torch.nn.Module, dummy_input: torch.Tensor, on_gpu: bool = True) -> float:
    """
    Measures the average inference latency of a model.

    Args:
        model: The model to profile.
        dummy_input: A sample input tensor for the model.
        on_gpu: Flag to indicate if profiling should be run on a GPU.

    Returns:
        The average latency in milliseconds.
    """
    if on_gpu and torch.cuda.is_available():
        model.cuda()
        dummy_input = dummy_input.cuda()
        # Warm-up runs
        for _ in range(10):
            _ = model(dummy_input)
        torch.cuda.synchronize()
    else:
        # Warm-up runs for CPU
        for _ in range(10):
            _ = model(dummy_input)

    # Measurement runs
    timings = []
    for _ in range(100):
        if on_gpu and torch.cuda.is_available():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            _ = model(dummy_input)
            end_time.record()
            torch.cuda.synchronize()
            timings.append(start_time.elapsed_time(end_time))
        else:
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            timings.append((end_time - start_time) * 1000)

    avg_latency_ms = sum(timings) / len(timings)
    return avg_latency_ms

def measure_cache_hits(config_path: str, state_dict_path: str, experiment_script_path: str = 'scripts/run_experiment.py') -> float:
    """
    Measures the L2 cache hit rate using NVIDIA Nsight Compute (ncu).

    Note: This function requires NVIDIA Nsight Compute (ncu) to be installed and
    in the system's PATH. If ncu is not found, it will print a warning and return 0.0.
    """
    if not shutil.which("ncu"):
        print("\n[Warning] NVIDIA Nsight Compute (ncu) not found in PATH.")
        print("          Skipping L2 cache hit rate measurement. Will return 0.0.")
        return 0.0

    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Skipping cache hit measurement.")
        return 0.0

    # Command to run the experiment script in a special profiling mode
    profile_command = [
        'python',
        experiment_script_path,
        '--config', config_path,
        '--_profile_for_ncu', state_dict_path # Internal, undocumented flag
    ]

    ncu_command = [
        'ncu',
        '--metrics', 'l2_tex_hit_rate.pct',
        '--log-file', 'ncu_output.txt',
    ]
    ncu_command.extend(profile_command)

    try:
        print("Running Nsight Compute profiler... This may take a moment.")
        # We use a file for output because parsing stdout can be tricky with ncu's verbose output
        result = subprocess.run(ncu_command, capture_output=True, text=True, check=True)
        
        with open('ncu_output.txt', 'r') as f:
            output = f.read()

        # Regex to find the L2 cache hit rate value
        # Example line: "l2_tex_hit_rate.pct                                     ,        pct,        85.7"
        match = re.search(r"l2_tex_hit_rate\.pct\s+,\s+pct,\s+([\d\.]+)", output)

        if match:
            hit_rate = float(match.group(1))
            print(f"NCU Profiler found L2 Cache Hit Rate: {hit_rate:.2f}%")
            return hit_rate
        else:
            raise RuntimeError("Could not parse L2 cache hit rate from ncu output. "
                               "Please check if ncu ran correctly and produced the expected metric.")
    except FileNotFoundError:
        raise RuntimeError("ncu command not found. Please ensure NVIDIA Nsight Compute is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Nsight Compute execution failed. This can happen if the model fails to run on the GPU.\nNCU stderr:\n{e.stderr}")
    finally:
        if os.path.exists('ncu_output.txt'):
            os.remove('ncu_output.txt') 