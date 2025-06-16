import torch
import time
import re
import subprocess
import tempfile
import os

def measure_latency(model, dummy_input, warmup_runs=5, measure_runs=20):
    """
    Measures the average latency of a model's forward pass.

    Args:
        model: The PyTorch model to profile.
        dummy_input: A tensor of appropriate shape to feed to the model.
        warmup_runs: Number of initial runs to discard.
        measure_runs: Number of runs to average for the measurement.

    Returns:
        The average latency in milliseconds.
    """
    # Move model and input to GPU if available
    if torch.cuda.is_available():
        model.cuda()
        dummy_input = dummy_input.cuda()
        torch.cuda.synchronize()

    # Warm-up runs
    for _ in range(warmup_runs):
        _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measurement runs
    timings = []
    for _ in range(measure_runs):
        start_time = time.perf_counter()
        _ = model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        timings.append((end_time - start_time) * 1000) # Convert to ms

    return sum(timings) / len(timings)

def measure_cache_hits(config_path: str, state_dict_path: str, experiment_script_path: str = 'scripts/run_experiment.py') -> float:
    """
    Measures the L2 cache hit rate of a model's forward pass using NVIDIA Nsight Compute (ncu).

    This function calls the main experiment script in a special profiling mode.

    Args:
        config_path: Path to the experiment's YAML config file.
        state_dict_path: Path to the .pt file containing the model's state_dict to profile.
        experiment_script_path: Path to the main experiment runner script.

    Returns:
        The L2 cache hit rate as a float, or raises an error if it cannot be parsed.
    """
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