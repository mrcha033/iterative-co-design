#!/usr/bin/env python3
"""
Dedicated Profiling Target Script for NCU.

This script is designed to be called by `ncu` (NVIDIA's Nsight Compute)
from the LatencyProfiler. It loads a model state dictionary and a dummy input
tensor, then runs a single forward pass for hardware-level profiling.

This approach avoids the complexities and inaccuracies of generating temporary
scripts or profiling simplified/mock models.

Usage:
    ncu --metrics <metrics> --csv python scripts/profiling_target.py <model_path> <input_path> <model_name_or_path> <task>
"""

import sys
from pathlib import Path

# Ensure the project's local 'src' package directory has highest import precedence
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Invalidate caches to ensure local modules are imported
import importlib
for _pkg in ("utils", "co_design", "models"):
    if _pkg in sys.modules:
        del sys.modules[_pkg]
importlib.invalidate_caches()

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification

def main():
    if len(sys.argv) != 5:
        print("Usage: python profiling_target.py <model_path> <input_path> <model_name_or_path> <task>")
        sys.exit(1)

    model_path = sys.argv[1]
    input_path = sys.argv[2]
    model_name_or_path = sys.argv[3]
    task = sys.argv[4]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Error: Profiling target requires a CUDA-enabled device.")
        sys.exit(1)

    try:
        # Load the model architecture from the pretrained name, then load the state dict
        # This is more robust than pickling the whole model class.
        config = AutoConfig.from_pretrained(model_name_or_path)
        
        if task == "sequence-classification":
            model = AutoModelForSequenceClassification.from_config(config)
        else: # "text-generation" or other tasks default to causal LM
            model = AutoModelForCausalLM.from_config(config)

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device).eval()

        # Load the dummy input
        dummy_input = torch.load(input_path, map_location=device)
        
        # Ensure input is on the correct device
        dummy_input = {k: v.to(device) for k, v in dummy_input.items()}

        # Run a single forward pass for ncu to profile
        with torch.no_grad():
            _ = model(**dummy_input)
            
        # Synchronization might be needed for accurate kernel timing
        torch.cuda.synchronize()

    except Exception as e:
        # Print error to stderr so it can be captured by the calling process
        print(f"Error during profiling target execution: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 