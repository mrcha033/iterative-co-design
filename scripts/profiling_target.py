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

import torch
import sys
from pathlib import Path
import traceback
from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification

# Ensure the project's 'src' directory is in the path to allow model classes
# to be unpickled correctly by torch.load.
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def main():
    if len(sys.argv) != 5:
        print("Usage: python profiling_target.py <model_path> <input_path> <model_name_or_path> <task>")
        sys.exit(1)

    model_path = sys.argv[1]
    input_path = sys.argv[2]
    model_name_or_path = sys.argv[3]
    task = sys.argv[4]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Robust model loading
        config = AutoConfig.from_pretrained(model_name_or_path)
        if task == "sequence-classification":
            model = AutoModelForSequenceClassification.from_config(config)
        else:
            model = AutoModelForCausalLM.from_config(config)
        
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        
        dummy_input = torch.load(input_path, map_location="cpu")

        # Robust input handling
        if isinstance(dummy_input, torch.Tensor):
            dummy_input = {"input_ids": dummy_input.to(device)}
        elif isinstance(dummy_input, list):
            dummy_input = {"input_ids": torch.tensor(dummy_input, device=device)}
        else:
            dummy_input = {k: v.to(device) for k, v in dummy_input.items()}

        # GPU Warm-up
        for _ in range(5):
            with torch.no_grad():
                _ = model(**dummy_input)
        torch.cuda.synchronize()

        # Flush cache before the profiled run
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Single profiled pass for NCU
        with torch.no_grad():
            _ = model(**dummy_input)

    except Exception:
        # Log the full traceback for better debugging
        print(f"Profiling target failed for model {model_name_or_path}:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Marker to indicate failure to the parent process
        print("__NCU_ERROR__", file=sys.stdout)
        sys.exit(1)

if __name__ == "__main__":
    main() 