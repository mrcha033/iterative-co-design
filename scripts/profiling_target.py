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

# Ensure the project's 'src' directory is in the path to allow model classes
# to be unpickled correctly by torch.load.
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def main():
    if len(sys.argv) != 5:
        print("Usage: python profiling_target.py <model_path> <input_path> <dummy_model_name> <dummy_task>")
        sys.exit(1)

    model_path = sys.argv[1]
    input_path = sys.argv[2]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Load the entire model object directly. The custom modules (e.g., HDSLinear)
        # can be found because we added 'src' to the Python path.
        model = torch.load(model_path, map_location=device)
        model.eval()
        
        dummy_input = torch.load(input_path, map_location=device)

        # Run one inference pass, which is what ncu will profile.
        with torch.no_grad():
            _ = model(**dummy_input)

    except Exception as e:
        # Exit with a non-zero code to indicate failure to the calling profiler script.
        print(f"Profiling target failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 