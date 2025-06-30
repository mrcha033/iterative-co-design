#!/usr/bin/env python3
"""
Dedicated Profiling Target Script for NCU.

This script is designed to be called by `ncu` (NVIDIA's Nsight Compute)
from the LatencyProfiler. It loads a model state dictionary and a dummy input
tensor, then runs a single forward pass for hardware-level profiling.

This approach avoids the complexities and inaccuracies of generating temporary
scripts or profiling simplified/mock models.

Usage:
    ncu --metrics <metrics> --csv python scripts/profiling_target.py <model_path> <input_path> <model_name_or_path> --task <task>
"""

import torch
import sys
from pathlib import Path
import traceback
import argparse
from transformers import AutoConfig, AutoModel

# Ensure the project's 'src' directory is in the path.
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def load_config_safely(name_or_path: str, model_path: str):
    """Try loading config from local model directory first, then from Hub."""
    local_dir = Path(model_path).parent
    try:
        # Prioritize local config to support offline use and custom models
        return AutoConfig.from_pretrained(local_dir, local_files_only=True, trust_remote_code=True)
    except Exception:
        # Fallback to Hub if local config is not available
        return AutoConfig.from_pretrained(name_or_path, local_files_only=False, trust_remote_code=True)

def main():
    parser = argparse.ArgumentParser(description="NCU Profiling Target")
    parser.add_argument("model_path", type=str, help="Path to the model's state_dict.pt file")
    parser.add_argument("input_path", type=str, help="Path to the dummy input .pt file")
    parser.add_argument("name_or_path", type=str, help="Hugging Face model name or path for config loading")
    parser.add_argument("--task", type=str, default="text-generation", help="Model task type")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("__NCU_SKIPPED_CPU__", file=sys.stdout)
        sys.exit(0)

    try:
        config = load_config_safely(args.name_or_path, args.model_path)
        # Use AutoModel for more robust architecture loading
        model = AutoModel.from_config(config, trust_remote_code=True)
        
        state_dict = torch.load(args.model_path, map_location="cpu")
        # Use strict=False to gracefully handle custom layers (e.g., LoRA, sparsity)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"[profiling_target.py] Warning: Unexpected keys in state_dict: {unexpected}", file=sys.stderr)
        if missing:
            print(f"[profiling_target.py] Warning: Missing keys in state_dict: {missing}", file=sys.stderr)

        model.to(device).eval()
        
        dummy_input = torch.load(args.input_path, map_location="cpu")

        # Automatically add attention_mask if it's missing for BERT-like models
        if "bert" in config.model_type and "attention_mask" not in dummy_input:
            dummy_input["attention_mask"] = (dummy_input["input_ids"] != config.pad_token_id).long()
        
        dummy_input = {k: v.to(device) for k, v in dummy_input.items()}

        # GPU Warm-up
        for _ in range(5):
            with torch.no_grad():
                _ = model(**dummy_input)
        torch.cuda.synchronize()

        # Single profiled pass for NCU
        with torch.no_grad():
            _ = model(**dummy_input)

    except Exception:
        print(f"Profiling target failed for model {args.name_or_path}:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("__NCU_ERROR__", file=sys.stdout)
        sys.exit(1)

if __name__ == "__main__":
    main() 