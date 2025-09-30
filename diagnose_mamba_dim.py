#!/usr/bin/env python3
"""Diagnostic script to understand Mamba dimension inference."""

import sys
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer

def main():
    model_name = "state-spaces/mamba-2.8b-hf"

    print(f"Loading config for {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  expand: {config.expand}")
    print(f"  state_size: {config.state_size}")

    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"\nLoading model...")
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")
    model.eval()

    print(f"\nCreating example inputs...")
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"  input_ids shape: {input_ids.shape}")

    print(f"\nChecking first layer dimensions...")
    first_layer = model.backbone.layers[0]
    print(f"  mixer type: {type(first_layer.mixer).__name__}")

    # Check mixer internals
    if hasattr(first_layer.mixer, 'in_proj'):
        print(f"  in_proj weight shape: {first_layer.mixer.in_proj.weight.shape}")
    if hasattr(first_layer.mixer, 'out_proj'):
        print(f"  out_proj weight shape: {first_layer.mixer.out_proj.weight.shape}")
    if hasattr(first_layer.mixer, 'x_proj'):
        print(f"  x_proj weight shape: {first_layer.mixer.x_proj.weight.shape}")

    print(f"\nRunning forward pass to check intermediate shapes...")
    with torch.no_grad():
        try:
            output = model(input_ids)
            print(f"  output.last_hidden_state shape: {output.last_hidden_state.shape}")
        except Exception as e:
            print(f"  Forward pass failed: {e}")

    print(f"\nChecking what graph builder would infer...")
    from icd.core.graph_pytorch import (
        _infer_feature_dim_from_tensor,
        _maybe_override_feature_dim_from_config,
        _infer_seq_len_from_tensor
    )

    fallback_dim = _infer_feature_dim_from_tensor(input_ids)
    seq_len = _infer_seq_len_from_tensor(input_ids)

    print(f"  fallback_dim (from input tensor last dim): {fallback_dim}")
    print(f"  seq_len (from input tensor -2 dim): {seq_len}")

    D = fallback_dim
    cfg_hs = config.hidden_size
    print(f"  D before override: {D}")
    print(f"  cfg_hs: {cfg_hs}")

    # Simulate the disambiguation logic
    if seq_len > 0 and cfg_hs > 0 and D == seq_len and cfg_hs != D:
        D = cfg_hs
        print(f"  -> D corrected to {D} via seq_len disambiguation")

    D, override_source = _maybe_override_feature_dim_from_config(model, D)
    print(f"  D after config override: {D} (source: {override_source})")

    print(f"\nConclusion:")
    print(f"  Expected: D should be {config.hidden_size} (hidden_size)")
    print(f"  Actual: D = {D}")
    print(f"  Match: {'✓' if D == config.hidden_size else '✗'}")

if __name__ == "__main__":
    main()