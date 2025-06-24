#!/usr/bin/env python3
"""
Debug script to inspect Mamba model layer names.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def inspect_mamba_model():
    """Inspect the layer structure of Mamba model."""
    print("Loading Mamba model...")
    
    try:
        # Load model and tokenizer
        model_name = "state-spaces/mamba-2.8b-hf"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Model loaded: {model_name}")
        print(f"Model type: {type(model)}")
        print(f"Model config: {type(model.config)}")
        print()
        
        print("=== All layers with weights/biases ===")
        layer_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'weight') or hasattr(module, 'bias'):
                print(f"{layer_count:2d}. {name:50s} -> {type(module).__name__}")
                layer_count += 1
        
        print(f"\nTotal layers with parameters: {layer_count}")
        
        print("\n=== Potential target layers for IASP ===")
        potential_targets = []
        for name, module in model.named_modules():
            if any(keyword in name.lower() for keyword in ['proj', 'linear', 'dense', 'fc']):
                if hasattr(module, 'weight'):
                    potential_targets.append(name)
                    print(f"  - {name} ({type(module).__name__})")
        
        print(f"\nFound {len(potential_targets)} potential target layers")
        
        if potential_targets:
            print(f"\nRecommended target layer: {potential_targets[0]}")
        
        return potential_targets
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

if __name__ == "__main__":
    inspect_mamba_model() 