#!/usr/bin/env python3
"""Quick test script for HuggingFace Mamba permutation support."""

import torch
from transformers import MambaForCausalLM


def test_hf_mamba_permutation():
    """Test that HF Mamba permutation doesn't crash."""
    print("=" * 60)
    print("Testing HuggingFace Mamba Permutation Support")
    print("=" * 60)

    # Load small model
    print("\n1. Loading mamba-130m-hf...")
    model = MambaForCausalLM.from_pretrained(
        "state-spaces/mamba-130m-hf",
    )
    model = model.cpu()  # Move to CPU for quick test
    print(f"   ✓ Model loaded: {model.config.hidden_size=}, {model.config.intermediate_size=}")

    # Collect modules
    print("\n2. Collecting Mamba modules...")
    from icd.runtime.runners_hf import _collect_mamba_modules_from_model
    modules = _collect_mamba_modules_from_model(model)
    print(f"   ✓ Found {len(modules)} Mamba modules")

    if modules:
        first = modules[0]
        print(f"   ✓ First module: {first['_module_name']}")
        print(f"   ✓ HF Mamba flag: {first.get('_hf_mamba', False)}")
        print(f"   ✓ Has module ref: {first.get('_module_ref') is not None}")

    # Create random permutation
    print("\n3. Creating random permutation...")
    hidden_size = model.config.hidden_size
    pi = torch.randperm(hidden_size)
    print(f"   ✓ Permutation shape: {pi.shape}")

    # Apply permutation
    print("\n4. Applying permutation to first module...")
    from icd.runtime.apply_pi import apply_pi_to_mamba_hf
    try:
        apply_pi_to_mamba_hf(modules[0], pi)
        print("   ✓ Permutation applied successfully!")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test forward pass
    print("\n5. Testing forward pass...")
    try:
        input_ids = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            output = model(input_ids)
        print(f"   ✓ Forward pass successful! Output shape: {output.logits.shape}")
        print(f"   ✓ No NaN/Inf: {torch.isfinite(output.logits).all()}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_hf_mamba_permutation()
    exit(0 if success else 1)