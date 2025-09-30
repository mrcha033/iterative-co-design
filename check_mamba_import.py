#!/usr/bin/env python3
"""Diagnostic script to check mamba-ssm installation."""

import sys
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print()

# Check if packages are installed
packages_to_check = [
    "torch",
    "transformers",
    "mamba_ssm",
    "causal_conv1d",
]

for pkg in packages_to_check:
    try:
        mod = __import__(pkg)
        location = getattr(mod, "__file__", "built-in")
        print(f"✓ {pkg:20s} FOUND at {location}")
    except ImportError as e:
        print(f"✗ {pkg:20s} NOT FOUND: {e}")

print()
print("=" * 60)
print("Testing mamba_ssm.models.mixer_seq_simple import:")
print("=" * 60)

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    print("✓ MambaLMHeadModel successfully imported")
    print(f"  Location: {MambaLMHeadModel.__module__}")
except ImportError as e:
    print(f"✗ Failed to import MambaLMHeadModel")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()