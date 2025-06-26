#!/usr/bin/env python3
"""Environment verification script for Docker containers."""

import torch

def main():
    print("🔍 Environment Verification")
    print("===========================")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    # Test mamba-ssm
    try:
        from mamba_ssm import MambaLMHeadModel # noqa: F401
        print("✅ mamba-ssm: Import successful")
    except ImportError as e:
        print(f"❌ mamba-ssm: Failed - {e}")
    
    # Test Transformers Mamba
    try:
        from transformers import MambaForCausalLM # noqa: F401
        print("✅ Transformers Mamba: Available")
    except ImportError as e:
        print(f"❌ Transformers Mamba: Failed - {e}")
    
    print()
    print("🚀 Ready to run experiments!")
    print("  BERT: python scripts/run_experiment.py model=bert_base dataset=sst2 method=dense")
    print("  Mamba: python scripts/run_experiment.py model=mamba_370m dataset=wikitext103 method=dense")

if __name__ == "__main__":
    main()
