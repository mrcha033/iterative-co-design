#!/bin/bash
# Mamba Installation Script for A100 GPU Environment
# Usage: bash scripts/install_mamba.sh

set -e

echo "🐍 Mamba Installation Script for A100 GPU Environment"
echo "======================================================"

# Check CUDA version
echo "📋 Checking CUDA environment..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    echo "CUDA Version: $(nvcc --version 2>/dev/null | grep "release" || echo "nvcc not found")"
else
    echo "⚠️  nvidia-smi not found. Proceeding with CPU-only installation."
fi

echo ""
echo "🔧 Step 1: Installing compatible PyTorch..."
# Install PyTorch compatible with CUDA 12.1
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "🔧 Step 2: Installing Transformers (latest version)..."
pip install transformers>=4.39.0

echo ""
echo "🔧 Step 3: Installing Mamba dependencies..."
# Install dependencies with build isolation disabled to avoid conflicts
pip install causal-conv1d>=1.2.0 --no-build-isolation
pip install mamba-ssm>=1.2.0 --no-build-isolation

echo ""
echo "🔧 Step 4: Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    if torch.cuda.device_count() > 0:
        print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

echo ""
python -c "
try:
    from transformers import MambaForCausalLM
    print('✅ Mamba model support: Available')
except ImportError as e:
    print(f'❌ Mamba model support: Not available - {e}')
"

echo ""
python -c "
try:
    import mamba_ssm
    print('✅ mamba-ssm: Available')
except ImportError as e:
    print(f'❌ mamba-ssm: Not available - {e}')
"

echo ""
python -c "
try:
    import causal_conv1d
    print('✅ causal-conv1d: Available')
except ImportError as e:
    print(f'❌ causal-conv1d: Not available - {e}')
"

echo ""
echo "🎉 Mamba installation complete!"
echo ""
echo "🚀 Test with: python scripts/run_experiment.py model=mamba_3b dataset=wikitext103 method=dense dry_run=true" 