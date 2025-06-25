#!/bin/bash
# Enhanced Mamba Installation Script
# Usage: bash scripts/install_mamba.sh

set -e

echo "🐍 Enhanced Mamba Installation Script"
echo "======================================================"
echo "This script will attempt to install Mamba dependencies, which require a"
echo "specific CUDA and C++ build environment."
echo ""

# --- Prerequisite Checks ---

# 1. Check for NVIDIA CUDA Compiler (nvcc)
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: nvcc (NVIDIA CUDA Compiler) not found."
    echo "Please install the NVIDIA CUDA Toolkit. Make sure nvcc is in your PATH."
    echo "Recommended version: 12.1"
    exit 1
fi

# 2. Check for C++ compiler (g++)
if ! command -v g++ &> /dev/null; then
    echo "❌ ERROR: g++ (C++ Compiler) not found."
    echo "Please install a C++ compiler. On Debian/Ubuntu: sudo apt-get install build-essential"
    exit 1
fi

echo "✅ Prerequisites found (nvcc, g++)"
echo ""
echo "CUDA Version detected by nvcc:"
nvcc --version
echo ""

# --- Installation ---

echo "🔧 Step 1: Installing Mamba and dependencies via pip..."
echo "This step compiles the packages and may take several minutes."

# Install using the 'mamba' optional dependency defined in pyproject.toml
pip install -e .[mamba]

if [ $? -ne 0 ]; then
    echo "❌ ERROR: 'pip install -e .[mamba]' failed."
    echo "This is common. Please check the following:"
    echo "  1. Your CUDA toolkit version is compatible with the PyTorch version."
    echo "  2. You have enough RAM (compilation can be memory-intensive)."
    echo "  3. Check the error messages above for specific compilation errors."
    echo ""
    echo "💡 For a more reliable setup, please consider using the Docker environment"
    echo "   as described in the README.md and DOCKER_GUIDE.md."
    exit 1
fi

# --- Verification ---

echo ""
echo "🔧 Step 2: Verifying installation..."
python -c "
import torch
import importlib

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if not torch.cuda.is_available():
    print('⚠️ WARNING: PyTorch cannot detect CUDA. Mamba will not work.')

mamba_spec = importlib.util.find_spec('mamba_ssm')
causal_conv_spec = importlib.util.find_spec('causal_conv1d')

if mamba_spec:
    print('✅ mamba-ssm: Found')
else:
    print('❌ mamba-ssm: Not Found')

if causal_conv_spec:
    print('✅ causal-conv1d: Found')
else:
    print('❌ causal-conv1d: Not Found')
"

echo ""
echo "🎉 Mamba installation process finished."
echo "If all checks above are green, you are ready to run Mamba experiments."
echo "If not, the Docker-based setup is strongly recommended."
echo ""
echo "🚀 Test with: python scripts/run_experiment.py model=mamba_3b dataset=wikitext103 method=dense dry_run=true"