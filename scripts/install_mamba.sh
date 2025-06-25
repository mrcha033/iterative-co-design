#!/bin/bash
# Mamba Installation Script
# Usage: bash scripts/install_mamba.sh

set -e

echo "🐍 Mamba Installation Script"
echo "======================================================"

# Install Mamba dependencies using the [mamba] extra
pip install -e .[mamba]

echo "
🎉 Mamba installation complete!"
echo "
🚀 Test with: python scripts/run_experiment.py model=mamba_3b dataset=wikitext103 method=dense dry_run=true"
