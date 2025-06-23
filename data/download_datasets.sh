#!/bin/bash
# This script downloads the required datasets using the Hugging Face datasets library
# with aria2c for faster downloads and proper caching.

# Exit on any error to prevent partial downloads
set -e

# Parse command line arguments
INSTALL_ARIA2=false
HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-aria2)
            INSTALL_ARIA2=true
            shift
            ;;
        -h|--help)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$HELP" = true ]; then
    echo "Dataset Download Script"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --install-aria2    Explicitly attempt to install aria2 (may require sudo)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "SECURITY NOTE:"
    echo "  This script may attempt to install aria2 using system package managers"
    echo "  which could require sudo privileges. Use --install-aria2 flag to"
    echo "  explicitly consent to potential sudo operations."
    echo ""
    echo "SUPPORTED PACKAGE MANAGERS:"
    echo "  - apt-get (Ubuntu/Debian)"
    echo "  - yum (CentOS/RHEL)" 
    echo "  - brew (macOS)"
    echo "  - conda"
    exit 0
fi

# Set cache directory - default to ~/.cache/huggingface/datasets if HF_HOME not set
if [ -z "$HF_HOME" ]; then
    export HF_HOME="$HOME/.cache/huggingface"
fi

# Function to check if user has sudo access
has_sudo() {
    if command -v sudo &> /dev/null && sudo -n true 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to warn about sudo operations
warn_sudo_operation() {
    echo ""
    echo "⚠️  WARNING: SUDO OPERATION REQUIRED ⚠️"
    echo "This script needs to install aria2 using your system package manager."
    echo "This requires administrative privileges (sudo access)."
    echo ""
    echo "SECURITY CONSIDERATIONS:"
    echo "- This will run package manager commands with elevated privileges"
    echo "- Only proceed if you trust this script and understand the risks"
    echo "- Alternative: Install aria2 manually or use conda instead"
    echo ""
    read -p "Do you want to proceed with sudo installation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled. You can:"
        echo "1. Install aria2 manually: sudo apt-get install aria2 (or equivalent)"
        echo "2. Use conda: conda install -c conda-forge aria2"
        echo "3. Run script with --install-aria2 flag to skip this prompt"
        echo ""
        echo "Proceeding without aria2 (slower download)..."
        return 1
    fi
    return 0
}

# Install aria2 if not present and user consents
if ! command -v aria2c &> /dev/null; then
    echo "aria2 not found."
    
    if [ "$INSTALL_ARIA2" = false ]; then
        echo ""
        echo "💡 For faster downloads, consider installing aria2:"
        echo "   - Run with --install-aria2 flag to attempt automatic installation"
        echo "   - Or install manually: sudo apt-get install aria2 (Ubuntu/Debian)"
        echo "   - Or use conda: conda install -c conda-forge aria2"
        echo ""
        echo "Proceeding without aria2 (slower download)..."
    else
        echo "Attempting to install aria2..."
        
        if command -v apt-get &> /dev/null; then
            if has_sudo || warn_sudo_operation; then
                echo "Installing aria2 with apt-get..."
                sudo apt-get update && sudo apt-get install -y aria2
            fi
        elif command -v yum &> /dev/null; then
            if has_sudo || warn_sudo_operation; then
                echo "Installing aria2 with yum..."
                sudo yum install -y aria2
            fi
        elif command -v brew &> /dev/null; then
            echo "Installing aria2 with brew (no sudo required)..."
            brew install aria2
        elif command -v conda &> /dev/null; then
            echo "Installing aria2 with conda (no sudo required)..."
            conda install -c conda-forge aria2 -y
        else
            echo "⚠️  Cannot install aria2 automatically. Supported package managers:"
            echo "   - apt-get (Ubuntu/Debian): sudo apt-get install aria2"
            echo "   - yum (CentOS/RHEL): sudo yum install aria2"
            echo "   - brew (macOS): brew install aria2"
            echo "   - conda: conda install -c conda-forge aria2"
            echo ""
            echo "Proceeding without aria2 (slower download)..."
        fi
    fi
fi

# Configure aria2c if available
if command -v aria2c &> /dev/null; then
    echo "⚡ Using aria2c for faster downloads"
    export HF_DATASETS_DOWNLOAD_MANAGER_TYPE="aria2c"
    export HF_DATASETS_ARIA2C_OPTS="--max-concurrent-downloads=8 --max-connection-per-server=8 --min-split-size=1M"
else
    echo "📥 Using standard HTTP downloads (slower)"
fi

echo "Using cache directory: $HF_HOME"
echo "Downloading datasets..."

echo "Downloading wikitext-103-raw-v1..."
python -c "
from datasets import load_dataset, config
config.HF_DATASETS_CACHE = '$HF_HOME/datasets'
load_dataset('wikitext', 'wikitext-103-raw-v1', cache_dir='$HF_HOME/datasets')
"

echo "Downloading sst2..."
python -c "
from datasets import load_dataset, config
config.HF_DATASETS_CACHE = '$HF_HOME/datasets'
load_dataset('glue', 'sst2', cache_dir='$HF_HOME/datasets')
"

echo "✅ All datasets downloaded and cached in $HF_HOME/datasets"