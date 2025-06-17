#!/bin/bash
# This script downloads the required datasets using the Hugging Face datasets library
# with aria2c for faster downloads and proper caching.

# Set cache directory - default to ~/.cache/huggingface/datasets if HF_HOME not set
if [ -z "$HF_HOME" ]; then
    export HF_HOME="$HOME/.cache/huggingface"
fi

# Install aria2 if not present
if ! command -v aria2c &> /dev/null; then
    echo "Installing aria2..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y aria2
    elif command -v yum &> /dev/null; then
        sudo yum install -y aria2
    elif command -v brew &> /dev/null; then
        brew install aria2
    else
        echo "Please install aria2 manually and retry"
        exit 1
    fi
fi

# Configure aria2c
export HF_DATASETS_DOWNLOAD_MANAGER_TYPE="aria2c"
export HF_DATASETS_ARIA2C_OPTS="--max-concurrent-downloads=8 --max-connection-per-server=8 --min-split-size=1M"

echo "Using cache directory: $HF_HOME"
echo "Downloading datasets with aria2c..."

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

echo "All datasets downloaded and cached in $HF_HOME/datasets" 