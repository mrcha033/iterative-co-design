#!/bin/bash
# This script downloads the required datasets using the Hugging Face datasets library.

echo "Downloading wikitext-103-raw-v1..."
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-raw-v1')"

echo "Downloading sst2..."
python -c "from datasets import load_dataset; load_dataset('glue', 'sst2')"

echo "All datasets downloaded." 