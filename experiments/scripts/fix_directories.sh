#!/bin/bash
# Quick fix to apply on RunPod if you're already running the old version
# This creates all necessary directories upfront

EXPERIMENTS_DIR="${1:-/workspace/iterative-co-design/experiments}"

echo "Creating experiment directory structure..."

# Table 1 directories
for arch in mamba bert resnet50 gcn_arxiv; do
    for baseline in dense algo_only linear iterative; do
        for run in {1..5}; do
            mkdir -p "$EXPERIMENTS_DIR/table1/$arch/$baseline/run_$run"
        done
    done
done

# Quantization directories
for strategy in quant_perm perm_quant iterative; do
    for run in {1..6}; do
        mkdir -p "$EXPERIMENTS_DIR/quantization/$strategy/run_$run"
    done
done

# Other directories
mkdir -p "$EXPERIMENTS_DIR/"{mechanism,ablations,generalization,figures,results}
mkdir -p "$EXPERIMENTS_DIR/mechanism/"{mamba_deepdive,synthetic,memory_hierarchy}

echo "✓ Directory structure created"
ls -la "$EXPERIMENTS_DIR"