#!/bin/bash
# Batch script for Quantization Experiment (Figure 2)
# 3 strategies × 6 runs = 18 experiments
# Estimated time: 4 hours on A100

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
LOG_FILE="$EXPERIMENTS_DIR/quantization/progress.log"

# Create log directory
mkdir -p "$EXPERIMENTS_DIR/quantization"

echo "Starting Quantization experiments at $(date)" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT"
source venv/bin/activate || source "$PROJECT_ROOT/venv/bin/activate"

RUNS=6
# Use mamba_3b.json for HuggingFace Transformers Mamba-2.8B-hf
CONFIG="configs/mamba_3b.json"

run_quant_experiment() {
    local strategy=$1
    local run=$2
    local transform_stage=$3
    local repermute=$4

    local out_dir="$EXPERIMENTS_DIR/quantization/$strategy/run_$run"

    # Create output directory first
    mkdir -p "$out_dir"

    echo "[$(date)] Running $strategy | run $run..." | tee -a "$LOG_FILE"

    python -m icd.cli.main run \
        -c "$CONFIG" \
        --override transform.quant.enable=true \
        --override transform.quant.dtype=int8 \
        --override pipeline.transform_stage=$transform_stage \
        --override pipeline.post_transform_repermute=$repermute \
        --override measure.ncu_enable=true \
        --override hardware.device=cuda \
        --out "$out_dir" \
        2>&1 | tee -a "$out_dir/experiment.log"

    if [ $? -eq 0 ]; then
        echo "[$(date)] ✓ Completed $strategy | run $run" | tee -a "$LOG_FILE"
    else
        echo "[$(date)] ✗ FAILED $strategy | run $run" | tee -a "$LOG_FILE"
    fi
}

for run in $(seq 1 $RUNS); do
    # Strategy 1: Quant-then-Permute
    run_quant_experiment "quant_perm" "$run" "pre" "never"

    # Strategy 2: Permute-then-Quant
    run_quant_experiment "perm_quant" "$run" "post" "never"

    # Strategy 3: Iterative (Permute-Quant-RePermute)
    run_quant_experiment "iterative" "$run" "post" "always"
done

echo "Quantization experiments completed at $(date)" | tee -a "$LOG_FILE"

# Aggregate results
python "$SCRIPT_DIR/aggregate_quantization.py" "$EXPERIMENTS_DIR/quantization" \
    --output "$EXPERIMENTS_DIR/quantization/summary.csv" \
    | tee -a "$LOG_FILE"

echo "Results saved to: $EXPERIMENTS_DIR/quantization/summary.csv"