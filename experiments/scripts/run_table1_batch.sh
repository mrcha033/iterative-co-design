#!/bin/bash
# Batch script for Table 1: Main Results (4 architectures × 4 baselines × 5 runs)
# Estimated time: 20 hours on A100

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
LOG_FILE="$EXPERIMENTS_DIR/table1/progress.log"

echo "Starting Table 1 experiments at $(date)" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT"
source venv/bin/activate || source "$PROJECT_ROOT/venv/bin/activate"

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'Using {torch.cuda.get_device_name(0)}')" | tee -a "$LOG_FILE"

# Architecture configs
ARCHS=("mamba" "bert" "resnet50" "gcn_arxiv")
RUNS=5

# Function to run a single experiment
run_experiment() {
    local arch=$1
    local baseline=$2
    local run=$3
    local extra_args=$4

    local out_dir="$EXPERIMENTS_DIR/table1/$arch/$baseline/run_$run"

    echo "[$(date)] Running $arch | $baseline | run $run..." | tee -a "$LOG_FILE"

    python -m icd.cli.main run \
        -c "configs/${arch}.json" \
        $extra_args \
        --override measure.ncu_enable=true \
        --override hardware.device=cuda \
        --out "$out_dir" \
        2>&1 | tee -a "$out_dir/experiment.log"

    if [ $? -eq 0 ]; then
        echo "[$(date)] ✓ Completed $arch | $baseline | run $run" | tee -a "$LOG_FILE"
    else
        echo "[$(date)] ✗ FAILED $arch | $baseline | run $run" | tee -a "$LOG_FILE"
    fi
}

# Main experiment loop
for arch in "${ARCHS[@]}"; do
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Starting architecture: $arch" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    for run in $(seq 1 $RUNS); do
        # (1) Dense Baseline
        run_experiment "$arch" "dense" "$run" "\
            --override transform.sparsity.enable=false \
            --override pipeline.mode=linear"

        # (2) Algorithm-Only (HDS sparsity, no permutation)
        run_experiment "$arch" "algo_only" "$run" "\
            --override transform.sparsity.enable=true \
            --override transform.sparsity.rate=0.5 \
            --override pipeline.mode=linear \
            --override solver.skip_solve=true"

        # (3) Linear Pipeline (Sparsify-then-Permute)
        run_experiment "$arch" "linear" "$run" "\
            --override transform.sparsity.enable=true \
            --override transform.sparsity.rate=0.5 \
            --override pipeline.mode=linear"

        # (4) Iterative Co-Design
        run_experiment "$arch" "iterative" "$run" "\
            --override transform.sparsity.enable=true \
            --override transform.sparsity.rate=0.5 \
            --override pipeline.mode=iterative"
    done
done

echo "========================================" | tee -a "$LOG_FILE"
echo "Table 1 experiments completed at $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Aggregate results
python "$SCRIPT_DIR/aggregate_table1.py" "$EXPERIMENTS_DIR/table1" \
    --output "$EXPERIMENTS_DIR/table1/results_summary.csv" \
    | tee -a "$LOG_FILE"

echo "Results saved to: $EXPERIMENTS_DIR/table1/results_summary.csv"