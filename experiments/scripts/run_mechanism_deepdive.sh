#!/usr/bin/env bash
# Deep dive into the Modularity → Cache → Latency chain (Table 3).
# Runs linear and iterative pipelines multiple times and collects Nsight metrics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments/mechanism/mamba_deepdive"
LOG_FILE="$PROJECT_ROOT/experiments/mechanism/deepdive.log"
CONFIG_PATH="${1:-$PROJECT_ROOT/configs/mamba_3b.json}"
RUNS=${RUNS:-5}
SEED_BASE=${SEED_BASE:-3401}

mkdir -p "$EXPERIMENTS_DIR"

cd "$PROJECT_ROOT"

# Activate virtual environment if available
if [ -d "venv" ]; then
    # shellcheck source=/dev/null
    source venv/bin/activate || true
fi

# Sanity check CUDA availability so that failures are caught early.
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available. Did you attach a GPU?"
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
PY

run_mode() {
    local mode=$1
    local run=$2
    local seed=$3

    local out_dir="$EXPERIMENTS_DIR/${mode}/run_${run}"
    mkdir -p "$out_dir"

    echo "[$(date)] Running mode=${mode} run=${run} seed=${seed}" | tee -a "$LOG_FILE"

    python -m icd.cli.main run \
        -c "$CONFIG_PATH" \
        --override pipeline.mode="$mode" \
        --override solver.rng_seed=$seed \
        --override measure.ncu_enable=true \
        --override hardware.device=cuda \
        --out "$out_dir" \
        2>&1 | tee "$out_dir/experiment.log"
}

for run in $(seq 1 "$RUNS"); do
    seed=$((SEED_BASE + run))
    run_mode linear "$run" "$seed"
    run_mode iterative "$run" "$seed"
    echo "[$(date)] Completed run $run/$RUNS" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

done

SUMMARY_OUT="$PROJECT_ROOT/experiments/results/table3_mamba_deepdive.csv"
RAW_OUT="$PROJECT_ROOT/experiments/results/table3_mamba_deepdive_raw.csv"
mkdir -p "$(dirname "$SUMMARY_OUT")"

python "$PROJECT_ROOT/scripts/extract_mechanism_metrics.py" \
    "$EXPERIMENTS_DIR" \
    --output "$SUMMARY_OUT" \
    --raw-output "$RAW_OUT"

cat <<EOF | tee -a "$LOG_FILE"
========================================
Mechanism deep dive complete
Summary CSV : $SUMMARY_OUT
Raw metrics : $RAW_OUT
========================================
EOF
