#!/bin/bash
# RunPod Quick Start - Run critical experiments only (14 hours)
# This gives you enough data to defend core claims

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "RunPod Quick Start - Minimal Viable Paper"
echo "========================================"
echo "This will run ~14 hours of experiments to validate core claims"
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

cd "$PROJECT_ROOT"

# Check environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Verify CUDA
echo "Verifying CUDA..."
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Create directories
mkdir -p experiments/{table1_minimal,quantization_minimal,mechanism_minimal,ablations_minimal}

# ========================================
# Phase 1: Table 1 - Mamba Only (4 hours)
# ========================================
echo ""
echo "========================================"
echo "Phase 1: Table 1 - Mamba-2.8B (4h)"
echo "========================================"

RUNS=5
# Use mamba_3b.json for HuggingFace Transformers Mamba-2.8B-hf
# (Original mamba-ssm requires mamba_ssm package which may not be installed)
CONFIG="configs/mamba_3b.json"

for run in $(seq 1 $RUNS); do
    echo "Run $run/$RUNS..."

    # Dense
    python -m icd.cli.main run -c $CONFIG \
        --override transform.sparsity.enable=false \
        --override pipeline.mode=linear \
        --override measure.ncu_enable=true \
        --out experiments/table1_minimal/dense/run_$run

    # Algo-only
    python -m icd.cli.main run -c $CONFIG \
        --override transform.sparsity.enable=true \
        --override pipeline.mode=linear \
        --override solver.skip_solve=true \
        --override measure.ncu_enable=true \
        --out experiments/table1_minimal/algo_only/run_$run

    # Linear
    python -m icd.cli.main run -c $CONFIG \
        --override transform.sparsity.enable=true \
        --override pipeline.mode=linear \
        --override measure.ncu_enable=true \
        --out experiments/table1_minimal/linear/run_$run

    # Iterative
    python -m icd.cli.main run -c $CONFIG \
        --override transform.sparsity.enable=true \
        --override pipeline.mode=iterative \
        --override measure.ncu_enable=true \
        --out experiments/table1_minimal/iterative/run_$run
done

# ========================================
# Phase 2: Quantization - Mamba (2 hours)
# ========================================
echo ""
echo "========================================"
echo "Phase 2: Quantization - Mamba (2h)"
echo "========================================"

QUANT_RUNS=6

for run in $(seq 1 $QUANT_RUNS); do
    echo "Run $run/$QUANT_RUNS..."

    # Quant→Permute
    python -m icd.cli.main run -c $CONFIG \
        --override transform.quant.enable=true \
        --override pipeline.transform_stage=pre \
        --override pipeline.post_transform_repermute=never \
        --override measure.ncu_enable=true \
        --out experiments/quantization_minimal/quant_perm/run_$run

    # Permute→Quant
    python -m icd.cli.main run -c $CONFIG \
        --override transform.quant.enable=true \
        --override pipeline.transform_stage=post \
        --override pipeline.post_transform_repermute=never \
        --override measure.ncu_enable=true \
        --out experiments/quantization_minimal/perm_quant/run_$run

    # Iterative
    python -m icd.cli.main run -c $CONFIG \
        --override transform.quant.enable=true \
        --override pipeline.transform_stage=post \
        --override pipeline.post_transform_repermute=always \
        --override measure.ncu_enable=true \
        --out experiments/quantization_minimal/iterative/run_$run
done

# ========================================
# Phase 3: Mechanistic Analysis (4 hours)
# ========================================
echo ""
echo "========================================"
echo "Phase 3: Mechanistic Analysis (4h)"
echo "========================================"

for run in $(seq 1 5); do
    echo "Run $run/5..."

    # Linear (capture modularity + L2 metrics)
    python -m icd.cli.main run -c $CONFIG \
        --override pipeline.mode=linear \
        --override measure.ncu_enable=true \
        --out experiments/mechanism_minimal/linear/run_$run

    # Iterative (capture modularity + L2 metrics)
    python -m icd.cli.main run -c $CONFIG \
        --override pipeline.mode=iterative \
        --override measure.ncu_enable=true \
        --out experiments/mechanism_minimal/iterative/run_$run
done

# ========================================
# Phase 4: Key Ablations (4 hours)
# ========================================
echo ""
echo "========================================"
echo "Phase 4: Key Ablations (4h)"
echo "========================================"

# Test: Modularity vs TSP objective
echo "Ablation: Modularity vs TSP..."
for run in $(seq 1 3); do
    # TSP-based
    python -m icd.cli.main run -c $CONFIG \
        --override solver.objective=tsp \
        --override pipeline.mode=iterative \
        --out experiments/ablations_minimal/tsp/run_$run

    # Modularity-based (default)
    python -m icd.cli.main run -c $CONFIG \
        --override solver.objective=modularity \
        --override pipeline.mode=iterative \
        --out experiments/ablations_minimal/modularity/run_$run
done

# Test: Iteration count
echo "Ablation: Iteration count..."
for iters in 0 1 2; do
    python -m icd.cli.main run -c $CONFIG \
        --override pipeline.max_iterations=$iters \
        --out experiments/ablations_minimal/iters_$iters/run_1
done

# ========================================
# Analysis
# ========================================
echo ""
echo "========================================"
echo "Running analysis scripts..."
echo "========================================"

python experiments/scripts/aggregate_table1_minimal.py experiments/table1_minimal/
python experiments/scripts/aggregate_quantization.py experiments/quantization_minimal/

# ========================================
# Summary
# ========================================
echo ""
echo "========================================"
echo "✓ Quick Start Experiments Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - experiments/table1_minimal/"
echo "  - experiments/quantization_minimal/"
echo "  - experiments/mechanism_minimal/"
echo "  - experiments/ablations_minimal/"
echo ""
echo "Next steps:"
echo "  1. Download results: rsync -avz root@[runpod-ip]:/workspace/iterative-co-design/experiments/ ./experiments_runpod/"
echo "  2. Generate figures: python scripts/generate_all_figures.py experiments/"
echo "  3. If reviewers request more data, run full batch scripts"
echo ""