#!/bin/bash
# Master script to run all 20 paper experiments

set -e  # Exit on error

# Default mode is 'real'; can be set to 'simulation' for synthetic experiments
MODE="${MODE:-real}"

echo "=================================="
echo "RUNNING ALL PAPER EXPERIMENTS"
echo "=================================="
echo "Mode: $MODE"
echo ""

# Create output directories
mkdir -p results/{baseline,validation,scaling,pareto,hardware,stats,memory,cross_vendor}

echo "=== EXPERIMENT 1: Table 1 (Main Results) ==="
python scripts/run_baseline_experiment.py \
    --model mamba \
    --output results/baseline/mamba.json

python scripts/aggregate_table1.py \
    --input-dir results/baseline \
    --output results/baseline/table1.json

echo "=== EXPERIMENT 2: Table 2 (Mechanistic) ==="
python scripts/validate_mechanistic_claim.py \
    --model mamba \
    --output results/validation/mechanistic.json

echo "=== EXPERIMENT 3: Table 5 (Kernel Fusion) ==="
python scripts/run_kernel_fusion_experiment.py \
    --model mamba \
    --output results/baseline/kernel_fusion.json

echo "=== EXPERIMENT 4: Figure 7 (Correlations) ==="
python scripts/validate_mechanistic_claim.py \
    --model mamba \
    --output results/validation/correlations.json

echo "=== EXPERIMENT 5: Figure quant_results ==="
python scripts/run_quantization_experiment.py \
    --model mamba \
    --output results/baseline/quantization.json

echo "=== EXPERIMENT 6: TSP Ablation ==="
python scripts/run_tsp_baseline.py \
    --model mamba \
    --output results/baseline/tsp_ablation.json

echo "=== EXPERIMENT 7: Figure batch_size_sensitivity ==="
python scripts/run_batch_size_sweep.py \
    --model mamba \
    --batch-sizes 1 2 4 8 16 32 64 128 256 \
    --output results/scaling/batch_size.json

echo "=== EXPERIMENT 8: Appendix C.2 (Hyperparameter) ==="
python scripts/run_hyperparameter_sweep.py \
    --model mamba \
    --output results/baseline/hyperparameter_sweep.json

echo "=== EXPERIMENT 9: Section 3.5 (Mediation) ==="
python scripts/mediation_analysis.py \
    --model mamba \
    --output results/validation/mediation.json

echo "=== EXPERIMENT 10: Figure bandwidth_saturation ==="
python scripts/run_bandwidth_saturation.py \
    --model mamba \
    --output results/scaling/bandwidth_saturation.json

echo "=== EXPERIMENT 11: ResNet/GCN Support ==="
# Already integrated into baseline experiments

echo "=== EXPERIMENT 12: Cross-Architecture Ablation ==="
python scripts/run_cross_arch_ablation.py \
    --models mamba bert resnet gcn \
    --output results/baseline/cross_arch_ablation.json

echo "=== EXPERIMENT 13: Synthetic Validation ==="
python scripts/run_synthetic_validation.py \
    --num-nodes 1024 \
    --num-communities 16 \
    --output results/validation/synthetic_validation.json

echo "=== EXPERIMENT 14: Model Width Scaling ==="
python scripts/run_width_scaling.py \
    --widths 64 128 256 512 1024 1536 2048 2560 \
    --mode $MODE \
    --output results/scaling/width_scaling.json

echo "=== EXPERIMENT 15: Pareto Frontier ==="
python scripts/run_pareto_frontier.py \
    --output results/pareto/pareto_frontier.json

echo "=== EXPERIMENT 16: Hardware Heatmap ==="
python scripts/run_hardware_heatmap.py \
    --gpus V100 A100 H100 \
    --models mamba bert resnet gcn \
    --mode $MODE \
    --output results/hardware/heatmap.json

echo "=== EXPERIMENT 17: Latency Distributions ==="
python scripts/run_latency_distributions.py \
    --models mamba bert \
    --num-runs 50 \
    --mode $MODE \
    --output results/stats/latency_dist.json

echo "=== EXPERIMENT 18: AutoTVM Comparison ==="
python scripts/run_autotvm.py \
    --model mamba \
    --output results/baseline/autotvm.json

echo "=== EXPERIMENT 19: Cross-Vendor Profiling ==="
python scripts/run_cross_vendor.py \
    --vendors nvidia amd intel \
    --models mamba bert resnet \
    --mode $MODE \
    --output results/cross_vendor/results.json

echo "=== EXPERIMENT 20: Memory Hierarchy Metrics ==="
python scripts/run_memory_hierarchy.py \
    --models mamba bert \
    --mode $MODE \
    --output results/memory/hierarchy_metrics.json

echo ""
echo "=================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=================================="
echo ""
echo "Mode: $MODE"
echo "Results saved to results/ directory"
echo ""
echo "Summary:"
echo "  - Baseline experiments: results/baseline/"
echo "  - Validation experiments: results/validation/"
echo "  - Scaling experiments: results/scaling/"
echo "  - Pareto frontier: results/pareto/"
echo "  - Hardware generalization: results/hardware/"
echo "  - Statistical analysis: results/stats/"
echo "  - Memory profiling: results/memory/"
echo "  - Cross-vendor: results/cross_vendor/"
echo ""
if [ "$MODE" = "real" ]; then
    echo "✅ Using REAL hardware profiling (requires CUDA + Nsight Compute)"
    echo "   To use simulation: MODE=simulation bash $0"
else
    echo "ℹ️  Using SIMULATION mode (no GPU required)"
    echo "   To use real hardware: MODE=real bash $0"
fi
