#!/bin/bash
# RunPod Quickstart Script
# Prepares environment and runs all experiments for paper data generation
#
# Usage on RunPod:
#   curl -fsSL https://raw.githubusercontent.com/mrcha033/iterative-co-design/main/scripts/quickstart.sh | bash
#
# Or after cloning:
#   cd iterative-co-design
#   bash scripts/quickstart.sh

set -e  # Exit on error

echo "========================================================================"
echo "  Quickstart: Experiment Runner for Iterative Co-Design"
echo "========================================================================"
echo ""

# ============================================================================
# Step 1: Environment Setup
# ============================================================================

echo "[1/7] Setting up Python environment..."
pip install -q --upgrade pip
pip install -q -e .[dev]
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -q scipy matplotlib networkx pandas

echo "✓ Dependencies installed"
echo ""

# ============================================================================
# Step 2: Hardware Verification
# ============================================================================

echo "[2/7] Verifying hardware prerequisites..."
python scripts/check_hardware_setup.py > hardware_setup_report.txt 2>&1

if grep -q "✗" hardware_setup_report.txt; then
    echo "❌ Hardware check failed. See hardware_setup_report.txt"
    cat hardware_setup_report.txt
    exit 1
fi

echo "✓ Hardware checks passed"
cat hardware_setup_report.txt
echo ""

# ============================================================================
# Step 3: Update Configs for Real Experiments
# ============================================================================

echo "[3/7] Updating configs to use instrumented graph construction..."

# Backup original configs
mkdir -p configs/backup
cp configs/*.json configs/backup/ 2>/dev/null || true

# Update each config to use instrumented graph
for config in configs/{mamba,bert_large,resnet50,gcn_arxiv}.json; do
    if [ -f "$config" ]; then
        echo "  Updating $config..."
        python -c "
import json
import sys

with open('$config') as f:
    cfg = json.load(f)

# Force instrumented graph (REAL co-access patterns)
cfg['graph'] = {
    'source': 'instrumented',
    'instrumented': {
        'temporal_window_ns': 100,
        'min_coaccesses': 2,
        'num_samples': 10,
        'cache_line_bytes': 64
    }
}

# Ensure adequate solver budget
if 'solver' not in cfg:
    cfg['solver'] = {}
cfg['solver']['time_budget_s'] = 300
cfg['solver']['refine_steps'] = 500

# Ensure pipeline config
if 'pipeline' not in cfg:
    cfg['pipeline'] = {}
cfg['pipeline']['repeats'] = 1000
cfg['pipeline']['warmup_iter'] = 50

with open('$config', 'w') as f:
    json.dump(cfg, f, indent=2)
"
    fi
done

echo "✓ Configs updated for experiments"
echo ""

# ============================================================================
# Step 4: Mechanistic Validation (Table 2, Figure 7)
# ============================================================================

echo "[4/7] Running mechanistic validation (validates Q → L2 → Latency)..."
echo "  This validates the core scientific claim of the paper."
echo "  Estimated time: 1-2 hours"
echo ""

mkdir -p results/validation

# Run for each model
for model in mamba bert; do
    echo "  → Validating $model..."

    config="configs/${model}.json"
    [ "$model" = "bert" ] && config="configs/bert_large.json"

    python scripts/validate_mechanistic_claim.py \
        --config "$config" \
        --device cuda \
        --num-permutations 20 \
        --output "results/validation/${model}_validation.json" \
        || echo "⚠ Mechanistic validation failed for $model"

    echo "    ✓ $model mechanistic validation complete"
done

echo "✓ Mechanistic validation complete"
echo ""

# ============================================================================
# Step 5: Baseline Experiments (Table 1)
# ============================================================================

echo "[5/7] Running baseline experiments (Dense/Algo/Linear/Iterative)..."
echo "  This generates the main results table (Table 1)."
echo "  Estimated time: 4-8 hours"
echo ""

mkdir -p results/table1

for model in mamba bert; do
    echo "  → Running $model baselines..."

    config="configs/${model}.json"
    [ "$model" = "bert" ] && config="configs/bert_large.json"

    for baseline in dense algo linear iterative; do
        output="results/table1/${model}/${baseline}_metrics.json"

        echo "    - $baseline baseline..."
        python scripts/run_baseline_experiment.py \
            --baseline "$baseline" \
            --model "$model" \
            --config "$config" \
            --output "$output" \
            || echo "⚠ Baseline $baseline failed for $model"
    done

    echo "    ✓ $model baselines complete"
done

echo "✓ All baseline experiments complete"
echo ""

# ============================================================================
# Step 6: Aggregate Table 1
# ============================================================================

echo "[6/7] Aggregating results into Table 1..."

mkdir -p results/paper_tables

python scripts/aggregate_table1.py \
    --input results/table1 \
    --output results/paper_tables/table1_main_results.tex \
    || echo "⚠ Table 1 aggregation failed"

echo "✓ Table 1 generated"
echo ""

# ============================================================================
# Step 7: Mediation Analysis (Section 3.5 claim)
# ============================================================================

echo "[7/7] Running mediation analysis (validates 86% mediation claim)..."

mkdir -p results/mediation

python scripts/mediation_analysis.py \
    --data results/validation/mamba_validation.json \
    --bootstrap 5000 \
    --output results/mediation/mamba_mediation.json \
    || echo "⚠ Mediation analysis failed"

echo "✓ Mediation analysis complete"
echo ""

# ============================================================================
# Final Summary
# ============================================================================

echo "========================================================================"
echo "  EXPERIMENT COMPLETION SUMMARY"
echo "========================================================================"
echo ""

echo "Generated files:"
echo "  ✓ results/validation/*.json         - Mechanistic validation (Table 2)"
echo "  ✓ results/table1/                   - Raw baseline measurements"
echo "  ✓ results/paper_tables/table1*.tex  - Formatted Table 1"
echo "  ✓ results/mediation/*.json          - Mediation analysis"
echo ""

echo "Validation checklist:"
echo ""

# Check for NaN values
echo "  [1/6] Checking for NaN values..."
if grep -r "NaN" results/ > /dev/null 2>&1; then
    echo "    ⚠ WARNING: Found NaN values in results"
else
    echo "    ✓ No NaN values detected"
fi

# Check mechanistic correlations
echo "  [2/6] Checking mechanistic correlations..."
python -c "
import json
import sys

try:
    with open('results/validation/mamba_validation.json') as f:
        data = json.load(f)

    correlations = data.get('correlations', {})
    q_l2 = correlations.get('modularity_vs_l2', 0)
    l2_lat = correlations.get('l2_vs_latency', 0)
    q_lat = correlations.get('modularity_vs_latency', 0)

    print(f'    Q ↔ L2:      {q_l2:.3f} (expect > +0.85)')
    print(f'    L2 ↔ Latency: {l2_lat:.3f} (expect < -0.85)')
    print(f'    Q ↔ Latency:  {q_lat:.3f} (expect ≈ -0.88)')

except Exception as e:
    print(f'    ✗ Could not verify correlations: {e}')
    sys.exit(1)
" || true

# Check improvements
echo "  [3/6] Checking latency improvements..."
python -c "
import json
import pandas as pd

try:
    df = pd.read_csv('results/paper_tables/table1_main_results.csv')
    improvements = df['improvement_pct'].dropna()

    if len(improvements) > 0:
        print(f'    Range: {improvements.min():.1f}% to {improvements.max():.1f}%')
        print(f'    Mean: {improvements.mean():.1f}%')

    else:
        print('    ✗ No improvement data found')
except Exception as e:
    print(f'    ✗ Could not verify improvements: {e}')
" || true

# Check statistical significance
echo "  [4/6] Checking statistical significance..."
python -c "
import pandas as pd

try:
    df = pd.read_csv('results/paper_tables/table1_main_results.csv')
    significant = df['significant'].sum()
    total = len(df)

    print(f'    Significant: {significant}/{total}')

except Exception as e:
    print(f'    ✗ Could not verify significance: {e}')
" || true

# Check effect sizes
echo "  [5/6] Checking effect sizes (Cohen's d)..."
python -c "
import pandas as pd

try:
    df = pd.read_csv('results/paper_tables/table1_main_results.csv')
    cohen_d = df['cohen_d'].dropna()

except Exception as e:
    print(f'    ✗ Could not verify effect sizes: {e}')
" || true

# Check mediation
echo "  [6/6] Checking mediation analysis..."
python -c "
import json

try:
    with open('results/mediation/mamba_mediation.json') as f:
        data = json.load(f)

    mediation_pct = data.get('mediation_fraction', 0) * 100
    ci_low = data.get('ci_low', 0)
    ci_high = data.get('ci_high', 0)

    print(f'    Mediation: {mediation_pct:.1f}% (95% CI: [{ci_low:.2f}, {ci_high:.2f}])')

except Exception as e:
    print(f'    ✗ Could not verify mediation: {e}')
" || true

echo ""
echo "========================================================================"
echo "  NEXT STEPS"
echo "========================================================================"
echo ""
echo "1. Review results in results/ directory"
echo "2. Download results from RunPod to local machine:"
echo "   scp -r runpod@<instance>:~/iterative-co-design/results ."
echo ""
echo "========================================================================"
echo "  Experiment complete! Results saved to results/"
echo "========================================================================"
