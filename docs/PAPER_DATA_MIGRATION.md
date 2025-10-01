# Paper Data Migration Guide

## Current Status

**File**: `docs/paper.tex`
**Status**: ⚠️ **CONTAINS MOCK DATA**

All numerical results in tables and figures are **placeholder values** based on theoretical predictions. These need to be replaced with real GPU measurements.

## Files

### 1. `docs/paper.tex` - Original Draft (MOCK DATA)
- Contains mock numerical results
- Now has warning comment at top
- Abstract updated with validation status
- **Use this for writing/iteration**
- **DO NOT submit without validation**

### 2. `docs/paper_TEMPLATE.tex` - Validation-Ready Template (NEW)
- Explicit red warning boxes around all placeholder data
- Clear instructions for generating real data
- Validation protocol in appendix
- **Use this to see what needs validation**

## What Needs to Be Replaced

### Tables with Mock Data

1. **Table 1** (Line ~192-195): Main Results
   ```latex
   (1) Dense Baseline & 35.2 ± 0.3 & 18.5 ± 0.2 & ...
   (2) Algorithm-Only & 31.5 ± 0.3 & 16.1 ± 0.2 & ...
   (3) Linear Pipeline & 24.1 ± 0.2 & 13.9 ± 0.1 & ...
   (4) Iterative (Ours) & 19.8 ± 0.2 & 11.9 ± 0.1 & ...
   ```
   **Replace with**: `results/paper_data/tables/table1_main_results.csv`

2. **Table 2** (Line ~226-227): Mechanistic Correlations
   ```latex
   Linear Pipeline & 0.47 ± 0.02 & 71.3% ± 0.8% & 24.1 ± 0.2
   Iterative (Ours) & 0.79 ± 0.03 & 89.5% ± 1.1% & 19.8 ± 0.2
   Correlation (r)  & -0.91       & -0.88         &
   ```
   **Replace with**: `results/paper_data/tables/table2_correlations.json`

3. **TVM Baseline Table** (Line ~349-353)
   ```latex
   Dense Baseline & 35.2 ± 0.3 & ...
   TVM Auto-Schedule & 31.8 ± 0.4 & ...
   Iterative (Ours) & 19.8 ± 0.2 & ...
   ```
   **Replace with**: Real TVM comparison measurements

4. **Production Metrics** (Line ~946-950)
   ```latex
   P50 Latency & 11.6 ± 0.2 & 9.9 ± 0.2 & ...
   P99 Latency & 27.4 ± 0.6 & 22.8 ± 0.5 & ...
   ```
   **Replace with**: Real production deployment metrics

### Figures with Mock/Missing Data

1. **Figure 1** (Line ~48): `figures/mamba_latency_scan_vs_perm.pdf`
   - Currently missing or mock
   - **Generate with**: `validate_mechanistic_claim.py`
   - Shows latency vs permutation quality

2. **Figure 2** (Line ~158): `figures/quantization_results_barchart.png`
   - Currently missing or mock
   - **Generate with**: Quantization experiment script

3. **Figure 3** (Line ~236): `figures/synthetic_validation.png`
   - Currently missing or mock
   - **Generate with**: Synthetic data validation

## How to Generate Real Data

### Step 1: Prerequisites Check (30 seconds)
```bash
python scripts/check_hardware_setup.py
```

**Expected output**:
```
✓ CUDA available: 1 device(s)
✓ Nsight Compute found
✓ All dependencies available
```

### Step 2: Quick Validation (30 minutes on GPU)
```bash
# Validate core mechanistic claim
python scripts/validate_mechanistic_claim.py \
    --config configs/mamba.json \
    --device cuda \
    --output validation_mamba.json
```

**This generates**:
- `validation_mamba.json` - Correlation data (Q ↔ L2 ↔ Latency)
- `validation_mamba.png` - Three-panel correlation plot

**Check if**:
- Modularity ↔ L2 Hit Rate: r > 0.7 (positive) ✅
- L2 ↔ Latency: r < -0.7 (negative) ✅
- Modularity ↔ Latency: r ≈ -0.88 (paper claim) ✅

### Step 3: Full Paper Data (4-8 hours on GPU)
```bash
# Generate ALL tables and figures
python scripts/generate_paper_data.py \
    --output results/paper_data \
    --models mamba bert resnet gcn
```

**This generates**:
```
results/paper_data/
├── mechanistic_validation/
│   ├── mamba_validation.json
│   ├── mamba_validation.png
│   ├── bert_validation.json
│   └── bert_validation.png
├── experimental_matrix/
│   └── [all baseline runs]
├── tables/
│   ├── table1_main_results.csv      ← Replace Table 1
│   └── table2_correlations.json     ← Replace Table 2
├── figures/
│   └── *.png                         ← Replace all figures
└── SUMMARY_REPORT.json               ← Validation summary
```

### Step 4: Review Results
```bash
# Check if paper claims validated
cat results/paper_data/SUMMARY_REPORT.json
```

**Look for**:
```json
{
  "paper_claims_validated": [
    {
      "claim": "Modularity → L2 correlation (r > 0.7)",
      "measured_value": 0.89,
      "validates": true
    },
    {
      "claim": "Modularity → Latency (r ≈ -0.88)",
      "measured_value": -0.87,
      "validates": true
    }
  ]
}
```

## Migration Scenarios

### Scenario A: Results Validate Paper Claims ✅

**If correlations match** (r ≈ -0.88, improvements 15-25%):

1. **Update Tables**:
   ```bash
   # Copy real data to LaTeX
   cp results/paper_data/tables/table1_main_results.csv paper_tables/
   # Manually transfer to paper.tex
   ```

2. **Update Figures**:
   ```bash
   cp results/paper_data/figures/*.png figures/
   ```

3. **Update Text**:
   - Replace "15-25% improvement" with actual measured range
   - Replace "r = -0.88" with actual measured correlation
   - Replace "p<0.001" with actual p-values
   - Remove all `[PLACEHOLDER]` and `[DRAFT STATUS]` warnings

4. **Update Abstract**:
   - Remove validation status warning
   - Add actual hardware specs (e.g., "measured on NVIDIA A100")
   - Include actual measured improvements

5. **Submit with Confidence** 🎉

### Scenario B: Results Partially Validate ⚠️

**If some correlations strong, others weak**:

1. **Report Actual Values**:
   - Don't inflate or cherry-pick
   - Report all measured correlations

2. **Reframe Contribution**:
   - "We identify boundary conditions for when co-design works"
   - "Framework works best for memory-bound architectures"
   - Still publishable as valuable insight

3. **Update Claims**:
   - Soften absolute claims ("consistently" → "often")
   - Add nuance about when it works vs. doesn't

4. **Discussion Section**:
   - Add subsection: "When Does Co-Design Matter?"
   - Analyze which models validated, which didn't
   - Provide insights for practitioners

### Scenario C: Results Don't Validate ❌

**If correlations weak or wrong sign**:

1. **Honest Reporting**:
   - Report measured values truthfully
   - Science advances through failures

2. **Reframe Paper**:
   - **Title remains**: "The Orthogonality Fallacy" (theoretical contribution)
   - **Abstract**: Emphasize theoretical model + negative empirical result
   - **Contribution**: Theory, methodology, infrastructure
   - **Results**: "We find limited empirical support for..."

3. **Still Publishable**:
   - Theoretical model is novel
   - Methodology is rigorous
   - Negative results are valuable
   - Infrastructure enables future work

4. **Add Value**:
   - Section: "Theory vs. Practice: Understanding the Gap"
   - Discuss why theory didn't match empirics
   - Propose refined theoretical model
   - Identify what needs further research

## Migration Checklist

### Before GPU Validation
- [ ] Review `paper.tex` - know what needs replacement
- [ ] Review `paper_TEMPLATE.tex` - see validation warnings
- [ ] Check hardware: `scripts/check_hardware_setup.py`
- [ ] Test quick validation: one model, 20 permutations
- [ ] Estimate time needed (see `docs/VALIDATION_README.md`)

### During GPU Validation
- [ ] Run `generate_paper_data.py` with `--models mamba bert`
- [ ] Monitor progress (check log files)
- [ ] Save intermediate results
- [ ] Review `SUMMARY_REPORT.json` as soon as available

### After GPU Validation
- [ ] Check if claims validated (SUMMARY_REPORT.json)
- [ ] Decide on scenario (A, B, or C above)
- [ ] Update tables with real data
- [ ] Update figures with real plots
- [ ] Update text with measured values
- [ ] Remove all placeholder warnings
- [ ] Update abstract with validation status
- [ ] Add hardware specs to methods section
- [ ] Verify all numbers consistent throughout paper
- [ ] Check references to measurements
- [ ] Review supplementary materials
- [ ] **Proofread entire paper**

### Pre-Submission
- [ ] All `[PLACEHOLDER]` removed
- [ ] All `[DRAFT STATUS]` removed
- [ ] All tables reference real data
- [ ] All figures generated from real measurements
- [ ] Abstract reflects actual validation status
- [ ] Acknowledgments include hardware access
- [ ] Supplementary includes SUMMARY_REPORT.json
- [ ] Code/data availability statement updated
- [ ] **Final check**: Search paper for "mock", "placeholder", "TODO"

## Quick Reference

### Key Scripts
```bash
# Check setup
scripts/check_hardware_setup.py

# Quick validation (30 min)
scripts/validate_mechanistic_claim.py

# Full data generation (4-8 hours)
scripts/generate_paper_data.py
```

### Key Files
```
docs/paper.tex              - Original draft (MOCK DATA, has warnings)
docs/paper_TEMPLATE.tex     - Template with explicit warnings
docs/VALIDATION_README.md   - Complete validation guide
docs/IMPLEMENTATION_SUMMARY.md - What was built
results/paper_data/         - Where real data goes
```

### Critical Correlations to Validate
1. **Modularity ↔ L2 Hit Rate**: Should be **positive** (r > 0.7)
2. **L2 Hit Rate ↔ Latency**: Should be **negative** (r < -0.7)
3. **Modularity ↔ Latency**: Should be **negative** (r ≈ -0.88)

If all three hold → Paper claims validate ✅
If 1-2 hold → Partial validation, reframe ⚠️
If none hold → Negative result, reframe ❌

---

**Last Updated**: Oct 1, 2025 (after implementation)
**Status**: Ready for GPU validation
**Blocker**: GPU hardware access
