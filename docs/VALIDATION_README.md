# Validation Infrastructure - From Mock to Real Data

## 🎯 Purpose

This infrastructure replaces **ALL mock data** in the paper with **real hardware measurements**. When you run these scripts on GPU hardware, you will generate publication-quality empirical data that validates (or refutes) the paper's theoretical claims.

## 📊 What Gets Validated

### Core Scientific Claim (Section 3.5)
**"High-modularity permutations improve cache locality, which reduces latency"**

The validation pipeline measures:
1. **Modularity scores (Q)** for different permutations
2. **L2 cache hit rates** for each permutation
3. **Latency** for each permutation
4. **Correlations** between these metrics

**Expected Results** (from paper):
- Q ↔ L2 Hit Rate: r ≈ +0.85 to +0.92
- L2 ↔ Latency: r ≈ -0.88
- Q ↔ Latency: r ≈ -0.88 to -0.91

### Main Experimental Results (Table 1)
**"Iterative co-design achieves 15-25% latency reduction over linear baseline"**

Compares four baselines:
1. Dense baseline (no optimization)
2. Algorithm-only (HDS sparsity)
3. Linear pipeline (sparsify → permute)
4. Iterative co-design (sparsify ↔ permute with feedback)

## 🚀 Quick Start

### Prerequisites Check
```bash
# Check if you have everything needed
python scripts/check_hardware_setup.py
```

**Expected output**:
```
✓ CUDA available: 1 device(s)
✓ Nsight Compute found: /usr/local/cuda/bin/ncu
✓ All dependencies available
✓ Sufficient disk space
```

### Single-Model Validation (30 minutes)
```bash
# Validate mechanistic claim on one model
python scripts/validate_mechanistic_claim.py \
    --config configs/mamba.json \
    --device cuda \
    --output validation_mamba.json
```

**This generates**:
- `validation_mamba.json` - Correlation data
- `validation_mamba.png` - Three-panel correlation plot
- Console output showing r-values

### Full Paper Data Generation (4-8 hours)
```bash
# Generate ALL data for the paper
python scripts/generate_paper_data.py \
    --output results/paper_data \
    --models mamba bert
```

**This generates**:
- `mechanistic_validation/` - Correlation data for each model
- `experimental_matrix/` - Linear vs Iterative results
- `tables/` - CSV/JSON tables ready for LaTeX
- `figures/` - PNG/PDF plots ready for paper
- `SUMMARY_REPORT.json` - Comprehensive validation report

## 📁 New Files (What We Built)

### Measurement Infrastructure
```
icd/measure/
├── l2_ncu.py              ✅ NEW - Real L2 cache profiling
└── cuda_latency.py        ✅ NEW - Real CUDA-based latency measurement
```

### Validation Scripts
```
scripts/
├── check_hardware_setup.py           ✅ NEW - Verify prerequisites
├── validate_mechanistic_claim.py     ✅ NEW - Validate Q → L2 → Latency
└── generate_paper_data.py            ✅ NEW - Master data generation script
```

### Documentation
```
docs/
├── Hardware_Profiling_Integration.md ✅ NEW - Complete integration guide
└── VALIDATION_README.md              ✅ NEW - This file
```

## 🔬 What Each Script Does

### 1. `check_hardware_setup.py`
**Purpose**: Verify you have required hardware and software

**Checks**:
- ✓ CUDA availability and GPU info
- ✓ Nsight Compute binary location
- ✓ Python dependencies (scipy, matplotlib, etc.)
- ✓ Disk space

**Usage**:
```bash
python scripts/check_hardware_setup.py
```

### 2. `validate_mechanistic_claim.py`
**Purpose**: Test core scientific claim with real measurements

**Algorithm**:
1. Generate 20 permutations with varying modularity (Q)
2. For each permutation:
   - Apply to model (weight reordering)
   - Measure L2 cache hit rate (via NCU)
   - Measure latency (via CUDA events)
3. Compute correlations between Q, L2, and latency
4. Generate correlation plots
5. Save results to JSON

**Usage**:
```bash
python scripts/validate_mechanistic_claim.py \
    --config configs/mamba.json \
    --device cuda \
    --num-permutations 20 \
    --output validation_results.json
```

**Output**:
```json
{
  "correlations": {
    "modularity_vs_l2": 0.89,        // Should be positive
    "l2_vs_latency": -0.85,          // Should be negative
    "modularity_vs_latency": -0.87   // Paper claims -0.88
  },
  "paper_claim_validated": true,
  "measurements": [ /* raw data */ ]
}
```

### 3. `generate_paper_data.py`
**Purpose**: Generate ALL data for paper in one run

**Pipeline**:
1. **Mechanistic Validation** (Step 1)
   - Runs `validate_mechanistic_claim.py` for each model
   - Generates correlation data

2. **Experimental Matrix** (Step 2)
   - Runs Dense, Algo-Only, Linear, Iterative baselines
   - Measures latency, L2, power for each
   - 5 runs per configuration for statistical rigor

3. **Table & Figure Generation** (Step 3)
   - Aggregates results into publication-ready format
   - Generates LaTeX-compatible tables
   - Creates figures with error bars, CIs

4. **Summary Report** (Step 4)
   - Validates paper claims against measured data
   - Identifies any discrepancies
   - Provides publication recommendations

**Usage**:
```bash
# Full validation (4-8 hours)
python scripts/generate_paper_data.py \
    --output results/paper_data \
    --models mamba bert resnet gcn

# Quick test (30 minutes)
python scripts/generate_paper_data.py \
    --output results/test \
    --models mamba \
    --quick
```

## 📈 Understanding the Results

### Good Result (Validates Paper)
```
CORRELATION RESULTS
================================================================================
Modularity ↔ L2 Hit Rate:   r = +0.89  (expect: positive) ✓
L2 Hit Rate ↔ Latency:      r = -0.86  (expect: negative) ✓
Modularity ↔ Latency:       r = -0.88  (expect: negative) ✓
================================================================================
✓ Paper claims VALIDATED
```

### Problematic Result (Does Not Validate)
```
CORRELATION RESULTS
================================================================================
Modularity ↔ L2 Hit Rate:   r = +0.23  (expect: positive) ✗ WEAK
L2 Hit Rate ↔ Latency:      r = +0.15  (expect: negative) ✗ WRONG SIGN
Modularity ↔ Latency:       r = -0.31  (expect: negative) ✗ WEAK
================================================================================
✗ Paper claims NOT validated
```

**If results don't validate**:
1. Check logs for measurement errors
2. Verify NCU is working: `ncu --version`
3. Try different model/configuration
4. If consistently fails → theoretical model needs revision (still publishable!)

## 🔧 Troubleshooting

### "NCU binary not found"
```bash
# Option 1: Install NCU
# Download from https://developer.nvidia.com/nsight-compute

# Option 2: Set path manually
export ICD_NCU_PATH=/usr/local/cuda/bin/ncu

# Option 3: Skip L2 profiling (latency-only validation)
python scripts/validate_mechanistic_claim.py \
    --skip-l2-profiling \
    --config configs/mamba.json
```

### "CUDA not available"
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with CUDA support
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### "Out of memory"
```python
# Reduce batch size in config
{
  "pipeline": {
    "batch_size": 1,  // Reduce if OOM
    "num_samples": 100  // Reduce from 1000 for testing
  }
}
```

### "Measurements return NaN"
This means profiling failed gracefully. Common causes:
1. Model not on GPU (check `.to('cuda')`)
2. NCU permission issues (try `sudo` or adjust permissions)
3. CUDA kernel timeout (increase timeout in NCU command)

Check detailed logs in validation output directory.

## 📊 Expected Time & Resources

### Hardware Requirements
| Experiment | GPU Memory | Time | Recommended GPU |
|------------|-----------|------|-----------------|
| Single validation | 8 GB | 30 min | RTX 3080 |
| Mamba + BERT | 16 GB | 2 hours | RTX 4090 |
| Full paper data | 40 GB | 4-8 hours | A100/H100 |

### Storage Requirements
- Raw measurements: ~500 MB per model
- Tables and figures: ~50 MB
- Full paper data: ~2-3 GB

## 🎓 Scientific Interpretation

### If Results Match Paper
✅ **Strong validation** - Empirical data supports theoretical claims
- Update paper: "mock data" → "measured on [GPU]"
- Add hardware specs to methods section
- Submit with confidence

### If Results Partially Match
⚠️ **Interesting finding** - Some correlations strong, others weak
- Investigate: which models validate? Which don't?
- Potential insight: framework works best for memory-bound architectures
- Paper contribution: identifying boundary conditions
- Still publishable with honest discussion

### If Results Don't Match
🔬 **Research opportunity** - Theory vs. practice gap
- THIS IS STILL PUBLISHABLE
- Reframe as:
  - Theoretical contribution (framework design)
  - Negative result (important for the field)
  - Future work: understanding the gap
- Science advances through failures too!

## 📝 Next Steps After Data Generation

1. **Review Results**
   ```bash
   # Check summary
   cat results/paper_data/SUMMARY_REPORT.json

   # Verify correlations
   cat results/paper_data/mechanistic_validation/*/validation.json
   ```

2. **Update Paper**
   - Replace Table 1 with `tables/table1_main_results.csv`
   - Replace Table 2 with `tables/table2_correlations.json`
   - Replace Figure 7 with `figures/*_validation.png`
   - Update text: cite specific measured values

3. **Prepare for Submission**
   - Include `SUMMARY_REPORT.json` in supplementary materials
   - Add hardware specs to methods (from `check_hardware_setup.py` output)
   - Describe any discrepancies honestly in discussion

## 🤝 Support

If you encounter issues:
1. Run `python scripts/check_hardware_setup.py` and share output
2. Check logs in validation output directories
3. Review `docs/Hardware_Profiling_Integration.md`
4. Open issue with:
   - Hardware specs
   - Error messages
   - Validation output logs

## 📚 Related Documentation

- **Integration Guide**: `docs/Hardware_Profiling_Integration.md`
- **Gap Analysis**: `docs/Gap_Analysis.md` (what was missing)
- **Experimental Procedures**: `docs/Experimental_Procedures.md`
- **Main README**: `README.md`

---

**Status**: ✅ Infrastructure Complete, Ready for GPU Validation

**Last Updated**: 2025-10-01 (Implementation phase)
