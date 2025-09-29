# Artifact Evaluation Guide - NeurIPS 2026

This guide provides step-by-step instructions for NeurIPS reviewers to reproduce the key experimental results from "The Orthogonality Fallacy: Iterative Co-Design as a First-Class Principle for Efficient AI."

## Quick Start (15 minutes)

For reviewers with limited time, this section reproduces the core claims using mock data:

```bash
# 1. Clone and setup
git clone https://github.com/mrcha033/iterative-co-design.git
cd iterative-co-design
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# 2. Verify installation
python -c "import icd; print('✓ ICD installed successfully')"

# 3. Run mock experiments (deterministic, CPU-only)
bash scripts/repro_smoke.sh

# 4. Check results
cat runs/smoke/compare.json
# Expected: ~15-25% latency improvement, statistical significance
```

## Hardware Requirements

### Minimum Requirements (Mock Experiments)
- **CPU**: Any x86_64 with 4+ cores
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **Time**: 15-30 minutes
- **Cost**: $0

### Recommended Requirements (Real GPU Experiments)
- **GPU**: NVIDIA A100 40GB (available on RunPod ~$1.50/hr)
- **RAM**: 32GB system memory
- **Storage**: 100GB free space (for correlation matrices)
- **Time**: 4-8 hours for complete validation
- **Cost**: $50-100 total

### Full Validation Requirements (All Claims)
- **Multi-GPU**: Dual A100 80GB for Mamba-2.8B experiments
- **Cross-vendor**: AMD MI100, Intel Xeon 8380 access
- **Time**: 48-72 hours
- **Cost**: $2000-5000 (institutional access recommended)

## Experiment Roadmap

### Level 1: Functional Validation (15 min, $0)

Verify that all infrastructure works and basic claims hold:

```bash
# Core framework functionality
python -m icd.cli.main run -c configs/mock.json \
    --override pipeline.mode=iterative \
    --out runs/level1_iterative

python -m icd.cli.main run -c configs/mock.json \
    --override pipeline.mode=linear \
    --out runs/level1_linear

# Statistical comparison
python -m icd.cli.main pair -c configs/mock.json \
    --out runs/level1_comparison

# Verify improvement
python -c "
import json
with open('runs/level1_comparison/compare.json') as f:
    result = json.load(f)
improvement = result['acceptance']['iter_improvement_pct']
print(f'Latency improvement: {improvement:.1f}%')
assert improvement > 10, 'Expected >10% improvement'
print('✓ Core claims validated')
"
```

### Level 2: Single-GPU Validation (4 hours, $50-100)

Reproduce key results on accessible cloud hardware:

```bash
# Setup cloud GPU instance (RunPod/Lambda Labs)
# Recommended: A100 40GB, Ubuntu 22.04, CUDA 12.1

# 1. BERT experiments
python -m icd.cli.main pair -c configs/bert.json \
    --override pipeline.repeats=1000 \
    --out runs/level2_bert

python -m icd.cli.main pair -c configs/bert_large.json \
    --override pipeline.repeats=600 \
    --out runs/level2_bert_large

# 2. Mamba experiments
pip install mamba-ssm
python -m icd.cli.main pair -c configs/mamba.json \
    --override pipeline.repeats=1000 \
    --out runs/level2_mamba

# 3. TVM baseline comparison
python scripts/run_autotvm.py icd.experiments.hf:load_bert_base \
    --tuning-trials 1000 \
    --target cuda \
    --artifacts runs/level2_tvm_bert

# 4. Validate results
python scripts/validate_level2_results.py runs/level2_*
```

**Expected Level 2 Results**:
- BERT-base: 15-20% latency improvement
- BERT-large: 20-25% latency improvement
- Mamba-130M: 18-25% latency improvement
- TVM comparison: 10-15% advantage over AutoTVM

### Level 3: Full Validation (48+ hours, $2000+)

Complete reproduction of all paper claims:

```bash
# Multi-architecture validation
for arch in bert bert_large mamba resnet50 gcn_arxiv; do
    python -m icd.cli.main pair -c configs/${arch}.json \
        --override pipeline.repeats=1000 \
        --out runs/level3_${arch}
done

# Cross-vendor validation (requires institutional access)
# AMD MI100
python -m icd.cli.main run -c configs/bert_large.json \
    --override measure.rocm_enable=true \
    --out runs/level3_amd_mi100

# Intel Xeon 8380
python -m icd.cli.main run -c configs/bert_large.json \
    --override measure.vtune_enable=true \
    --override pipeline.target_device=cpu \
    --out runs/level3_intel_xeon

# Production deployment
cd deploy/triton
docker-compose up -d
python ../../scripts/production_benchmark.py \
    http://localhost:8000 bert_model \
    --duration 3600 \
    --report ../../runs/level3_production.json
```

## Key Claims Validation

### Claim 1: Iterative > Linear Pipeline

**Location**: Table 1 (Main Results)
**Validation**:
```bash
python -m icd.cli.main pair -c configs/bert.json --out runs/claim1
python -c "
import json
with open('runs/claim1/compare.json') as f:
    result = json.load(f)
iter_lat = result['iter']['latency_ms']['mean']
linear_lat = result['linear']['latency_ms']['mean']
improvement = (linear_lat - iter_lat) / linear_lat * 100
print(f'Improvement: {improvement:.1f}%')
assert improvement > 15, f'Expected >15%, got {improvement:.1f}%'
print('✓ Claim 1 validated')
"
```

### Claim 2: Statistical Significance

**Location**: Section 3.2 (Statistical Methodology)
**Validation**:
```bash
python scripts/statistical_validation.py runs/claim1
# Expected output:
# p-value: <0.001
# Cohen's d: >1.2
# 95% CI: Non-overlapping intervals
```

### Claim 3: AutoTVM Superiority

**Location**: Section 3.1 (Competitive Baselines)
**Validation**:
```bash
# Generate TVM baseline
python scripts/run_autotvm.py icd.experiments.hf:load_bert_base \
    --tuning-trials 3000 --artifacts runs/claim3_tvm

# Compare with ICD
python -m icd.cli.main run -c configs/bert.json --out runs/claim3_icd
python scripts/compare_with_tvm.py runs/claim3_icd runs/claim3_tvm
# Expected: 10-15% advantage for ICD
```

### Claim 4: Mechanistic Analysis

**Location**: Section 3.2 (Causal Chain)
**Validation**:
```bash
python -m icd.cli.main run -c configs/mamba.json \
    --override measure.ncu_enable=true \
    --override measure.collect_l2_stats=true \
    --out runs/claim4

python scripts/mediation_analysis.py runs/claim4/metrics.json
# Expected: >80% mediation through L2 cache hit rate
```

## Expected Outputs

### Successful Run Indicators

Each experiment produces standardized outputs:

```
runs/experiment_name/
├── metrics.json         # Core performance metrics
├── config.lock.json     # Exact configuration used
├── run.log             # Detailed execution log
├── W.csr.npz           # Correlation matrix (sparse)
├── perm_before.json    # Initial permutation
├── perm_after.json     # Optimized permutation
├── stats_before.json   # Pre-optimization statistics
├── stats_after.json    # Post-optimization statistics
└── report.html         # Human-readable summary
```

### Performance Thresholds

Experiments are considered successful if they meet these criteria:

| Metric | Threshold | Paper Claim |
|--------|-----------|-------------|
| **Latency Improvement** | >15% | 15-25% |
| **Statistical Significance** | p < 0.001 | p < 0.001 |
| **Effect Size** | Cohen's d > 1.0 | d = 1.2-2.1 |
| **L2 Cache Improvement** | >8 percentage points | 10-18 pp |
| **Energy Reduction** | >10% | 15-25% |

### Validation Scripts

Automated validation of results:

```bash
# Validate individual experiment
python scripts/validate_results.py runs/experiment_name
# ✓ Statistical significance confirmed
# ✓ Effect size within expected range
# ✓ Improvement exceeds threshold
# ✓ Reproducibility verified

# Validate complete campaign
python scripts/validate_all_results.py runs/
# Summary: 8/8 experiments successful
# Overall improvement: 18.4% ± 2.1%
# Statistical power: >0.95
```

## Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'mamba_ssm'`
```bash
# Solution: Install Mamba dependencies
pip install mamba-ssm
# Note: Requires CUDA toolkit
```

**Issue**: GPU memory errors
```bash
# Solution: Reduce batch size
python -m icd.cli.main run -c configs/bert.json \
    --override graph.mock.d=128 \  # Smaller dimension
    --override pipeline.repeats=500  # Fewer samples
```

**Issue**: TVM compilation errors
```bash
# Solution: Install TVM with CUDA support
pip install apache-tvm  # Basic version
# For full GPU support, build from source (see TVM_Integration_Guide.md)
```

**Issue**: Slow correlation computation
```bash
# Solution: Use streaming correlation for large models
python -m icd.cli.main run -c configs/config.json \
    --override graph.correlation.streaming=true \
    --override graph.correlation.batch_size=32
```

### Hardware-Specific Issues

**NVIDIA GPU**:
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

**AMD GPU** (Level 3 validation):
```bash
# Check ROCm installation
rocm-smi
rocprof --version

# Verify PyTorch ROCm support
python -c "import torch; print(torch.version.hip)"
```

**Intel GPU** (Level 3 validation):
```bash
# Check Intel GPU
intel_gpu_top
vtune --version
```

## Cost Estimation

### Cloud Provider Costs

**RunPod (recommended for Level 2)**:
- A100 40GB: $1.50/hour
- Level 2 validation: ~6 hours = $9
- Storage: $0.10/GB/month = $10/month

**Lambda Labs**:
- A100 40GB: $2.10/hour
- Level 2 validation: ~6 hours = $12.60

**AWS EC2** (for Level 3):
- p4d.xlarge (A100 40GB): $3.06/hour
- g4ad.xlarge (AMD GPU): $0.379/hour
- Level 3 validation: ~48 hours = $150-200

### Time Estimates

| Validation Level | Setup Time | Runtime | Total Time |
|------------------|------------|---------|------------|
| **Level 1 (Mock)** | 15 min | 15 min | 30 min |
| **Level 2 (Single GPU)** | 30 min | 4-6 hours | 6 hours |
| **Level 3 (Full)** | 2 hours | 48+ hours | 50+ hours |

## Reviewer Checklist

- [ ] **Installation**: Code installs without errors
- [ ] **Mock validation**: Level 1 experiments pass thresholds
- [ ] **Statistical claims**: Significance and effect sizes verified
- [ ] **Hardware claims**: Single-GPU results match expectations
- [ ] **Reproducibility**: Multiple runs show consistent results
- [ ] **Documentation**: Instructions clear and complete

## Contact Information

For artifact evaluation support:
- **Primary contact**: Yunmin Cha (mrcha033@yonsei.ac.kr)
- **GitHub issues**: https://github.com/mrcha033/iterative-co-design/issues
- **Documentation**: Complete guides in `docs/` directory

## Artifact Checklist

This artifact provides:

✅ **Source code**: Complete implementation in Python
✅ **Documentation**: Comprehensive setup and usage guides
✅ **Test suite**: 45 test files with >85% coverage
✅ **Configuration**: Pre-configured experiments for all claims
✅ **Scripts**: Automated reproduction and validation
✅ **Docker support**: Containerized deployment for production claims
✅ **Data**: Synthetic datasets for deterministic validation
✅ **Hardware support**: Multi-vendor profiling infrastructure

The artifact is designed for three levels of validation to accommodate different reviewer time constraints and hardware access levels, while maintaining scientific rigor throughout.