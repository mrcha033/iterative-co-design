# Hardware Profiling Integration Guide

## Overview

This document describes the REAL measurement infrastructure for validating the paper's mechanistic claims. This replaces the mock/stub implementations with actual hardware profiling.

**Status**: ✅ Infrastructure implemented, integration pending

## What Changed

### Before (Stubs)
```python
# icd/measure/l2_ncu.py (OLD)
def collect_l2_section_stub() -> Dict[str, float]:
    return {"l2_tex__t_sector_hit_rate.pct": float("nan")}
```

### After (Real Implementation)
```python
# icd/measure/l2_ncu.py (NEW)
def collect_l2_metrics(model, inputs, ncu_path=None) -> Dict[str, float]:
    # Real Nsight Compute profiling
    # Extracts L2 cache hit rate from GPU
    # Returns actual measured metrics
```

## Key Components

### 1. Real L2 Cache Profiling (`icd/measure/l2_ncu.py`)

**Purpose**: Measure L2 cache hit rates using NVIDIA Nsight Compute

**Functions**:
- `find_ncu_binary()` - Auto-detect NCU installation
- `parse_ncu_json(path)` - Extract metrics from NCU output
- `collect_l2_metrics(model, inputs)` - Run profiling and return metrics

**Usage**:
```python
from icd.measure.l2_ncu import collect_l2_metrics

# Profile a model
cache_metrics = collect_l2_metrics(
    model=my_model,
    inputs=my_inputs,
    output_dir="profiling_output",
)

# Returns: {"l2_hit_rate_pct": 87.4, "l2_throughput_pct": 68.2, ...}
```

**Environment Variables**:
- `ICD_NCU_PATH` - Custom path to ncu binary
- Default locations checked: `/usr/local/cuda/bin/ncu`, `ncu` in PATH

**Requirements**:
- NVIDIA Nsight Compute installed
- CUDA-capable GPU
- PyTorch with CUDA support

### 2. CUDA Latency Measurement (`icd/measure/cuda_latency.py`)

**Purpose**: Replace mock inference with real GPU kernel timing

**Functions**:
- `warmup_model(model, inputs, num_iterations)` - Thermal stabilization
- `measure_cuda_latency(model, inputs, num_repeats)` - Precise timing
- `measure_latency_with_stats(...)` - Full statistical analysis
- `compare_latencies(baseline, treatment)` - Statistical comparison

**Usage**:
```python
from icd.measure.cuda_latency import measure_latency_with_stats

# Measure with statistical rigor (paper methodology)
stats = measure_latency_with_stats(
    model=my_model,
    inputs=my_inputs,
    num_repeats=1000,  # Paper uses 1000
    warmup=50,         # Paper uses 50
    confidence=0.95,   # Paper uses 95% CI
    device="cuda",
)

# Returns: {
#   "mean": 12.3,
#   "std": 0.8,
#   "p50": 12.1, "p95": 13.9, "p99": 15.2,
#   "ci_lower": 11.8, "ci_upper": 12.8,
#   "cv": 0.065,
#   "n_samples": 1000,
# }
```

### 3. Mechanistic Validation Script (`scripts/validate_mechanistic_claim.py`)

**Purpose**: Validate core scientific claim: Modularity → Cache → Latency

**What it does**:
1. Generates permutations with varying modularity scores (Q)
2. Measures L2 cache hit rate for each permutation
3. Measures latency for each permutation
4. Computes correlations:
   - Q ↔ L2 hit rate (expect: positive)
   - L2 ↔ Latency (expect: negative)
   - Q ↔ Latency (expect: negative, r ≈ -0.88 per paper)
5. Generates validation plots
6. Saves results to JSON

**Usage**:
```bash
# Basic validation
python scripts/validate_mechanistic_claim.py \
    --config configs/mamba.json \
    --device cuda \
    --output validation_results.json

# Advanced: more permutations for better correlation
python scripts/validate_mechanistic_claim.py \
    --config configs/mamba.json \
    --device cuda \
    --num-permutations 50 \
    --output validation_results.json
```

**Output Files**:
- `validation_results.json` - Correlation data, measurements
- `validation_results.png` - Three-panel correlation plots

## Integration Steps

### Step 1: Install Requirements

```bash
# 1. CUDA toolkit with Nsight Compute
# Download from: https://developer.nvidia.com/nsight-compute

# 2. PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Additional dependencies
pip install scipy matplotlib networkx

# 4. Verify installation
which ncu
ncu --version
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 2: Enable Real Profiling in Configs

Update your configuration files to use real measurements:

```json
{
  "measure": {
    "ncu_enable": true,
    "ncu_path": null,  // Auto-detect
    "power_enable": true,
    "latency_method": "cuda_events",  // NEW: use CUDA events
    "num_warmup": 50,
    "num_samples": 1000
  },
  "graph": {
    "source": "instrumented",  // NEW: use real co-access measurement
    "instrumented": {
      "temporal_window_ns": 100,
      "min_coaccesses": 2,
      "cache_line_bytes": 64
    }
  }
}
```

### Step 3: Run Validation on Real Hardware

```bash
# Prerequisites check
python scripts/check_hardware_setup.py

# Run the fully integrated pipeline (preferred)
icd validate --out results/full_validation --device cuda

# Or run the mechanistic validation script directly
python scripts/validate_mechanistic_claim.py \
    --config configs/mamba.json \
    --device cuda \
    --num-permutations 20 \
    --output results/validation_$(date +%Y%m%d).json
```

The `icd validate` command wraps the entire pipeline described in this document,
including mechanistic validation, the experimental matrix, table aggregation, and
summary reporting. It accepts the same high-level options as the standalone
scripts (for example `--quick`, `--skip-matrix`, and `--models`).

### Step 4: Verify Results Match Paper Claims

Expected correlation values from paper (Section 3.5, Table 2):
- **Modularity ↔ L2 Hit Rate**: r ≈ +0.85 to +0.92
- **L2 Hit Rate ↔ Latency**: r ≈ -0.88
- **Modularity ↔ Latency**: r ≈ -0.88 to -0.91

Your validation results should be in this range. If not, there are three possibilities:
1. Implementation issue (check logs)
2. Hardware differences (expected variation)
3. Theoretical model needs refinement (publishable finding!)

## NCU Integration Details

### Running NCU Manually (For Testing)

```bash
# Profile a single model inference
ncu --target-processes all \
    --metrics l2_tex__t_sector_hit_rate.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --export ncu_output \
    --force-overwrite \
    python -c "import torch; model = ...; model(inputs)"

# Export to JSON for parsing
ncu --export json --export-file output.json ...

# Parse results
python -c "from icd.measure.l2_ncu import parse_ncu_json; print(parse_ncu_json('output.json'))"
```

### NCU Metrics Reference

Key metrics collected:
- `l2_tex__t_sector_hit_rate.pct` - L2 cache hit rate (primary)
- `lts__t_sector_hit_rate.pct` - L2 unified cache hit rate (alternative)
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` - DRAM bandwidth utilization

Full metric list: `ncu --query-metrics`

## Instrumented Graph Construction

The instrumented path (`icd/core/graph_instrumented.py`) measures REAL co-access patterns instead of assuming spatial locality.

### Differences from Heuristic Path

| Aspect | Heuristic (`graph_pytorch.py`) | Instrumented (`graph_instrumented.py`) |
|--------|--------------------------------|----------------------------------------|
| **Approach** | Assumes dimension i correlates with i+1 | Measures which dimensions are actually co-accessed |
| **Data Source** | FX metadata, op names | Real tensor access traces during execution |
| **Temporal Awareness** | No | Yes - tracks access within temporal window |
| **Cache Awareness** | No | Yes - groups by cache line size |
| **Validation** | Cannot validate | Can correlate with real cache metrics |

### Enable Instrumented Graph

```json
{
  "graph": {
    "source": "instrumented",
    "instrumented": {
      "temporal_window_ns": 100,      // Co-access window (L1 cache latency)
      "min_coaccesses": 2,             // Minimum co-accesses to create edge
      "num_samples": 10,               // Number of forward passes to sample
      "cache_line_bytes": 64,          // Group dimensions by cache line
      "activation_threshold": 0.0,     // Filter inactive dimensions
      "store_access_log": true,        // Persist trace for analysis
      "trace_output_path": "access_trace.jsonl"
    }
  }
}
```

## Troubleshooting

### NCU Not Found
```bash
# Set explicit path
export ICD_NCU_PATH=/usr/local/cuda/bin/ncu

# Or install NCU
# Download from: https://developer.nvidia.com/nsight-compute
```

### CUDA Not Available
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### NCU Profiling Fails
Common issues:
1. **Permission denied**: Run with sudo or adjust permissions
2. **Kernel not found**: Model needs to actually use GPU (check `.to('cuda')`)
3. **Timeout**: Large models need longer profiling time

### Measurements Return NaN
This means profiling failed gracefully. Check:
1. Is NCU installed? (`which ncu`)
2. Is CUDA available? (`python -c "import torch; print(torch.cuda.is_available())"`)
3. Check logs for specific error messages

## Next Steps

### For Immediate Use (Without GPU)
You can still:
- Review the implementation
- Run unit tests: `pytest tests/unit/test_cuda_latency.py`
- Validate code structure
- Plan experiments

### For Real Validation (With GPU)
1. Set up CUDA environment
2. Install Nsight Compute
3. Run validation script
4. Generate real data for paper

### For Paper Submission
Once you have real data:
1. Update Table 1 with measured latencies
2. Update Table 2 (mediation analysis) with real correlations
3. Replace Figure 8 with real correlation plots
4. Update text: "mock data" → "measured on [GPU]"

## File Locations

**New Implementations**:
- `icd/measure/l2_ncu.py` - Real L2 profiling (NEW)
- `icd/measure/cuda_latency.py` - Real latency measurement (NEW)
- `scripts/validate_mechanistic_claim.py` - Validation pipeline (NEW)

**Updated Configurations**:
- Enable in `configs/*.json` by setting:
  - `measure.ncu_enable: true`
  - `measure.latency_method: "cuda_events"`
  - `graph.source: "instrumented"`

**Documentation**:
- This file: `docs/Hardware_Profiling_Integration.md`
- Gap analysis: `docs/Gap_Analysis.md`
- Experimental procedures: `docs/Experimental_Procedures.md`

## Contact & Support

If measurement infrastructure fails or you need help integrating:
1. Check logs for specific error messages
2. Verify hardware setup: `python scripts/check_hardware_setup.py`
3. Review this documentation
4. Open issue with logs and environment details

---

**Status Summary**:
- ✅ L2 profiling infrastructure implemented
- ✅ CUDA latency measurement implemented
- ✅ Validation script created
- ⚠️ Requires GPU hardware to run
- ⚠️ Model-specific permutation application pending
- ❌ Not yet integrated into main pipeline (requires config changes)

**When this is fully integrated**, you will have REAL data to replace all mock measurements in the paper.
