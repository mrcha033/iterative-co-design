# Cross-Vendor Profiling Guide

This guide explains how to use the cross-vendor profiling infrastructure for AMD ROCm and Intel VTune, enabling the multi-vendor validation experiments referenced in the paper.

## Overview

The ICD framework provides vendor-agnostic profiling interfaces through:
- `icd.measure.rocm_profiler`: AMD GPU profiling via `rocprof`
- `icd.measure.vtune_profiler`: Intel GPU/CPU profiling via `vtune`

Both modules provide consistent APIs that integrate seamlessly with the existing measurement stack.

## AMD ROCm Profiling Setup

### Installation Requirements

```bash
# Ubuntu/Debian
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update && sudo apt install rocm-dev rocprofiler-dev

# Add user to render group
sudo usermod -a -G render $USER

# Verify installation
rocprof --version
rocm-smi
```

### ROCm Environment Setup

```bash
# Add to ~/.bashrc
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Verify GPU detection
rocm-smi -a
# Expected output: List of AMD GPUs with utilization stats
```

### Basic ROCm Profiling

```python
from icd.measure.rocm_profiler import ROCmProfiler, ROCmProfilerConfig
from pathlib import Path

# Configure profiling session
config = ROCmProfilerConfig(
    binary=Path("./my_inference_binary"),
    output_dir=Path("runs/rocm_profile"),
    metrics=["SQ_WAVES", "VALUUtil", "SALUUtil", "MemUnit"],
    kernel_regex=".*gemm.*|.*conv.*"  # Focus on compute kernels
)

# Run profiling
profiler = ROCmProfiler(config)
results = profiler.collect()

# Results structure:
# {
#   "metrics": [
#     {"KernelName": "gemm_kernel", "SQ_WAVES": "45.2", "VALUUtil": "78.3%"},
#     ...
#   ]
# }
```

### Integration with ICD Pipeline

```bash
# Enable ROCm profiling in any experiment
python -m icd.cli.main run -c configs/bert.json \
    --override measure.rocm_enable=true \
    --override measure.rocm_metrics='["SQ_WAVES","VALUUtil"]' \
    --out runs/bert_rocm
```

## Intel VTune Profiling Setup

### Installation Requirements

```bash
# Download Intel oneAPI Base Toolkit
wget https://registrationcenter.intel.com/en/products/postregistration/?sn=YOUR_SERIAL

# Install VTune Profiler
sudo ./l_BaseKit_p_2024.0.0.49564_offline.sh
# Select: Intel VTune Profiler component

# Source environment
source /opt/intel/oneapi/setvars.sh

# Verify installation
vtune --version
```

### GPU Profiling Setup

```bash
# Enable GPU profiling (requires Intel GPU)
sudo modprobe i915
echo 'dev.i915.perf_stream_paranoid=0' | sudo tee /etc/sysctl.d/60-vtune.conf
sudo sysctl --system

# Verify GPU access
vtune -collect gpu-hotspots -- echo "test"
```

### Basic VTune Profiling

```python
from icd.measure.vtune_profiler import VTuneProfiler, VTuneProfilerConfig
from pathlib import Path

# Configure profiling session
config = VTuneProfilerConfig(
    binary=Path("./my_inference_binary"),
    output_dir=Path("runs/vtune_profile"),
    analysis_type="gpu-hotspots",  # or "cpu-hotspots", "memory-access"
    result_name="icd_analysis"
)

# Run profiling
profiler = VTuneProfiler(config)
results = profiler.collect()

# Results structure:
# {
#   "result_dir": "/path/to/vtune_result",
#   "summary": {
#     "elapsed_time": "1.23s",
#     "gpu_utilization": "67.8%",
#     ...
#   }
# }
```

### Integration with ICD Pipeline

```bash
# Enable VTune profiling in any experiment
python -m icd.cli.main run -c configs/bert.json \
    --override measure.vtune_enable=true \
    --override measure.vtune_analysis=gpu-hotspots \
    --out runs/bert_vtune
```

## Cross-Vendor Experiment Workflows

### Multi-Vendor Validation Campaign

```bash
# Script for systematic cross-vendor validation
#!/bin/bash

MODELS=("bert" "mamba" "resnet50" "gcn")
VENDORS=("nvidia" "amd" "intel")

for model in "${MODELS[@]}"; do
    for vendor in "${VENDORS[@]}"; do
        echo "Running ${model} on ${vendor}..."

        case $vendor in
            nvidia)
                python -m icd.cli.main run -c configs/${model}.json \
                    --override measure.ncu_enable=true \
                    --out runs/${model}_${vendor}
                ;;
            amd)
                python -m icd.cli.main run -c configs/${model}.json \
                    --override measure.rocm_enable=true \
                    --out runs/${model}_${vendor}
                ;;
            intel)
                python -m icd.cli.main run -c configs/${model}.json \
                    --override measure.vtune_enable=true \
                    --out runs/${model}_${vendor}
                ;;
        esac
    done
done

# Generate cross-vendor comparison report
python scripts/analyze_cross_vendor_results.py runs/
```

### Vendor-Specific Configuration

#### AMD MI100 Configuration
```json
{
  "measure": {
    "rocm_enable": true,
    "rocm_metrics": ["SQ_WAVES", "VALUUtil", "SALUUtil", "L2CacheHit"],
    "rocm_kernel_regex": ".*mfma.*|.*gemm.*",
    "rocm_trace_hip": true
  },
  "solver": {
    "time_budget_s": 2.0  # ROCm compilation can be slower
  }
}
```

#### Intel Xeon 8380 Configuration
```json
{
  "measure": {
    "vtune_enable": true,
    "vtune_analysis": "memory-access",
    "vtune_collect_stack": true
  },
  "pipeline": {
    "target_device": "cpu",
    "warmup_iter": 100  # CPU needs more warmup
  }
}
```

## Hardware-Specific Optimizations

### AMD GPU Optimization

```python
# AMD-specific configuration for optimal performance
def configure_amd_experiment(model_config):
    return {
        **model_config,
        "measure": {
            "rocm_enable": True,
            "rocm_metrics": [
                "SQ_WAVES",      # Wavefront utilization
                "VALUUtil",      # Vector ALU efficiency
                "SALUUtil",      # Scalar ALU efficiency
                "L2CacheHit",    # Memory hierarchy performance
                "MemUnit",       # Memory bandwidth utilization
                "GPUBusy"        # Overall GPU utilization
            ],
            "rocm_additional_args": [
                "--hsa-trace",   # HSA API tracing
                "--sys-trace"    # System-level events
            ]
        }
    }
```

### Intel CPU/GPU Optimization

```python
# Intel-specific configuration
def configure_intel_experiment(model_config):
    return {
        **model_config,
        "measure": {
            "vtune_enable": True,
            "vtune_analysis": "gpu-hotspots",  # or "cpu-hotspots"
            "vtune_env": {
                "OMP_NUM_THREADS": "64",       # Optimize for Xeon
                "OMP_PROC_BIND": "spread",
                "OMP_PLACES": "threads"
            }
        }
    }
```

## Performance Metrics Mapping

### Vendor-Agnostic Metrics

The framework automatically maps vendor-specific metrics to common performance indicators:

| Common Metric | NVIDIA (NCU) | AMD (rocprof) | Intel (VTune) |
|---------------|--------------|---------------|---------------|
| **GPU Utilization** | `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` | `GPUBusy` | `GPU Utilization` |
| **Memory Bandwidth** | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | `MemUnit` | `Memory Bandwidth` |
| **Cache Hit Rate** | `l2_cache_hit_rate` | `L2CacheHit` | `L2 Hit Rate` |
| **Compute Efficiency** | `tensor_precision_fu_utilization` | `VALUUtil` | `FP Efficiency` |

### Automated Metric Collection

```python
from icd.measure.cross_vendor import CrossVendorProfiler

# Unified interface for all vendors
profiler = CrossVendorProfiler.auto_detect()  # Detects available hardware
results = profiler.collect_unified_metrics(model, inputs)

# Unified output format:
# {
#   "vendor": "amd",
#   "device": "MI100",
#   "metrics": {
#     "gpu_utilization": 78.3,
#     "memory_bandwidth_pct": 65.2,
#     "cache_hit_rate": 89.1,
#     "compute_efficiency": 82.7
#   }
# }
```

## Cloud Provider Integration

### AMD GPU Cloud Access

**Recommended Providers**:
- **AWS EC2 G4ad instances** (AMD Radeon Pro V520)
- **Azure NV-series** (AMD Radeon Instinct MI25)
- **Google Cloud** (coming soon)

```bash
# Launch AMD GPU instance
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \
    --instance-type g4ad.xlarge \
    --key-name my-key \
    --security-groups rocm-sg \
    --user-data file://rocm_setup.sh
```

### Intel GPU Cloud Access

**Recommended Providers**:
- **Intel Developer Cloud** (Intel Flex 170, Max 1550)
- **AWS EC2** (Intel Iris Xe instances, limited availability)

```bash
# Intel Developer Cloud setup
# Register at: https://cloud.intel.com/
ssh devcloud
source /opt/intel/oneapi/setvars.sh
```

## Troubleshooting

### AMD ROCm Issues

**Error**: `rocprof: command not found`
```bash
# Reinstall ROCm profiler
sudo apt install rocprofiler-dev
export PATH=/opt/rocm/bin:$PATH
```

**Error**: `Permission denied accessing GPU`
```bash
# Add user to render group
sudo usermod -a -G render $USER
# Logout and login again
```

**Error**: `No AMD GPU detected`
```bash
# Check GPU detection
rocm-smi
lspci | grep AMD
# Verify driver installation
dmesg | grep amdgpu
```

### Intel VTune Issues

**Error**: `vtune: command not found`
```bash
# Source Intel environment
source /opt/intel/oneapi/setvars.sh
# Add to ~/.bashrc for persistence
```

**Error**: `Cannot collect GPU data`
```bash
# Enable GPU profiling
sudo modprobe i915
echo 'dev.i915.perf_stream_paranoid=0' | sudo tee /etc/sysctl.d/60-vtune.conf
sudo sysctl --system
```

**Error**: `Intel GPU not detected`
```bash
# Check Intel GPU
lspci | grep Intel
intel_gpu_top  # If available
```

## Performance Validation

### Cross-Vendor Reproduction

To reproduce the cross-vendor results from the paper:

```bash
# 1. NVIDIA baseline (A100)
python -m icd.cli.main run -c configs/bert_large.json \
    --override measure.ncu_enable=true \
    --out runs/bert_nvidia_a100

# 2. AMD validation (MI100)
python -m icd.cli.main run -c configs/bert_large.json \
    --override measure.rocm_enable=true \
    --out runs/bert_amd_mi100

# 3. Intel validation (Xeon 8380)
python -m icd.cli.main run -c configs/bert_large.json \
    --override measure.vtune_enable=true \
    --override pipeline.target_device=cpu \
    --out runs/bert_intel_xeon

# 4. Generate comparison
python scripts/compare_cross_vendor.py \
    runs/bert_nvidia_a100/metrics.json \
    runs/bert_amd_mi100/metrics.json \
    runs/bert_intel_xeon/metrics.json
```

### Expected Performance Ranges

Based on hardware characteristics:

| Platform | Relative Performance | Notes |
|----------|---------------------|-------|
| **NVIDIA A100** | 1.00x (baseline) | Optimal for tensor operations |
| **AMD MI100** | 0.85-0.95x | Strong FP64, memory bandwidth |
| **Intel Xeon 8380** | 0.60-0.80x | CPU-based, higher latency |

### Validation Criteria

For cross-vendor experiments to be considered successful:

1. **Functional**: All models run without errors
2. **Performance**: >10% improvement over vendor-specific baselines
3. **Consistency**: Effect sizes remain >0.8 across platforms
4. **Reproducibility**: <5% variance across multiple runs

## Integration Examples

### Paper Reproduction

Reproduce Table X (Cross-vendor validation):

```bash
# Generate all cross-vendor data
./scripts/run_cross_vendor_campaign.sh

# Expected runtime: 24-48 hours across all platforms
# Expected cost: $500-1000 in cloud compute
```

### Custom Cross-Vendor Study

```python
from icd.measure import CrossVendorValidator

validator = CrossVendorValidator([
    "nvidia-a100",
    "amd-mi100",
    "intel-xeon-8380"
])

results = validator.validate_model(
    model_path="my_model.pth",
    config_path="my_config.json",
    metrics=["latency", "memory_bandwidth", "energy"]
)

validator.generate_report(results, "cross_vendor_report.html")
```

This cross-vendor infrastructure enables the multi-platform validation claims in the paper while providing practical tools for researchers to extend the work to new hardware platforms.