# Reproducibility Checklist

This document provides a comprehensive checklist to ensure reproducible results for the iterative-co-design framework.

## 🔧 Environment Setup

### Option 1: Docker (Recommended)
```bash
# Build the Docker image
make build-docker

# Run the container with GPU support
make run-docker
```

**Expected Docker Image Hash:** `sha256:TBD` (will be updated after first successful build)

### Option 2: Conda Environment
```bash
# Create conda environment
conda env create -f environment.yml
conda activate iterative-co-design
```

### Option 3: Virtual Environment
```bash
# Create virtual environment
make setup-venv
source venv/bin/activate
```

## 🖥️ Hardware Requirements

### Minimum Requirements
- **GPU:** NVIDIA GPU with compute capability ≥ 7.0
- **VRAM:** 16GB+ (for Mamba-3B experiments)
- **RAM:** 32GB+ (for correlation matrix computation)
- **Storage:** 100GB+ (for datasets and results)

### Tested Configurations
- ✅ **NVIDIA A100 (40GB)** - Primary validation platform
- ✅ **NVIDIA H100 (80GB)** - Extended validation
- ✅ **NVIDIA V100 (32GB)** - Compatibility testing
- ⚠️ **NVIDIA RTX 4090 (24GB)** - Community supported

### Driver/CUDA Requirements
- **CUDA Version:** 11.8 or 12.x
- **Driver Version:** ≥ 525.85
- **cuDNN Version:** 8.7.0+

## 📊 Data Dependencies

### Datasets
The framework automatically downloads the following datasets:
- **WikiText-103** (~500MB)
- **ImageNet** (validation set, ~6GB)
- **OGB ogbn-arxiv** (~170MB)

### Pre-computed Correlation Matrices (Optional)
```bash
# Download pre-computed matrices (2TB total)
# Note: This is optional but significantly speeds up experiments
wget https://example.com/correlation_matrices.tar.gz
tar -xzf correlation_matrices.tar.gz -C data/
```

## 🚀 Quick Validation

### 1. Environment Test
```bash
# Test basic functionality
python -c "
import torch
import numpy as np
import sklearn
print('✅ PyTorch version:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ Environment ready!')
"
```

### 2. Framework Test
```bash
# Run a minimal test
python scripts/test_framework.py --quick
```

### 3. Model Loading Test
```bash
# Test model loading
python scripts/test_model_loading.py --model mamba-3b
```

## 📈 Replication Commands

### Table 1: Main Results
```bash
# Replicate Table 1 (Mamba-3B results)
make replicate-table-1

# Expected runtime: ~2-3 hours
# Expected latency improvement: 15-25%
```

### Table 2: Causal Mechanism
```bash
# Replicate Table 2 (Cache hit rates)
make replicate-table-2

# Expected L2 cache hit rate: >88%
# Expected DRAM bandwidth reduction: >18%
```

### Quantization Results
```bash
# Replicate quantization experiments
python scripts/run_experiment.py --strategy iterative_quant --model mamba-3b
```

## 🔍 Expected Results & Tolerances

### Performance Metrics
| Model | Strategy | Expected Latency (ms) | Tolerance |
|-------|----------|---------------------|-----------|
| Mamba-3B | Baseline | 35.2 ± 0.3 | ±5% |
| Mamba-3B | Linear Pipeline | 24.1 ± 0.2 | ±5% |
| Mamba-3B | Iterative Co-Design | 19.8 ± 0.2 | ±5% |

### Hardware Metrics
| Metric | Expected Value | Tolerance |
|--------|---------------|-----------|
| L2 Cache Hit Rate | >88% | ±2% |
| DRAM Bandwidth Reduction | >18% | ±3% |
| Modularity Score | >0.75 | ±0.05 |

## 🐛 Common Issues & Solutions

### Issue 1: CUDA Out of Memory
**Symptoms:** RuntimeError: CUDA out of memory
**Solutions:**
- Reduce `batch_size` in config.yaml
- Reduce `num_samples` for correlation computation
- Use smaller model variant

### Issue 2: Slow Correlation Matrix Computation
**Symptoms:** Taking >30 minutes for correlation computation
**Solutions:**
- Download pre-computed matrices
- Reduce `num_samples` in config
- Use multiple GPUs if available

### Issue 3: Inconsistent Results
**Symptoms:** Results vary between runs
**Solutions:**
- Ensure `seed` is set in config.yaml
- Enable `deterministic` mode
- Check CUDA version compatibility

### Issue 4: Docker Build Failures
**Symptoms:** Docker build fails with dependency errors
**Solutions:**
- Update Docker to latest version
- Check internet connectivity
- Clear Docker cache: `docker system prune -a`

## 📋 Validation Checklist

Before running experiments, ensure:

- [ ] Hardware meets minimum requirements
- [ ] CUDA/Driver versions are compatible
- [ ] All dependencies are installed correctly
- [ ] Framework tests pass
- [ ] Model loading test passes
- [ ] Configuration file is properly set
- [ ] Output directories exist and are writable
- [ ] Sufficient disk space available (100GB+)
- [ ] GPU memory is available (16GB+)

## 📞 Support

If you encounter issues:

1. **Check Known Issues** in this document
2. **Search GitHub Issues** for similar problems
3. **Create New Issue** with:
   - Hardware specifications
   - Software versions
   - Error messages
   - Steps to reproduce

## 🔄 Version Information

- **Framework Version:** 1.0.0
- **Paper Version:** Original submission
- **Last Updated:** 2024-01-15

## 🎯 Success Criteria

Your environment is ready when:
- [ ] All validation tests pass
- [ ] Docker image builds successfully
- [ ] Model loading completes without errors
- [ ] Baseline experiment runs and produces expected latency
- [ ] Results are within specified tolerances

---

**Note:** This checklist is continuously updated. Please check for the latest version before running experiments.