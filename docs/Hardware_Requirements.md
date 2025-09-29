# Hardware Requirements Guide

This guide specifies the hardware requirements for different levels of experimental validation, from basic functional testing to complete cross-vendor reproduction of all paper claims.

## Overview

The ICD framework supports three validation tiers:
- **Tier 1**: Functional validation (no specialized hardware)
- **Tier 2**: Single-vendor GPU validation (accessible cloud hardware)
- **Tier 3**: Complete cross-vendor validation (institutional access)

## Tier 1: Functional Validation

### Purpose
- Verify software installation and basic functionality
- Reproduce algorithmic correctness using mock data
- Validate statistical analysis framework
- **Suitable for**: Code review, continuous integration, initial testing

### Hardware Requirements

**Minimum Specifications**:
```
CPU: Any x86_64 processor (2+ cores)
RAM: 4GB minimum, 8GB recommended
Storage: 10GB free space
GPU: None required
Network: Standard internet connection
```

**Supported Platforms**:
- **Linux**: Ubuntu 20.04+, CentOS 8+, any recent distribution
- **macOS**: 10.15+ (Intel or Apple Silicon)
- **Windows**: Windows 10+ with WSL2

### Software Dependencies

```bash
# Core dependencies (installed automatically)
python >= 3.10
numpy >= 1.24
scipy >= 1.11
jsonschema >= 4.21

# Optional (for extended mock testing)
torch >= 2.1  # CPU version sufficient
```

### Validation Commands

```bash
# Complete Tier 1 validation
python -m pytest tests/unit tests/integration -v
bash scripts/repro_smoke.sh
python scripts/validate_tier1.py

# Expected runtime: 15-30 minutes
# Expected results: All tests pass, mock improvements >15%
```

## Tier 2: Single-Vendor GPU Validation

### Purpose
- Reproduce core experimental claims on real hardware
- Validate GPU profiling and measurement infrastructure
- Generate publication-quality results for single vendor
- **Suitable for**: Most researchers, reproducible science

### NVIDIA GPU Requirements

**Recommended Configurations**:

| Model | VRAM | Suitable Experiments | Cloud Cost (Est.) |
|-------|------|----------------------|-------------------|
| **RTX 3090** | 24GB | BERT-base, Mamba-130M | N/A (personal) |
| **RTX 4090** | 24GB | BERT-base, Mamba-130M | N/A (personal) |
| **A100 40GB** | 40GB | All single-GPU experiments | $1.50-2.10/hr |
| **A100 80GB** | 80GB | Large models (BERT-large) | $2.50-3.50/hr |
| **H100 80GB** | 80GB | Fastest execution | $4.00-6.00/hr |

**Required CUDA Stack**:
```
CUDA: 11.8+ or 12.x
cuDNN: 8.6+
NVIDIA Driver: 520+
Python packages:
  torch >= 2.1 (with CUDA support)
  nvidia-ml-py >= 11.5
```

### Cloud Provider Options

#### RunPod (Recommended)
```bash
# Launch A100 40GB instance
# Browse to runpod.io/console/pods
# Select: NVIDIA A100 40GB, PyTorch template
# Cost: ~$1.50/hour

# Setup script
git clone https://github.com/mrcha033/iterative-co-design.git
cd iterative-co-design
pip install -e .[experiments]
python scripts/verify_gpu_setup.py
```

#### Lambda Labs
```bash
# Reserve GPU instance at lambdalabs.com
# Select: A100 (40GB) instance
# Cost: ~$2.10/hour

# Connect via SSH
ssh ubuntu@<instance-ip>
# Follow standard setup
```

#### Google Colab Pro+ (Limited)
```python
# Colab Pro+ provides A100 access (limited hours)
!git clone https://github.com/mrcha033/iterative-co-design.git
%cd iterative-co-design
!pip install -e .[experiments]

# Note: Limited to smaller experiments due to time constraints
```

### Tier 2 Validation Protocol

```bash
# Complete single-vendor validation (~6 hours on A100)

# 1. Environment verification
python scripts/verify_gpu_setup.py
python scripts/check_cuda_env.py

# 2. Core experiments
python -m icd.cli.main pair -c configs/bert.json \
    --override pipeline.repeats=1000 --out runs/t2_bert

python -m icd.cli.main pair -c configs/mamba.json \
    --override pipeline.repeats=1000 --out runs/t2_mamba

# 3. TVM baseline comparison
python scripts/run_autotvm.py icd.experiments.hf:load_bert_base \
    --tuning-trials 1000 --target cuda --artifacts runs/t2_tvm

# 4. Statistical validation
python scripts/validate_tier2_results.py runs/t2_*

# Expected improvements:
# BERT-base: 18-25% latency reduction
# Mamba-130M: 20-28% latency reduction
# vs TVM: 10-15% advantage
```

### Memory Requirements by Model

| Model | Min VRAM | Recommended VRAM | Batch Size | Notes |
|-------|----------|------------------|------------|--------|
| **BERT-base** | 4GB | 8GB | 32 | Fits on most GPUs |
| **BERT-large** | 12GB | 24GB | 16 | Requires larger GPU |
| **Mamba-130M** | 6GB | 16GB | 32 | Memory efficient |
| **Mamba-2.8B** | 32GB | 48GB | 8 | Requires A100 80GB |
| **ResNet-50** | 4GB | 8GB | 64 | Vision workloads |
| **GCN (ArXiv)** | 8GB | 16GB | 1024 | Graph processing |

## Tier 3: Cross-Vendor Validation

### Purpose
- Complete reproduction of all paper claims
- Multi-vendor hardware validation
- Production deployment scenarios
- **Suitable for**: Institutional validation, comprehensive evaluation

### AMD GPU Requirements

**Target Hardware**:
```
GPU: AMD MI100, MI210, or MI250X
VRAM: 32GB+ recommended
Host: 64GB+ system RAM
```

**ROCm Software Stack**:
```bash
# ROCm 5.6+ installation (Ubuntu)
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update
sudo apt install rocm-dev rocprofiler-dev

# Verify installation
rocm-smi
rocprof --version
```

**Cloud Access**:
- **AWS EC2**: g4ad instances (Radeon Pro V520)
- **Azure**: NV-series with AMD GPUs
- **Academic clouds**: Some universities provide MI100 access

### Intel GPU/CPU Requirements

**Target Hardware**:
```
CPU: Intel Xeon 8380 (40 cores) or equivalent
GPU: Intel Flex 170, Max 1550 (if available)
RAM: 128GB+ for large CPU experiments
```

**Intel Software Stack**:
```bash
# Intel oneAPI Base Toolkit
wget https://registrationcenter.intel.com/en/products/
# Install Intel VTune Profiler component

# Environment setup
source /opt/intel/oneapi/setvars.sh

# Verify installation
vtune --version
intel_gpu_top  # If Intel GPU present
```

**Cloud Access**:
- **Intel Developer Cloud**: Free tier with Flex/Max GPUs
- **AWS**: Some Xeon instances available
- **Google Cloud**: Intel CPU instances

### Multi-GPU Requirements

**For Large Model Experiments** (Mamba-2.8B+):
```
Configuration: 2x A100 80GB or H100 80GB
Connection: NVLink or high-bandwidth interconnect
Memory: 160GB+ total VRAM
Host RAM: 256GB+ system memory
```

**Cloud Options**:
```bash
# AWS p4d.xlarge (A100 40GB x8, can use subset)
# Cost: ~$3.06/hour per GPU

# Google Cloud a2-highgpu-2g (A100 40GB x2)
# Cost: ~$3.67/hour per GPU

# Azure NC24s_v3 (V100 32GB x4, older but functional)
# Cost: ~$3.60/hour per GPU
```

### Complete Tier 3 Hardware Matrix

| Vendor | Model | VRAM/Cores | Use Case | Access Method | Est. Cost |
|--------|-------|------------|----------|---------------|-----------|
| **NVIDIA** | A100 40GB | 40GB | Primary validation | RunPod/Lambda | $1.50/hr |
| **NVIDIA** | A100 80GB | 80GB | Large models | AWS/GCP | $3.00/hr |
| **NVIDIA** | H100 80GB | 80GB | Performance baseline | Limited cloud | $6.00/hr |
| **AMD** | MI100 | 32GB | Cross-vendor GPU | Academic/AWS | $2.00/hr |
| **AMD** | MI210 | 64GB | Large AMD models | Limited access | $4.00/hr |
| **Intel** | Xeon 8380 | 40 cores | CPU validation | AWS/GCP | $1.00/hr |
| **Intel** | Flex 170 | 16GB | Intel GPU | Intel Cloud | Free tier |

## Storage Requirements

### Disk Space by Tier

| Tier | Base Install | Datasets | Correlation Matrices | Results | Total |
|------|-------------|----------|---------------------|---------|-------|
| **Tier 1** | 2GB | 1GB | 0.5GB | 1GB | **5GB** |
| **Tier 2** | 2GB | 10GB | 5GB | 10GB | **30GB** |
| **Tier 3** | 2GB | 50GB | 50GB | 100GB | **200GB** |

### Network Requirements

```
Tier 1: Standard broadband (10+ Mbps)
Tier 2: High-speed internet (100+ Mbps for model downloads)
Tier 3: Enterprise connection (1+ Gbps for large dataset transfers)
```

## Performance Benchmarks

### Expected Runtimes

| Experiment | Hardware | Tier 1 | Tier 2 | Tier 3 |
|------------|----------|--------|--------|--------|
| **Mock validation** | CPU | 15 min | N/A | N/A |
| **BERT-base** | A100 40GB | N/A | 45 min | N/A |
| **Complete campaign** | A100 40GB | N/A | 6 hours | N/A |
| **Cross-vendor study** | Multi-vendor | N/A | N/A | 48 hours |
| **Production deployment** | Production | N/A | N/A | 72 hours |

### Cost Estimates

| Validation Level | Hardware Cost | Time | Total Cost |
|------------------|---------------|------|------------|
| **Tier 1** | $0 (personal hardware) | 0.5 hours | **$0** |
| **Tier 2** | $1.50/hr (A100 cloud) | 6 hours | **$10-15** |
| **Tier 3** | $3.00/hr average | 72 hours | **$200-300** |
| **Full campaign** | Multi-vendor | 200 hours | **$1000-2000** |

## Hardware-Specific Optimizations

### NVIDIA GPU Optimization

```bash
# Optimal configuration for A100
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Enable Tensor Core optimization
python -m icd.cli.main run -c configs/bert.json \
    --override measure.enable_tensorcore=true \
    --override measure.mixed_precision=true
```

### AMD GPU Optimization

```bash
# ROCm-specific optimizations
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0

python -m icd.cli.main run -c configs/bert.json \
    --override measure.rocm_enable=true \
    --override measure.rocm_optimize=true
```

### Intel CPU Optimization

```bash
# Intel-specific CPU optimizations
export OMP_NUM_THREADS=40
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

python -m icd.cli.main run -c configs/bert.json \
    --override pipeline.target_device=cpu \
    --override measure.vtune_enable=true
```

## Troubleshooting Hardware Issues

### GPU Detection Issues

```bash
# NVIDIA GPU troubleshooting
nvidia-smi  # Should show GPU(s)
python -c "import torch; print(torch.cuda.is_available())"

# AMD GPU troubleshooting
rocm-smi  # Should show GPU(s)
python -c "import torch; print(torch.version.hip)"

# Intel GPU troubleshooting
intel_gpu_top  # Should show Intel GPU
lspci | grep Intel
```

### Memory Issues

```bash
# Monitor GPU memory
nvidia-smi -l 1  # NVIDIA
rocm-smi -d  # AMD

# Reduce memory usage
python -m icd.cli.main run -c configs/bert.json \
    --override graph.mock.d=128 \        # Smaller dimensions
    --override pipeline.repeats=100 \     # Fewer samples
    --override solver.refine_steps=100    # Less optimization
```

### Performance Issues

```bash
# Check thermal throttling
nvidia-smi --query-gpu=temperature.gpu --format=csv -l 1

# Verify CPU usage
htop
iostat -x 1

# Check for interference
ps aux | grep python
nvidia-smi pmon  # Process monitoring
```

## Institutional Hardware Access

### Academic Partnerships

**Recommended institutions with multi-vendor GPU access**:
- **NCSA (University of Illinois)**: A100, MI100 clusters
- **TACC (University of Texas)**: Large-scale GPU systems
- **Pittsburgh Supercomputing Center**: Multi-vendor access
- **Oak Ridge National Lab**: Summit system with diverse hardware

### Commercial Cloud Combinations

**Multi-vendor validation strategy**:
```bash
# NVIDIA baseline (RunPod)
# Cost: $20-30 for complete validation

# AMD validation (AWS g4ad)
# Cost: $50-100 for cross-vendor comparison

# Intel validation (Intel Developer Cloud)
# Cost: Free tier available

# Total multi-vendor cost: $100-200
```

This hardware requirements guide enables researchers to plan appropriate validation strategies based on their access to computing resources while ensuring scientific rigor at each tier.