# 🐳 Complete Docker Guide for Iterative Co-Design

## 📋 Table of Contents
1. [Quick Start](#-quick-start)
2. [System Requirements](#-system-requirements)
3. [Usage Options](#-usage-options)
4. [Custom Experiments](#-custom-experiments)
5. [Configuration Changes](#-configuration-changes)
6. [Deployment Guide](#-deployment-guide)
7. [Troubleshooting](#-troubleshooting)
8. [Performance Tips](#-performance-tips)

## 🚀 Quick Start

### Step 1: Clone Repository
```bash
git clone https://github.com/mrcha033/iterative-co-design.git
cd iterative-co-design
```

### Step 2: Choose Your Approach

**Option A: Use Pre-built Image (Recommended - 5 minutes)**
```bash
# Pull from Docker Hub
docker pull mrcha033/iterative-co-design:latest

# Verify environment
docker-compose -f docker-compose.hub.yml run base
```

**Option B: Build Locally (20+ minutes)**
```bash
# Build from source
docker-compose build base

# Verify environment
docker-compose run base
```

**Option C: CPU-Only Mode**
```bash
# For systems without GPU
docker-compose -f docker-compose.cpu.yml run base
```

### Step 3: Run Experiments

**BERT Experiments (Stable)**
```bash
# With pre-built image
docker-compose -f docker-compose.hub.yml run trainer

# With local build
docker-compose run trainer
```

**Mamba Experiments (Advanced, GPU required)**
```bash
# With pre-built image
docker-compose -f docker-compose.hub.yml run mamba-trainer

# With local build
docker-compose run mamba-trainer
```

## 🔧 System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, Windows with WSL2
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 30GB free space
- **Docker**: Docker Desktop or Docker Engine
- **GPU**: NVIDIA GPU with CUDA drivers ≥ 520.61.05 (for GPU usage)

### NVIDIA Docker Setup
```bash
# Install NVIDIA Docker (Linux)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

## 📊 Usage Options

### Available Docker Compose Files

| File | Purpose | Use Case |
|------|---------|----------|
| `docker-compose.yml` | Local build | Development, customization |
| `docker-compose.hub.yml` | Pre-built image | Quick experiments, production |
| `docker-compose.cpu.yml` | CPU-only | No GPU systems, testing |

### Key Commands

#### With Pre-built Image (Recommended)
```bash
# Environment verification
docker-compose -f docker-compose.hub.yml run base

# Individual experiments
docker-compose -f docker-compose.hub.yml run trainer          # BERT single experiment
docker-compose -f docker-compose.hub.yml run mamba-trainer    # Mamba single experiment

# Complete experiment suites (NEW!)
docker-compose -f docker-compose.hub.yml run all-train        # All Mamba methods
docker-compose -f docker-compose.hub.yml run bert-all-train   # All BERT methods

# Interactive shell
docker-compose -f docker-compose.hub.yml run shell

# Enhanced Jupyter Lab with visualization tools (NEW!)
docker-compose -f docker-compose.hub.yml up jupyter
# Access at http://localhost:8888

# TensorBoard for experiment monitoring (NEW!)
docker-compose -f docker-compose.hub.yml up tensorboard
# Access at http://localhost:6007
```

#### With Local Build
```bash
# Environment verification
docker-compose run base

# Individual experiments
docker-compose run trainer          # BERT single experiment
docker-compose run mamba-trainer    # Mamba single experiment

# Complete experiment suites (NEW!)
docker-compose run all-train        # All Mamba methods
docker-compose run bert-all-train   # All BERT methods

# Interactive shell
docker-compose run shell

# Enhanced Jupyter Lab with visualization tools (NEW!)
docker-compose up jupyter
# Access at http://localhost:8888

# TensorBoard for experiment monitoring (NEW!)
docker-compose up tensorboard
# Access at http://localhost:6007
```

#### CPU-Only Mode
```bash
# All operations use CPU
docker-compose -f docker-compose.cpu.yml run trainer
docker-compose -f docker-compose.cpu.yml run shell
```

## 🛠️ Custom Experiments

### Using Run Script
```bash
# Interactive shell first
docker-compose -f docker-compose.hub.yml run shell

# Inside container
python scripts/run_experiment.py model=bert_base dataset=sst2 method=iterative
python scripts/run_experiment.py model=mamba_3b dataset=wikitext103 method=dense
```

### Available Configurations

**Models:**
- `bert_base`: BERT Base model (stable)
- `mamba_3b`: Mamba 3B model (requires GPU)

**Datasets:**
- `sst2`: Stanford Sentiment Treebank
- `wikitext103`: WikiText-103

**Methods:**
- `iterative`: Iterative co-design approach
- `dense`: Dense training baseline

### Results Location
- **Host**: `outputs/` and `results/` directories
- **Container**: `/workspace/outputs/` and `/workspace/results/`

## 🔄 Configuration Changes

### What's New in the Docker Setup

#### Base Image & CUDA
- **Image**: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- **PyTorch**: 2.2.2+cu118
- **Reason**: Proven stability with Mamba pre-built wheels

#### Mamba Installation Strategy
- **Method**: Pre-built wheel from GitHub releases
- **Wheel**: `mamba_ssm-2.2.4+cu11torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
- **Benefits**: No compilation errors, consistent builds

#### Key Dependencies
```dockerfile
ARG CUDA_VERSION=11.8.0
ARG TORCH_VERSION=2.2.2
ARG CU_TAG=cu118
ARG MAMBA_WHL=mamba_ssm-2.2.4+cu11torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

#### Installation Order
1. System packages (build-essential, python3, git, cmake)
2. Python environment (pip, setuptools<70, wheel)
3. PyTorch stack (torch, torchvision, torchaudio)
4. Core dependencies (numpy, scikit-learn, pandas)
5. Mamba components (pre-built wheel + causal-conv1d)
6. Transformers (4.42.4 with Mamba support)
7. Project installation (editable mode)

## 🚀 Deployment Guide

### GitHub Container Registry (Recommended)

#### Prerequisites
```bash
# Create Personal Access Token at:
# https://github.com/settings/tokens
# Permissions needed: write:packages

# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

#### Build and Deploy
```bash
# Build
docker-compose build base

# Tag for GitHub
docker tag iterative-co-design:latest ghcr.io/yunmin-cha/iterative-co-design:latest
docker tag iterative-co-design:latest ghcr.io/yunmin-cha/iterative-co-design:v1.0-mamba

# Push
docker push ghcr.io/yunmin-cha/iterative-co-design:latest
docker push ghcr.io/yunmin-cha/iterative-co-design:v1.0-mamba
```

### Docker Hub Alternative

#### Prerequisites
```bash
# Login to Docker Hub
docker login
```

#### Deploy
```bash
# Tag for Docker Hub
docker tag iterative-co-design:latest yourusername/iterative-co-design:latest
docker tag iterative-co-design:latest yourusername/iterative-co-design:v1.0-mamba

# Push
docker push yourusername/iterative-co-design:latest
docker push yourusername/iterative-co-design:v1.0-mamba
```

### Current Deployment Status
- ✅ **Docker Hub**: `mrcha033/iterative-co-design:latest`
- ✅ **Size**: 26.4GB
- ✅ **Digest**: `sha256:3bbfe9b0e0e18aa262b57abec3f96aa8f5e06c45bd35d79acb011155ad2951f3`
- ✅ **Public Access**: Available worldwide

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. GPU Not Detected
```bash
# Check NVIDIA Docker installation
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Expected output: GPU information
# If failed: Reinstall NVIDIA Docker
```

#### 2. Docker Hub Login Issues

**Problem**: "unauthorized: incorrect username or password"

**Solutions**:
```bash
# Option A: Check credentials in browser
# Visit https://hub.docker.com/login
# Verify username/password work

# Option B: Use Access Token (if 2FA enabled)
# Docker Hub → Account Settings → Security → New Access Token
docker login -u username
# Use token as password

# Option C: Create repository manually
# Visit https://hub.docker.com/repositories
# Create "iterative-co-design" repository
```

#### 3. Out of Memory/Disk Space
```bash
# Clean up Docker
docker system prune -a

# Remove unused images
docker image prune -a

# Check disk usage
docker system df
```

#### 4. Mamba Import Errors
```bash
# Verify Mamba installation
docker-compose run base python -c "import mamba_ssm; print('✅ Mamba working')"

# If failed, use CPU-only mode
docker-compose -f docker-compose.cpu.yml run trainer
```

#### 5. CUDA Version Mismatch
```bash
# Check CUDA driver version
nvidia-smi

# Ensure driver ≥ 520.61.05 for CUDA 11.8 support
# Update driver if needed
```

### Environment Verification Checklist

When running `docker-compose run base`, expect:
```
🔍 Environment Verification
===========================
PyTorch: 2.2.2+cu118
CUDA available: True
CUDA devices: 1
✅ BERT models: Available
✅ mamba-ssm: Import successful
✅ Transformers Mamba: Available

🚀 Ready to run experiments!
```

### Alternative Solutions

#### If Docker Hub fails:
1. **GitHub Container Registry**: Use `ghcr.io` instead
2. **Local Build**: Users build from source
3. **File Sharing**: Share Docker image as tar file

#### If GPU issues persist:
1. **CPU Mode**: Use `docker-compose.cpu.yml`
2. **Cloud GPU**: Use Google Colab, AWS, etc.
3. **Local Installation**: Follow manual setup guide

## ⚡ Performance Tips

### Build Optimization
```bash
# Use multi-core builds
docker-compose build --parallel

# Use build cache
docker-compose build --no-cache  # only when needed
```

### Runtime Optimization
```bash
# Allocate more memory (Docker Desktop)
# Settings → Resources → Memory: 8GB+

# Use SSD for Docker data
# Better I/O performance

# Clean up between runs
docker-compose down --volumes
```

### Memory Management
```bash
# Monitor memory usage
docker stats

# Limit container memory
docker-compose run --memory=8g trainer
```

## 📚 Additional Resources

- **Full Documentation**: [README.md](README.md)
- **Configuration Files**: [configs/](configs/)
- **Scripts**: [scripts/](scripts/)
- **Test Suite**: [tests/](tests/)
- **Example Notebooks**: [notebooks/](notebooks/)

## 🎯 Success Metrics

After following this guide, you should achieve:
- ✅ **5-minute setup** with pre-built images
- ✅ **Stable Mamba support** with proven configuration
- ✅ **Global accessibility** via Docker Hub
- ✅ **Multiple usage modes** (GPU/CPU/local/cloud)
- ✅ **Comprehensive troubleshooting** for common issues

---

**Goal**: Enable researchers worldwide to immediately start iterative co-design experiments without complex environment setup!

## 🛠️ Enhanced Features

### Complete Experiment Suites (NEW!)

Run all 5 co-design methods automatically:

**Mamba Complete Suite:**
```bash
# Runs: dense → permute_only → sparsity_only → linear_pipeline → iterative
docker-compose -f docker-compose.hub.yml run all-train
```

**BERT Complete Suite:**
```bash
# Runs: dense → permute_only → sparsity_only → linear_pipeline → iterative
docker-compose -f docker-compose.hub.yml run bert-all-train
```

**Progress Tracking:**
- 📊 Real-time progress updates
- ✅ Clear completion status
- 📁 Results automatically saved to `outputs/` and `results/`

### Enhanced Jupyter Environment (NEW!)

**Included Visualization Libraries:**
- 📊 **matplotlib, seaborn**: Statistical plotting
- 🎨 **plotly, bokeh**: Interactive visualizations
- 📈 **altair**: Grammar of graphics
- 🔧 **ipywidgets**: Interactive widgets
- 📋 **pandas-profiling**: Data analysis reports
- 📊 **tensorboard**: Experiment monitoring

**Features:**
- 🚀 **Jupyter Lab** (modern interface)
- 🔍 **TensorBoard integration** on port 6006
- 📁 **Auto-mounted volumes** for outputs/results
- 🌐 **No authentication** for quick access

### TensorBoard Monitoring (NEW!)

**Real-time Experiment Tracking:**
```bash
# Start TensorBoard
docker-compose -f docker-compose.hub.yml up tensorboard

# Access at http://localhost:6007
```

**Features:**
- 📊 Live experiment metrics
- 📈 Loss and accuracy curves  
- 🔄 Auto-refresh every 30 seconds
- 📁 Monitors `outputs/` directory

### Available Services Summary

| Service | Purpose | Access |
|---------|---------|--------|
| `base` | Environment verification | Terminal output |
| `trainer` | Single BERT experiment | Terminal logs |
| `mamba-trainer` | Single Mamba experiment | Terminal logs |
| `all-train` | **All Mamba methods** | Terminal progress |
| `bert-all-train` | **All BERT methods** | Terminal progress |
| `shell` | Interactive development | Terminal access |
| `jupyter` | **Enhanced analysis** | http://localhost:8888 |
| `tensorboard` | **Experiment monitoring** | http://localhost:6007 | 