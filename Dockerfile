# Multi-stage Docker build for iterative-co-design with Mamba support
# Compatible with CUDA 12.1 and A100 GPU

########################  Build-time arguments  ########################
ARG CUDA_VERSION=11.8.0
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04
ARG TORCH_VERSION=2.2.2
ARG CU_TAG=cu118
ARG PYVER=cp310
ARG MAMBA_WHL=mamba_ssm-2.2.4+cu11torch2.2cxx11abiFALSE-${PYVER}-${PYVER}-linux_x86_64.whl

########################  Base image  ##################################
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG TORCH_VERSION CU_TAG PYVER MAMBA_WHL

########################  Env & flags  #################################
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TOKENIZERS_PARALLELISM=false \
    MAX_JOBS=4 \
    MAMBA_SKIP_CUDA_BUILD=TRUE

########################  System packages  #############################
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3 python3-pip python-is-python3 \
        git wget curl unzip vim cmake ninja-build pkg-config \
        libgl1 libgfortran5 && \
    rm -rf /var/lib/apt/lists/*

########################  Python deps  #################################
WORKDIR /workspace
COPY requirements.txt .

# 1) pip upgrade
RUN pip install --no-cache-dir --upgrade pip 'setuptools<70' wheel packaging numpy

# 2) PyTorch & friends - official CU118 index
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}+${CU_TAG}" \
        "torchvision==0.17.2+${CU_TAG}" "torchaudio==2.2.2+${CU_TAG}" \
        --index-url https://download.pytorch.org/whl/${CU_TAG}

# 3) Core dependencies
RUN pip install --no-cache-dir \
    numpy>=1.21.0 \
    scikit-learn>=1.3.0 \
    pandas>=1.5.0 \
    tqdm>=4.64.0 \
    pyyaml>=6.0 \
    hydra-core>=1.3.0 \
    omegaconf>=2.3.0 \
    wandb>=0.15.0 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    pytest>=8.0.2 \
    datasets>=2.14.0

# 4) Install mamba-ssm wheel + transformers
ENV MAMBA_SKIP_CUDA_BUILD=TRUE
RUN wget -q https://github.com/state-spaces/mamba/releases/download/v2.2.4/${MAMBA_WHL} \
        && pip install --no-cache-dir ${MAMBA_WHL} \
        && rm ${MAMBA_WHL}

# Install causal-conv1d
RUN pip install --no-cache-dir causal-conv1d>=1.2.0

# Install transformers with Mamba support
RUN pip install --no-cache-dir "transformers>=4.42.4"

# 5) Copy application source
COPY . .

# 6) Install the package
RUN pip install --no-cache-dir -e .

########################  Diagnostic script  ###########################
RUN python - <<'PY'
import torch
print("✅ PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("✅ CUDA devices:", torch.cuda.device_count() if torch.cuda.is_available() else 0)

try:
    from transformers import BertModel
    print("✅ BERT models: Available")
except ImportError as e:
    print("❌ BERT models: Failed -", e)

try:
    from mamba_ssm import MambaLMHeadModel
    print("✅ mamba-ssm: Import successful")
except ImportError as e:
    print("❌ mamba-ssm: Failed -", e)

try:
    from transformers import MambaForCausalLM
    print("✅ Transformers Mamba: Available")
except ImportError as e:
    print("❌ Transformers Mamba: Failed -", e)

print("\n🚀 Environment ready!")
print("  BERT: python scripts/run_experiment.py model=bert_base dataset=sst2 method=dense")
print("  Mamba: python scripts/run_experiment.py model=mamba_3b dataset=wikitext103 method=dense")
PY

########################  Default entrypoint  ###########################
ENTRYPOINT ["/bin/bash"]
