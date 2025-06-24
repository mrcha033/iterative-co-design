# Multi-stage Docker build for iterative-co-design with Mamba support
# Compatible with CUDA 12.1 and A100 GPU

# Build stage for CUDA tools
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as cuda_tools

# Install build tools and NVIDIA tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cuda-nsight-compute-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Final stage
FROM nvidia/cuda:12.1-base-ubuntu22.04

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages and build tools for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    bzip2 \
    build-essential \
    ninja-build \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy NVIDIA tools from build stage
COPY --from=cuda_tools /usr/local/cuda-12.1/nsight-compute-2023.3.1 /usr/local/cuda-12.1/nsight-compute-2023.3.1
ENV PATH=/usr/local/cuda-12.1/nsight-compute-2023.3.1:$PATH

# Install Miniconda efficiently
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# Create and activate environment
RUN conda create -n iterative-co-design python=3.10 -y \
    && conda clean -afy

# Activate environment and install core dependencies
ENV CONDA_DEFAULT_ENV=iterative-co-design
ENV PATH=/opt/conda/envs/iterative-co-design/bin:$PATH

# Install PyTorch with CUDA 12.1 support
RUN /opt/conda/envs/iterative-co-design/bin/pip install \
    torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN /opt/conda/envs/iterative-co-design/bin/pip install \
    numpy>=1.21.0 \
    transformers>=4.36.0 \
    datasets>=2.14.0 \
    scikit-learn>=1.3.0 \
    pandas>=1.5.0 \
    tqdm>=4.64.0 \
    pyyaml>=6.0 \
    hydra-core>=1.3.0 \
    omegaconf>=2.3.0 \
    wandb>=0.15.0 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    pytest>=8.0.2

# Set up workspace
WORKDIR /workspace
COPY . .

# Install the package
RUN /opt/conda/envs/iterative-co-design/bin/pip install -e .

# Optional: Install Mamba dependencies (may fail, but will continue)
RUN /opt/conda/envs/iterative-co-design/bin/pip install \
    transformers>=4.39.0 \
    causal-conv1d>=1.2.0 \
    mamba-ssm>=1.2.0 \
    --no-build-isolation || echo "⚠️ Mamba installation failed, continuing with BERT support only"

# Verification script
RUN echo '#!/bin/bash\n\
echo "🔍 Environment Verification"\n\
echo "==========================="\n\
python -c "import torch; print(f\"PyTorch: {torch.__version__}\")"\n\
python -c "import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")"\n\
python -c "import torch; print(f\"CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}\")"\n\
python -c "from transformers import BertModel; print(\"✅ BERT models: Available\")"\n\
python -c "from transformers import MambaForCausalLM; print(\"✅ Mamba models: Available\")" 2>/dev/null || echo "❌ Mamba models: Not available (use BERT instead)"\n\
echo ""\n\
echo "🚀 Ready to run experiments!"\n\
echo "  BERT: python scripts/run_experiment.py model=bert_base dataset=sst2 method=dense"\n\
echo "  Mamba: python scripts/run_experiment.py model=mamba_3b dataset=wikitext103 method=dense"\n\
' > /workspace/verify_env.sh && chmod +x /workspace/verify_env.sh

# Set default shell to use conda environment  
SHELL ["/opt/conda/envs/iterative-co-design/bin/python", "-c"]

# Default command
CMD ["/bin/bash", "/workspace/verify_env.sh"] 