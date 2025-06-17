# Build stage for CUDA tools
FROM nvidia/cuda:12.4-base-ubuntu22.04 as cuda_tools

# Install NVIDIA tools (ncu, etc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-nsight-compute-12-4 \
    && rm -rf /var/lib/apt/lists/*

# Final stage
FROM nvidia/cuda:12.4-base-ubuntu22.04

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages and clean up in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy NVIDIA tools from build stage
COPY --from=cuda_tools /usr/local/cuda-12.4/nsight-compute-2024.1.0 /usr/local/cuda-12.4/nsight-compute-2024.1.0
ENV PATH=/usr/local/cuda-12.4/nsight-compute-2024.1.0:$PATH

# Install Miniconda efficiently
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && echo "source /opt/conda/bin/activate co-design" >> ~/.bashrc

ENV PATH=$CONDA_DIR/bin:$PATH

# Create Conda environment
COPY environment.yml .
RUN conda env create -f environment.yml \
    && conda clean -afy

# Set up workspace
WORKDIR /workspace
COPY . .

# Set default shell to use conda environment
SHELL ["conda", "run", "-n", "co-design", "/bin/bash", "-c"]

CMD ["echo", "Environment ready. Run experiments with 'python scripts/run_experiment.py ...'"] 