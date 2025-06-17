FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set non-interactive frontend to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install git, wget, and other basic dependencies
RUN apt-get update && apt-get install -y git wget bzip2 && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create Conda environment from the environment.yml file
COPY environment.yml .
RUN conda env create -f environment.yml

# Make Conda available in bash shells
RUN echo "source /opt/conda/bin/activate co-design" >> ~/.bashrc

# Set up the working directory and copy the project files
WORKDIR /workspace
COPY . .

# Set the default entrypoint to use the conda environment
SHELL ["conda", "run", "-n", "co-design", "/bin/bash", "-c"]

CMD ["echo", "Environment ready. Run experiments with 'python scripts/run_experiment.py ...'"] 