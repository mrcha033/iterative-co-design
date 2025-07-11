# Use NVIDIA PyTorch base image with CUDA 11.8
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with exact versions
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create non-root user for security
RUN useradd -m -s /bin/bash researcher && \
    chown -R researcher:researcher /workspace

# Switch to non-root user
USER researcher

# Set default command
CMD ["/bin/bash"]