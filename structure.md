# Project Structure for "Iterative Co-Design"

This document outlines the project structure designed to systematically implement and reproduce the key experiments from the paper, "The Orthogonality Fallacy: Iterative Co-Design as a First-Class Principle for Efficient AI." The structure emphasizes modularity, reproducibility, and scalability.

## 1. Top-Level Directory Structure

The overall project layout is as follows:

```
iterative-co-design/
├── .github/                  # CI workflows for automated testing
│   └── workflows/
│       └── test.yml         # Comprehensive CI workflow (multi-Python, linting, tests)
├── configs/                  # Hydra configuration files for experiments
│   ├── config.yaml          # Main configuration entry point
│   ├── defaults.yaml        # Common configurations and target paths
│   ├── model/              # Model-specific configurations
│   └── dataset/            # Dataset-specific configurations
├── data/                    # Scripts for downloading and preprocessing datasets
│   ├── download_datasets.sh # Uses aria2c for fast, cached downloads
│   └── LICENSES.md         # Dataset licenses and attributions
├── docs/                    # Source files for the documentation website
│   ├── index.md
│   └── api/                # Auto-generated API docs
├── notebooks/               # Jupyter notebooks for analysis and visualization
│   ├── 1_explore_correlation.ipynb  # Updated with Figure 1 generation capability
│   └── 2_analyze_results.ipynb      # Analysis of experimental results (Tables & Figures 2-4)
├── figures/                 # Generated figures for the paper
│   ├── figure1_mamba_latency_scan_vs_perm.pdf  # Figure 1: Random vs optimized permutation latency
│   ├── figure1_data.json                       # Raw data for Figure 1
│   └── (other generated figures)
├── outputs/                 # Default output directory for Hydra (organized by YYYY-MM/DD-HHMM)
├── scripts/                 # Executable scripts for running experiments
│   ├── run_experiment.py
│   ├── run_quant_test.py
│   └── generate_all_figures.py # Comprehensive figure generation suite
├── src/                     # Core logic and reusable source code modules
│   ├── co_design/
│   ├── models/
│   └── utils/
│       ├── cleanup.py      # Manages old experiment outputs
│       ├── profiler.py     # Hardware profiling with GPU fallback
│       └── logging.py      # W&B logging with offline mode support
├── sweeps/                  # Weights & Biases sweep configurations
│   └── hds_tuning_sweep.yaml
├── tests/                   # Comprehensive automated tests (71 tests total)
│   ├── test_config.py      # Configuration loading tests
│   ├── test_iasp.py        # IASP algorithm and correlation tests  
│   ├── test_modularity.py  # Modularity calculation tests
│   ├── test_profiler.py    # Latency profiling and caching tests
│   ├── test_wrapper.py     # Model wrapper functionality tests
│   └── requirements.txt    # Test-specific dependencies
│
├── Dockerfile              # Multi-stage build for optimized image size
├── docker-compose.yml      # Orchestrates multi-service Docker applications
├── environment.yml         # Conda environment with conda-forge channel
├── mkdocs.yml             # Documentation with auto-API generation
├── README.md              # Project description and setup instructions
└── requirements.txt       # Pinned Python dependencies
```

## 2. Detailed Description of Components

### ⚙️ `configs/`

This directory uses [Hydra](https://hydra.cc/) to manage all hyperparameters and settings, with improved organization:

*   `config.yaml`: Main entry point with monthly output organization and debug presets
*   `defaults.yaml`: Centralizes common configurations and target paths
*   `model/` & `dataset/`: Modular configurations that inherit from defaults

### 📦 `data/`

Scripts and documentation for dataset management:

*   `download_datasets.sh`: Uses aria2c for efficient downloads with proper caching
*   `LICENSES.md`: Documents dataset licenses and attributions

### 🔬 `src/` (Source Code)

The core logic with improved utilities:

*   **`src/co_design/`**: Core co-design implementations
    *   `iasp.py`: IO-Aware Scan Permutation with modularity optimization
    *   `hds.py`: Hardware-Native Differentiable Sparsity implementation
    *   `modularity.py`: Graph modularity calculations for cache optimization
*   **`src/models/`**: Model management modules with enhanced permutation handling
    *   `wrapper.py`: Streamlined model wrapper with deterministic permutation operations
    *   `utils.py`: Tensor manipulation utilities
*   **`src/utils/`**: Enhanced utilities with robust error handling
    *   `profiler.py`: Hardware profiling with deterministic caching and graceful GPU fallback
    *   `cleanup.py`: Manages experiment outputs with retention policies
    *   `logging.py`: W&B integration with offline mode support
    *   `evaluation.py`: Perplexity and accuracy calculation with improved data handling

### 🚀 `scripts/`

Command-line entry points with debug support:

*   `run_experiment.py`: Now supports quick-run debug mode via `+experiment=debug`
*   `run_quant_test.py`: Quantization experiments with profiler integration

### 📈 Experiment Tracking & Storage

*   **Results Organization**: 
    - Outputs stored in `YYYY-MM/DD-HHMM` format
    - Automatic cleanup of old results via `cleanup.py`
    - Configurable retention policy
*   **W&B Integration**:
    - Offline mode by default for CI
    - Cached logging for network-constrained environments
*   **Figure Generation**:
    - Dedicated `figures/` directory for publication-quality outputs (initially empty)
    - Figures generated by running `python scripts/generate_all_figures.py`
    - JSON data files accompanying each figure for reproducibility

### 🧪 Testing & CI

*   **Tests**: 
    - `pytest` markers for CUDA vs CPU tests
    - Comprehensive test coverage for all core modules
    - Deterministic hashing tests for reproducible caching
*   **CI Workflow**:
    - `test.yml`: Multi-Python version testing (3.8-3.11), linting with ruff, and import validation
    - Smart dependency caching for faster CI runs

### 🐳 Containerization & Environment

Improved container and environment management:

*   `Dockerfile`: 
    - Multi-stage build for minimal image size
    - CUDA 12.4 base image
    - Optimized layer caching
*   `environment.yml`: 
    - conda-forge channel priority
    - Explicit version pins
*   `requirements.txt`: 
    - Fully pinned dependencies
    - Development tools (ruff, pytest)

### 📚 Documentation

Enhanced documentation setup:

*   **`docs/`**: 
    - Auto-generated API reference
    - Integration guides
*   **`mkdocs.yml`**: 
    - Material theme configuration
    - Auto-rebuild on changes

## 3. Experimental Workflow Example

### Setup

Choose **one** of the following methods:

1.  **Docker (Recommended)**:
    ```bash
    # Build optimized image
    docker-compose build
    # Run with GPU support
    docker-compose run --gpus all --rm trainer /bin/bash
    ```

2.  **Conda**:
    ```bash
    conda env create -f environment.yml
    conda activate co-design
    ```

3.  **pip**:
    ```bash
    pip install -r requirements.txt
    ```

Download datasets with caching:
```bash
# Uses aria2c for efficient downloads
HF_HOME=/path/to/cache ./data/download_datasets.sh
```

### Run Experiments

Examples with new features:

```bash
# Quick debug run
python scripts/run_experiment.py +experiment=debug

# Production run with monthly organization
python scripts/run_experiment.py model=mamba_3b dataset=wikitext103

# Generate Figure 1 (Random vs Optimized Permutation Latency)
python scripts/generate_figure1.py model=mamba_3b dataset=wikitext103

# Generate all paper figures
python scripts/generate_all_figures.py

# Offline mode for air-gapped environments
WANDB_MODE=offline python scripts/run_experiment.py
```

### Maintenance

Manage experiment outputs:
```bash
# Preview cleanup (dry run)
python -c "from src.utils.cleanup import cleanup_old_runs; cleanup_old_runs(['outputs', 'multirun'], max_age_days=30, dry_run=True)"

# Actually cleanup old results
python -c "from src.utils.cleanup import cleanup_old_runs; cleanup_old_runs(['outputs', 'multirun'], max_age_days=30)"
```

This updated structure reflects the improvements in reproducibility, efficiency, and maintainability of the project.
 
 