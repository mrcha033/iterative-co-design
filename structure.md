# Project Structure for "Iterative Co-Design"

This document outlines the project structure designed to systematically implement and reproduce the key experiments from the paper, "The Orthogonality Fallacy: Iterative Co-Design as a First-Class Principle for Efficient AI." The structure emphasizes modularity, reproducibility, and scalability.

## 1. Top-Level Directory Structure

The overall project layout is as follows:

```
iterative-co-design/
├── .github/                  # CI workflows for automated testing
│   └── workflows/
│       └── ci.yml
├── configs/                  # Hydra configuration files for experiments
│   ├── config.yaml           # Main configuration entry point
│   ├── model/                # Model-specific configurations
│   └── dataset/              # Dataset-specific configurations
├── data/                     # Scripts for downloading and preprocessing datasets
│   └── download_datasets.sh
├── docs/                     # Source files for the documentation website
│   ├── index.md
│   └── api/
├── notebooks/                # Jupyter notebooks for analysis and visualization
│   ├── 1_explore_correlation.ipynb
│   └── 2_analyze_results.ipynb
├── results/                  # Default output directory for Hydra (logs, metrics)
├── scripts/                  # Executable scripts for running experiments
│   ├── run_experiment.py
│   └── run_quant_test.py
├── src/                      # Core logic and reusable source code modules
│   ├── co_design/
│   ├── models/
│   └── utils/
├── sweeps/                   # Weights & Biases sweep configurations
│   └── hds_tuning_sweep.yaml
├── tests/                    # Automated tests
│   └── test_modularity.py
│
├── Dockerfile                # Defines the container for a reproducible environment
├── docker-compose.yml        # Orchestrates multi-service Docker applications
├── environment.yml           # Conda environment definition
├── mkdocs.yml                # Configuration for the documentation site
├── README.md                 # Project description and setup instructions
└── requirements.txt          # Python dependencies
```

## 2. Detailed Description of Components

### ⚙️ `configs/`

This directory uses [Hydra](https://hydra.cc/) to manage all hyperparameters and settings, decoupling them from the code. This is a significant improvement over static YAML files, enabling powerful command-line overrides and composition.

*   `config.yaml`: The main entry point for configuration. It defines the default model, dataset, and experiment parameters.
*   `model/` & `dataset/`: Subdirectories containing different configurations for models (e.g., `bert_base.yaml`, `mamba_3b.yaml`) and datasets (e.g., `sst2.yaml`, `wikitext103.yaml`). These can be swapped on the command line.

### 📦 `data/`

Scripts to prepare the necessary datasets.

*   `download_datasets.sh`: A shell script that uses the Hugging Face `datasets` library to download WikiText-103 and SST-2.

### 🔬 `src/` (Source Code)

The core logic of the project resides here.

*   **`src/co_design/`**: Modules for the novel co-design techniques.
    *   `iasp.py`: Implements IO-Aware Scan Permutation (IASP).
    *   `hds.py`: Implements Hardware-Native Differentiable Sparsity (HDS).
    *   `modularity.py`: Calculates the Modularity score that IASP optimizes.
*   **`src/models/`**: Modules for model loading and management.
    *   `wrapper.py`: A wrapper for Hugging Face models with methods to apply permutations.
    *   `utils.py`: Utilities for model `state_dict` manipulation.
*   **`src/utils/`**: Essential utilities.
    *   `profiler.py`: Measures latency and L2 Cache Hit Rate (using NVIDIA `ncu`).
    *   `evaluation.py`: Calculates model performance (Perplexity, Accuracy).
    *   `logging.py`: Initializes Weights & Biases for experiment tracking.

### 🚀 `scripts/`

The command-line entry points for all experiments, powered by Hydra.

*   `run_experiment.py`: Reproduces Tables 1 & 2 from the paper.
    *   The experiment method and all parameters are now set via the command line.
*   `run_quant_test.py`: Reproduces the quantization experiment (Figure 2).

### 📈 Experiment Tracking & Sweeps

The project is integrated with [Weights & Biases (W&B)](https://wandb.ai) for logging metrics, comparing runs, and orchestrating hyperparameter sweeps.

*   **`sweeps/`**: Contains W&B sweep configuration files.
    *   `hds_tuning_sweep.yaml`: An example sweep to tune HDS parameters.
*   **Usage**: Sweeps can be initiated with `wandb sweep sweeps/hds_tuning_sweep.yaml`, followed by running one or more agents with the provided `wandb agent` command.

### 🧪 Testing & CI

*   **`tests/`**: Contains automated tests using `pytest` to ensure correctness of core components like modularity calculations.
*   **`.github/workflows/ci.yml`**: A GitHub Actions workflow that automatically runs a linter (`ruff`) and unit tests (`pytest`) on every push and pull request to the main branch, ensuring code quality and correctness.

### 🐳 Containerization & Environment

To ensure full reproducibility, the project includes multiple ways to set up the environment.

*   `Dockerfile`: A complete definition to build a container image with all necessary system libraries, CUDA, and Python dependencies.
*   `docker-compose.yml`: Orchestrates running services, such as a `trainer` for experiments and a `jupyter` notebook server for analysis.
*   `environment.yml`: A Conda environment file that specifies channels (including `conda-forge`) and core dependencies.

### 📚 Documentation

*   **`docs/`**: Source files for the project's documentation website.
*   **`mkdocs.yml`**: Configuration for [MkDocs](https://www.mkdocs.org/) and the Material for MkDocs theme. The site includes auto-generated API reference documentation from the source code docstrings.

## 3. Experimental Workflow Example

This structure enables a clear, powerful, and reproducible workflow.

### Setup

Choose **one** of the following methods:

1.  **Docker (Recommended)**:
    ```bash
    # Build all services
    docker-compose build
    # Run a container with an interactive shell
    docker-compose run --rm trainer /bin/bash
    ```

2.  **Conda**:
    ```bash
    conda env create -f environment.yml
    conda activate co-design
    ```
3.  **pip**:
    ```bash
    pip install -r requirements.txt # Uses pinned versions for reproducibility
    ```

After setup, download the datasets:
```bash
./data/download_datasets.sh
```

### Run Experiments

Hydra's command-line interface makes it easy to run any experiment. The `method` is now a parameter alongside model and dataset choices.

```bash
# Run the (1) Dense Baseline for BERT-base on SST-2
python scripts/run_experiment.py model=bert_base dataset=sst2 method=dense

# Run the (2) Sparsity-Only (HDS) baseline
python scripts/run_experiment.py model=bert_base dataset=sst2 method=sparsity_only

# Run the (5) Iterative Co-Design experiment for Mamba on WikiText-103
python scripts/run_experiment.py model=mamba_3b dataset=wikitext103 method=iterative num_iterations=2
```

Each run saves its outputs to a unique, timestamped directory in `results/`, and all metrics are logged to Weights & Biases if configured.

### Analysis and Visualization

1.  Open `notebooks/2_analyze_results.ipynb`.
2.  The notebook can be configured to read from local `results/` directories or to pull data directly from the W&B API.
3.  Generate tables and plots that mirror the figures from the paper.

This project structure provides a robust, scalable, and reproducible framework for validating the paper's claims and serves as a solid foundation for future research in co-design.