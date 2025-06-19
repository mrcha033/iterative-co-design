# Official Code for "The Orthogonality Fallacy: Iterative Co-Design as a First-Class Principle for Efficient AI"

This repository contains the official implementation for the paper, "The Orthogonality Fallacy." The work dismantles the assumption that algorithmic optimizations (like sparsity) and hardware-level optimizations (like memory layout) are separable problems. It introduces an **Iterative Co-Design** framework that alternates between algorithmic state changes and memory layout optimization to find a more efficient Pareto-optimal model.

## Core Concepts

- **Orthogonality Fallacy**: The mistaken belief that algorithmic and hardware optimizations can be performed independently without loss of optimality.
- **Iterative Co-Design**: A novel framework that creates a feedback loop between algorithmic changes (e.g., **Hardware-Native Differentiable Sparsity, HDS**) and hardware-interface optimization (e.g., **IO-Aware Scan Permutation, IASP**).
- **Modularity**: A key metric used by IASP to find cache-friendly permutations of a model's state dimensions, which is shown to have a causal link to reducing latency.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository_url>
cd iterative-co-design
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

#### Option A: Automated Setup Script (Recommended)

```bash
# For CPU-only PyTorch (faster installation, good for testing)
python scripts/setup.py --device cpu --test

# For GPU PyTorch (required for full experiments with CUDA)
python scripts/setup.py --device cuda --dev --test

# For development with all dependencies
python scripts/setup.py --device cuda --dev
```

#### Option B: Manual Installation

```bash
# Install with pip (uses CPU PyTorch by default)
pip install -e .

# OR install from requirements.txt (for development)
pip install -r requirements.txt -r tests/requirements.txt
```

#### PyTorch Installation Notes

- **CPU-only**: Faster to install, suitable for testing and development
- **GPU (CUDA)**: Required for full experiments and NVIDIA Nsight Compute profiling
- **Installation size**: PyTorch can be 1-3GB. The setup script uses CPU wheels by default for faster CI/development

#### Verify Installation

```bash
# Quick verification
python -c "import torch; import numpy; import yaml; print('All core dependencies available')"

# Run basic functional test
pytest tests/test_config.py::TestConfigLoader::test_load_yaml_config_basic -v
```

#### Troubleshooting Installation Issues

**Error: "Could not find a version that satisfies the requirement numpy==X.X.X"**

This usually means your Python version or pip is too old. Try:

1. **Update pip and build tools:**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

2. **Check Python version (3.9+ recommended):**
   ```bash
   python --version
   ```

3. **Use flexible versions instead of pinned ones:**
   ```bash
   pip install -e .  # Uses flexible requirements from pyproject.toml
   ```

4. **Manual installation for compatibility:**
   ```bash
   pip install 'numpy>=1.21.0' 'torch>=2.0.0' 'transformers>=4.30.0'
   pip install -e . --no-deps
   ```

**Error: "No module named 'src'"**

Make sure to set PYTHONPATH or install in editable mode:
```bash
export PYTHONPATH=$(pwd)  # On Windows: set PYTHONPATH=%cd%
# OR
pip install -e .
```

### 3. Download Datasets

The required datasets (`wikitext-103-raw-v1` and `sst2`) are downloaded and cached automatically by the `datasets` library when you run an experiment for the first time. Alternatively, you can pre-download them using the provided script.

On Linux or macOS:
```bash
bash data/download_datasets.sh
```

On Windows (or as an alternative):
```bash
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-raw-v1')"
python -c "from datasets import load_dataset; load_dataset('glue', 'sst2')"
```

#### Dataset Licenses

This project uses standard academic datasets:

- **SST-2 (Stanford Sentiment Treebank)**: Academic benchmark for sentiment analysis. Original paper by Socher et al. (2013). Generally available for research use.
- **WikiText-103**: Licensed under [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/). You are free to share and adapt the data with proper attribution.

For complete licensing details, see [`data/LICENSES.md`](data/LICENSES.md).

---

## Running Experiments

All experiments are orchestrated through the main runner script, `scripts/run_experiment.py`.

### Basic Usage

The script uses Hydra for configuration management. Specify configurations using the `key=value` format:
- `model=<model_config>`: The model configuration name (e.g., `mamba_3b`, `bert_base`).
- `dataset=<dataset_config>`: The dataset configuration name (e.g., `wikitext103`, `sst2`).
- `method=<method_name>`: The experimental condition to run.

### Available Methods

- `dense`: (Baseline 1) The original, unmodified model.
- `permute_only`: (Baseline 3) Applies IASP to the dense model.
- `sparsity_only`: (Baseline 2) Applies HDS to the dense model.
- `linear_pipeline`: (Baseline 4) Applies IASP, then HDS.
- `iterative`: (Ours) Applies HDS and IASP in a loop.

### Figure Generation

To reproduce the figures from the paper:

- **Figure 1**: Random vs. Optimized Permutation Latency
- **Figures 2-4**: Analysis results from experiments

### Example Commands

To run the "dense" baseline experiment with the Mamba-3B configuration:

```bash
python scripts/run_experiment.py model=mamba_3b dataset=wikitext103 method=dense
```

To run the iterative co-design method with BERT:

```bash
python scripts/run_experiment.py model=bert_base dataset=sst2 method=iterative
```

#### Generating Paper Figures

**All Figures (1-4)**:
```bash
python scripts/generate_all_figures.py
```

**Specific Figure**:
```bash
python scripts/generate_all_figures.py --figure 1  # Just Figure 1
python scripts/generate_all_figures.py --figure 2  # Just Figure 2
# ... etc
```

**Quick Mode** (for testing):
```bash
python scripts/generate_all_figures.py --quick
```

**Interactive Figure Generation** (Jupyter):
```bash
jupyter notebook notebooks/1_explore_correlation.ipynb
```

**Available Figures:**
- **Figure 1**: Random vs. Optimized Permutation Latency (~25-35% improvement)
- **Figure 2**: Quantization Co-Design Strategies (iterative vs. linear)
- **Figure 3**: Metrics vs. Iteration (causal chain in action)
- **Figure 4**: Pareto Frontier (all methods comparison)

### Model Configuration

Model YAML files under `configs/model` include an `iasp` section that controls
the permutation search. The key `cluster_size_range` defines the minimum and
maximum cluster sizes (in neurons) considered when searching for an optimal
permutation.

### Dry Run

To see the sequence of operations for a method without executing the full, expensive run, use the `dry_run=true` parameter:

```bash
python scripts/run_experiment.py model=mamba_3b dataset=wikitext103 method=iterative dry_run=true
```

### Results

All results, including measured metrics (perplexity, latency, modularity, etc.) and any generated artifacts, are saved as JSON files in the `outputs/` directory, organized by timestamp (managed by Hydra). 

### Testing

**Important**: Tests require dependencies to be installed and PYTHONPATH to be set correctly.

#### Method 1: Using the test script (Recommended)

```bash
./scripts/run_tests.sh
```

This script automatically:
- Installs all required dependencies
- Sets PYTHONPATH correctly  
- Runs pytest with appropriate flags

#### Method 2: Manual testing

```bash
# Install test dependencies (if not already done)
pip install -e .[test]
# OR: pip install -r requirements.txt -r tests/requirements.txt

# Set PYTHONPATH (required for src imports)
export PYTHONPATH=$(pwd)  # On Windows: set PYTHONPATH=%cd%

# Run tests
pytest
```

#### Method 3: Using pyproject.toml configuration

```bash
# Install the package in editable mode
pip install -e .[test]

# Run tests (PYTHONPATH handled by pyproject.toml)
pytest
```

#### Test Requirements

Tests require the following dependencies:
- PyTorch (for model wrapper tests)  
- NumPy (for numerical computations)
- PyYAML (for configuration loading)
- pytest (testing framework)

**Note**: If you get import errors when running `pytest` directly, make sure to either:
1. Use the provided `scripts/run_tests.sh` script, or
2. Set `PYTHONPATH=$(pwd)` before running pytest, or  
3. Install in editable mode with `pip install -e .`
