# Official Code for "The Orthogonality Fallacy: Iterative Co-Design as a First-Class Principle for Efficient AI"

This repository contains the official implementation for the paper, "The Orthogonality Fallacy." The work dismantles the assumption that algorithmic optimizations (like sparsity) and hardware-level optimizations (like memory layout) are separable problems. It introduces an **Iterative Co-Design** framework that alternates between algorithmic state changes and memory layout optimization to find a more efficient Pareto-optimal model.

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute this code for academic and commercial purposes with proper attribution.

## Core Concepts

- **Orthogonality Fallacy**: The mistaken belief that algorithmic and hardware optimizations can be performed independently without loss of optimality.
- **Iterative Co-Design**: A novel framework that creates a feedback loop between algorithmic changes (e.g., **Hardware-Native Differentiable Sparsity, HDS**) and hardware-interface optimization (e.g., **IO-Aware Scan Permutation, IASP**).
- **Modularity**: A key metric used by IASP to find cache-friendly permutations of a model's state dimensions, which is shown to have a causal link to reducing latency.

## 🏆 Code Quality & Reliability

This repository maintains high code quality standards with:
- ✅ **Comprehensive test coverage** (39 tests across all modules)
- ✅ **Deterministic caching** for reproducible results
- ✅ **Multi-Python support** (3.8, 3.9, 3.10, 3.11)
- ✅ **Automated linting** with ruff format + check
- ✅ **Flexible dependency management** for broad compatibility
- ✅ **Robust error handling** and graceful fallbacks

## ✨ Recent Improvements

**All issues resolved as of latest update:**
- ✅ **Module-level docstrings added** to all core modules for better documentation
- ✅ **Dry-run functionality implemented** - use `dry_run=true` to preview experiment operations 
- ✅ **Code formatting standardized** with ruff for consistent style across codebase
- ✅ **Python 3.8 compatibility fixed** - replaced modern type annotations with backward-compatible versions
- ✅ **Import path inconsistencies resolved** - scripts now work in both development and installed environments
- ✅ **Version synchronization** - CITATION.cff and pyproject.toml now use consistent version numbers
- ✅ **Configuration file corruption fixed** - removed null characters causing YAML parsing errors
- ✅ **Learning rate configuration corrected** - HDS now properly reads dataset-specific learning rates
- ✅ **Task-appropriate metrics implemented** - automatically uses accuracy for classification, perplexity for language modeling
- ✅ **Dataset download script enhanced** - handles restricted environments without sudo access
- ✅ **Test dependency documentation** - clear setup instructions prevent import errors

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

This usually means your Python version or pip is too old. The project now uses flexible version requirements for better compatibility:

1. **Update pip and build tools:**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

2. **Check Python version (3.8+ supported, 3.9+ recommended):**
   ```bash
   python --version
   ```

3. **Install with flexible requirements:**
   ```bash
   pip install -e .  # Uses flexible requirements from pyproject.toml
   ```

4. **Manual installation for older environments:**
   ```bash
   pip install 'numpy>=1.21.0' 'torch>=2.0.0' 'transformers>=4.30.0'
   pip install -e . --no-deps
   ```

**Error: "No module named 'src'"**

**Fixed!** All user-facing scripts now use correct import paths that work both in development and after package installation. If you encounter this error with older versions, make sure to:
```bash
export PYTHONPATH=$(pwd)  # On Windows: set PYTHONPATH=%cd%
# OR (recommended)
pip install -e .
```

The scripts in `scripts/` automatically use the correct package structure (`utils.`, `co_design.`, `models.`) rather than development paths (`src.utils.`, etc.).

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

The project includes comprehensive tests covering all core modules with deterministic behavior and robust error handling.

#### Prerequisites for Running Tests

**⚠️ IMPORTANT: Install dependencies before running tests to avoid import errors.**

Tests require several packages that must be installed first. If you encounter errors like:
```
ModuleNotFoundError: No module named 'yaml'
ModuleNotFoundError: No module named 'numpy'  
ModuleNotFoundError: No module named 'torch'
```

This means required dependencies are not installed. Follow one of these installation methods:

**Method 1: Install with test dependencies (Recommended)**
```bash
pip install -e .[test]
```

**Method 2: Use setup script with testing**
```bash
python scripts/setup.py --test
```

**Method 3: Manual installation**
```bash
pip install -r requirements.txt -r tests/requirements.txt
pip install -e .
```

**Method 4: Development setup (includes all dependencies)**
```bash
pip install -e .[dev]  # Includes test, docs, and development tools
```

**What gets installed:**
- **Core packages**: torch, numpy, transformers, datasets, pyyaml
- **Testing framework**: pytest, pytest-cov  
- **Scientific computing**: scikit-learn, pandas, matplotlib, seaborn
- **Configuration**: hydra-core, omegaconf
- **Development tools**: ruff (linting), mkdocs (documentation)

**System Requirements:**
- **Python 3.8+** (3.9+ recommended for best compatibility)
- **Operating Systems**: Windows, macOS, Linux
- **Hardware**: CPU-only systems supported (GPU optional for full experiments)

**Troubleshooting Test Issues:**

1. **Import errors**: Ensure all dependencies are installed using one of the methods above
2. **GPU tests skipped**: Normal on CPU-only systems - tests will skip gracefully 
3. **Slow tests**: Use `pytest -x` to stop on first failure for faster debugging
4. **Permission errors**: Run `pip install --user` if you encounter permission issues
5. **Version conflicts**: Update pip with `python -m pip install --upgrade pip setuptools wheel`

#### Quick Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_config.py -v              # Configuration tests
pytest tests/test_profiler.py -v            # Profiler and caching tests  
pytest tests/test_wrapper.py -v             # Model wrapper tests
```

#### Complete Test Suite

The test suite includes:
- **39 total tests** covering all major components
- **Deterministic hashing tests** for reliable caching
- **GPU fallback testing** (automatically skips when CUDA unavailable)
- **Import validation** for all core modules
- **Error handling** for edge cases

#### CI Testing

Our GitHub Actions workflow automatically:
- Tests across Python 3.8, 3.9, 3.10, 3.11
- Runs linting with ruff (format + check)
- Validates imports work correctly
- Uses smart caching for faster runs

**Note**: All tests now work out-of-the-box with proper `pyproject.toml` configuration. No manual PYTHONPATH setup required!
