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

- ✅ **Comprehensive test coverage** (71 tests across all modules)
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

- **CPU-only**: Faster to install (~300MB), suitable for testing and development
- **GPU (CUDA)**: Required for full experiments and NVIDIA Nsight Compute profiling (~2-3GB)
- **Installation size**: CUDA PyTorch can be 2-3GB. For CI/testing, use CPU-only version to save disk space
- **Disk space issues**: If you encounter "No space left on device" errors, use the minimal installation option above

#### Mamba Model Support (Optional)

This project supports **Mamba** models from `state-spaces/mamba-2.8b-hf`, but due to compilation complexity, they are **optional**.

**For stable usage (recommended):**
- Use BERT models: `python scripts/run_experiment.py model=bert_base`
- Transformers 4.36+ (stable version)

**For Mamba models (advanced users only):**
- **Requirements**: A100 GPU, CUDA 12.1, compilation tools
- **Transformers**: >=4.39.0 (Mamba support)
- **CUDA dependencies**: `causal-conv1d>=1.2.0`, `mamba-ssm>=1.2.0`
- **Installation**: Use the dedicated script below

#### Verify Installation

```bash
# Quick verification
python -c "import torch; import numpy; import yaml; print('All core dependencies available')"

# Comprehensive dependency check (recommended)
python scripts/check_test_dependencies.py

# Auto-install missing dependencies if needed
python scripts/check_test_dependencies.py --install

# Run basic functional test
pytest tests/test_config.py::TestConfigLoader::test_load_yaml_config_basic -v

# Or use the automated test script (handles dependencies)
bash scripts/run_tests.sh
```

#### Troubleshooting Installation Issues

##### Error: "Could not find a version that satisfies the requirement numpy==X.X.X"

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
   pip install 'numpy>=1.21.0' 'torch>=2.0.0' 'transformers>=4.39.0'
   pip install -e . --no-deps
   ```

### Advanced: Mamba Model Installation

**⚠️ Warning**: Mamba installation is complex and may fail. Use BERT models for stable experiments.

**On A100 GPU (Ubuntu/Linux):**
```bash
# Method 1: Use dedicated installation script
bash scripts/install_mamba.sh

# Method 2: Manual installation
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.39.0
pip install causal-conv1d>=1.2.0 --no-build-isolation
pip install mamba-ssm>=1.2.0 --no-build-isolation
```

**Troubleshooting Mamba Installation:**
- **CUDA version mismatch**: Use `torch==2.3.1` with CUDA 12.1
- **Compilation errors**: Install build tools: `apt-get install build-essential`
- **404 errors**: Try `--no-build-isolation` flag
- **Alternative**: Use BERT models instead

##### Error: "No module named 'src'"

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

**⚠️ SECURITY NOTE for data/download_datasets.sh:**

- The script may attempt to install `aria2` using system package managers (apt-get, yum)
- This could require `sudo` privileges for system-wide installation
- **Use `--install-aria2` flag to explicitly consent to potential sudo operations**
- **Alternative**: Install aria2 manually (`sudo apt-get install aria2`) or use conda (`conda install -c conda-forge aria2`)

```bash
# Safe usage - explicitly consent to aria2 installation
bash data/download_datasets.sh --install-aria2

# Help and security information
bash data/download_datasets.sh --help
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

## Responsible Use

This research focuses on performance optimization techniques for neural networks. While the methods are generally applicable, users should be aware of the following considerations:

**Intended Use**: This codebase is designed for academic research and development of more efficient AI systems. The optimization techniques (sparsity and memory layout) are intended to reduce computational costs and energy consumption.

**Limitations & Biases**:

- Performance improvements may vary significantly across different model architectures, hardware configurations, and datasets
- The evaluation focuses on specific model families (BERT, Mamba) and tasks (sentiment analysis, language modeling) - results may not generalize to all domains
- Dataset biases present in SST-2 and WikiText-103 may be preserved or amplified through optimization

**Ethical Considerations**:

- More efficient models can democratize access to AI capabilities, but may also lower barriers to potentially harmful applications
- Users should evaluate the appropriateness of optimized models for their specific use cases
- When deploying optimized models, consider the same ethical guidelines that apply to the base models

**Dual Use**: While optimization techniques themselves are neutral, they could potentially be used to make both beneficial and harmful AI systems more efficient. Users are responsible for ensuring their applications align with ethical AI principles.

---

## Running Experiments

### 🐳 Docker Method (Recommended)

Docker provides a stable environment for running experiments without dependency issues.

📖 **[Docker Guide](DOCKER_GUIDE.md)**

#### Option 1: Use Pre-built Image (Recommended)

```bash
# 1. Pull pre-built image from Docker Hub
docker pull mrcha033/iterative-co-design:latest

# 2. Use pre-built image with docker-compose
# For GPU environments:
docker-compose -f docker-compose.hub.yml run base

# For CPU-only environments (Windows/WSL):
docker-compose -f docker-compose.cpu.yml run base

# Run experiments:
docker-compose -f docker-compose.cpu.yml run trainer
```

**Available Images:**
- `mrcha033/iterative-co-design:latest` - Latest stable build
- `mrcha033/iterative-co-design:v1.0-mamba` - Version with Mamba support

#### Option 2: Build Locally

```bash
# 1. Build image locally
docker-compose build base

# 2. Verify environment
docker-compose run base

# 3. Run BERT experiment (stable)
docker-compose run trainer

# 4. Run Mamba experiment (advanced)
docker-compose run mamba-trainer

# 5. Interactive shell
docker-compose run shell

# 6. Jupyter notebook (http://localhost:8888)
docker-compose up jupyter
```

#### Custom Experiment Execution

```bash
# Run specific experiment command
docker-compose run --rm trainer bash -c "
  source /opt/conda/bin/activate iterative-co-design &&
  python scripts/run_experiment.py model=bert_base dataset=sst2 method=iterative
"

# Check result files
ls outputs/
ls results/
```

#### Docker Advantages

✅ **Environment Isolation**: Independent execution environment from host system  
✅ **Stable Mamba Support**: Pre-built mamba-ssm wheel (v2.2.4) for reliable installation  
✅ **CUDA 11.8 Compatibility**: Proven stable configuration with PyTorch 2.2.2  
✅ **Reproducibility**: Consistent experiment results in identical environment  
✅ **GPU Support**: NVIDIA Docker with CUDA 11.8 + cuDNN 8  
✅ **No Conda Overhead**: Direct pip installation for faster builds

#### Docker Requirements

- **NVIDIA Docker**: Required for GPU usage
- **CUDA Driver**: Compatible with CUDA 11.8 (driver ≥ 520.61.05)
- **Memory**: Minimum 8GB RAM (for image building)
- **Disk**: Minimum 12GB free space

### 💻 Local Installation Method

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

**⚠️ Note**: The `figures/` directory starts empty. All figures must be generated by running the scripts below.

##### All Figures (1-4)

```bash
python scripts/generate_all_figures.py
```

##### Specific Figure

```bash
python scripts/generate_all_figures.py --figure 1  # Just Figure 1
python scripts/generate_all_figures.py --figure 2  # Just Figure 2
# ... etc
```

##### Quick Mode (for testing)

```bash
python scripts/generate_all_figures.py --quick
```

##### Interactive Figure Generation (Jupyter)

```bash
jupyter notebook notebooks/1_explore_correlation.ipynb
```

**Available Figures:**

- **Figure 1**: Random vs. Optimized Permutation Latency (~25-35% improvement)
- **Figure 2**: Quantization Co-Design Strategies (iterative vs. linear)
- **Figure 3**: Metrics vs. Iteration (causal chain in action)
- **Figure 4**: Pareto Frontier (all methods comparison)

#### Quantization Experiments (CPU-Only)

**⚠️ Important**: Quantization experiments use PyTorch's dynamic quantization, which **only supports CPU inference**. This is a PyTorch limitation, not a project limitation.

```bash
# Quantization experiments - always run on CPU
python scripts/run_quant_test.py model=mamba_3b method=quant_then_permute
python scripts/run_quant_test.py model=mamba_3b method=permute_then_quant  
python scripts/run_quant_test.py model=mamba_3b method=permute_quant_repermute
```

**Key Points:**
- Quantized models automatically run on CPU regardless of CUDA availability
- Latency measurements for quantization compare CPU performance only
- This is consistent with PyTorch's current quantization API limitations
- Other experiments (sparsity, permutation) can use GPU when available

For more details, see the warning in [`scripts/run_quant_test.py`](scripts/run_quant_test.py).

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

#### Automatic Cleanup

The experiment runner automatically cleans up old experiment outputs before starting new runs to manage disk space:

- **Default Settings**: Removes experiment directories older than 30 days from `outputs/` and `multirun/` directories
- **Configuration**: Cleanup settings are defined in `configs/defaults.yaml` under the `storage` section
- **Dry Run Support**: When using `dry_run=true`, cleanup operations are shown but not executed
- **Error Handling**: Cleanup failures are logged but don't prevent experiments from running

To modify cleanup behavior, edit `configs/defaults.yaml`:

```yaml
storage:
  cleanup_older_than_days: 14  # Clean up files older than 2 weeks
```

To disable cleanup entirely, remove or comment out the `cleanup` section in `configs/config.yaml`.

### Testing

The project includes comprehensive tests covering all core modules with deterministic behavior and robust error handling.

#### Prerequisites for Running Tests

**⚠️ Important**: Tests require specific dependencies to be installed. Running `pytest` without proper setup will fail with import errors.

##### 🚀 Quickest Setup (Recommended)

```bash
pip install -e .[test]
```

This installs all required dependencies including PyTorch, NumPy, and testing frameworks in one command.

##### If you see "ModuleNotFoundError" when running pytest:

```bash
# Solution 1: Install test dependencies 
pip install -e .[test]

# Solution 2: Manual dependency installation
pip install torch numpy scikit-learn transformers datasets pyyaml hydra-core omegaconf pytest
pip install -e .

# Solution 3: Use setup script
python scripts/setup.py --test
```

##### Environment Isolation (Highly Recommended)

For best results, use a virtual environment to avoid dependency conflicts:

```bash
# Create and activate virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install test dependencies
pip install -e .[test]

# Verify installation
python -c "import torch, numpy, co_design; print('✅ All test dependencies available')"
```

##### Dependency Verification

Before running tests, verify all required packages are available:

```bash
# Quick verification command
python -c "
import sys
required = ['torch', 'numpy', 'sklearn', 'transformers', 'datasets', 'yaml', 'hydra', 'omegaconf', 'pytest']
missing = []
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f'❌ Missing packages: {missing}')
    print('Run: pip install -e .[test]')
    sys.exit(1)
else:
    print('✅ All required test dependencies are available')
"
```

##### Required Dependencies for Tests

If `pip install -e .[test]` doesn't work in your environment, install these packages manually:

**Core Dependencies:**
- `torch` (>=2.0.0) - Neural network framework
- `numpy` (>=1.21.0) - Numerical computing
- `scikit-learn` (>=1.0.0) - Machine learning utilities (for SpectralClustering)
- `transformers` (>=4.30.0) - Pre-trained models
- `datasets` (>=2.0.0) - Dataset loading
- `pyyaml` (>=6.0) - Configuration file parsing
- `hydra-core` (>=1.3.0) - Configuration management
- `omegaconf` (>=2.3.0) - Configuration framework

**Testing Framework:**
- `pytest` (>=7.0.0) - Test runner
- `pytest-cov` (>=4.0.0) - Coverage reporting

**Quick Manual Installation:**

```bash
pip install torch numpy scikit-learn transformers datasets pyyaml hydra-core omegaconf pytest pytest-cov
pip install -e .
```

#### Network Access Requirements

**⚠️ Network Note**: Some operations require internet access:
- **First-time dataset downloads**: Uses Hugging Face Hub to download WikiText-103 and SST-2
- **Model downloads**: Downloads pre-trained models (BERT, Mamba) from Hugging Face
- **Package installation**: Installing dependencies from PyPI

**Offline Usage**: Once datasets and models are cached locally, most experiments can run offline. Datasets are cached in `~/.cache/huggingface/` by default.

##### Alternative Installation Methods

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

**Test Performance Expectations:**

- **Total runtime**: ~15-30 seconds on modern CPU (varies by system)
- **Test count**: 71 tests across 6 test modules
- **Skipped tests**: 2 GPU tests will be skipped on CPU-only systems (normal behavior)
- **Memory usage**: Peak ~1-2GB RAM (PyTorch model loading)

**Troubleshooting Test Issues:**

1. **GPU tests skipped**: Normal on CPU-only systems - tests will skip gracefully
2. **Slow tests**: Use `pytest -x` to stop on first failure for faster debugging
3. **Permission errors**: Run `pip install --user` if you encounter permission issues
4. **Version conflicts**: Update pip with `python -m pip install --upgrade pip setuptools wheel`
5. **Import errors**: If you still get import errors after `pip install -e .[test]`, try reinstalling: `pip uninstall iterative-co-design && pip install -e .[test]`
6. **PyTorch version issues**: If you get tensor operation errors, ensure PyTorch >= 2.0.0 with `pip install torch>=2.0.0`
7. **Windows path issues**: On Windows, use forward slashes or raw strings in paths
8. **Conda environment conflicts**: If using conda, consider `conda install pytorch` instead of pip for PyTorch

**Environment-Specific Notes:**

- **macOS Apple Silicon**: PyTorch MPS backend may cause warnings (safe to ignore)
- **Windows**: Some tests may run slower due to Windows Defender scanning
- **Docker/CI**: Use `pytest --tb=short -q` for cleaner output in automated environments
- **Corporate networks**: Some tests download small datasets - ensure internet access or pre-download datasets

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

- **71 total tests** covering all major components
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
