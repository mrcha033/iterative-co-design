# Official Code for "The Orthogonality Fallacy: Iterative Co-Design as a First-Class Principle for Efficient AI"

This repository contains the official implementation for the paper, "The Orthogonality Fallacy." The work dismantles the assumption that algorithmic optimizations (like sparsity) and hardware-level optimizations (like memory layout) are separable problems. It introduces an **Iterative Co-Design** framework that alternates between algorithmic state changes and memory layout optimization to find a more efficient Pareto-optimal model.

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute this code for academic and commercial purposes with proper attribution.

## Core Concepts

- **Orthogonality Fallacy**: The mistaken belief that algorithmic and hardware optimizations can be performed independently without loss of optimality.
- **Iterative Co-Design**: A novel framework that creates a feedback loop between algorithmic changes (e.g., **Hardware-Native Differentiable Sparsity, HDS**) and hardware-interface optimization (e.g., **IO-Aware Scan Permutation, IASP**).
- **Modularity**: A key metric used by IASP to find cache-friendly permutations of a model's state dimensions, which is shown to have a causal link to reducing latency.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mrcha033/iterative-co-design.git
cd iterative-co-design
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade build tools (recommended)
pip install --upgrade pip setuptools wheel
```

### 3. Choose Your Installation Method

#### Option A: Basic Installation (BERT/GPT models only)
```bash
# Install core dependencies - works with BERT, GPT models
pip install -e .

# Test with BERT model
python scripts/run_experiment.py dataset=sst2 method=dense model=bert_base
```

#### Option B: Full Installation (All models including Mamba)
```bash
# Install core dependencies first
pip install -e .

# Add Mamba support (requires compilation)
pip install -e ".[mamba]"

# Test with Mamba model  
python scripts/run_experiment.py dataset=wikitext103 method=dense model=mamba_3b
```

#### Option C: Development Installation
```bash
# Install all dependencies including testing and documentation tools
pip install -e ".[dev]"
```

#### Option D: Alternative Installation (if editable install fails)
```bash
# Use non-editable installation if you encounter build issues
pip install .
pip install ".[mamba]"  # Optional: add Mamba support
```

### 4. Docker Installation (Recommended for complex dependencies)

```bash
# Build and run with Docker Compose
docker compose build base
docker compose run shell

# Or run specific experiments
docker compose run trainer  # BERT experiments
docker compose run mamba-trainer  # Mamba experiments
```

#### PyTorch Installation Notes

- **CPU-only**: Faster to install (~300MB), suitable for testing and development
- **GPU (CUDA)**: Required for full experiments and NVIDIA Nsight Compute profiling (~2-3GB)
- **Installation size**: CUDA PyTorch can be 2-3GB. For CI/testing, use CPU-only version to save disk space
- **Disk space issues**: If you encounter "No space left on device" errors, use the minimal installation option above

### Mamba Model Support

This project supports **Mamba** models, but they require a specific, complex build environment that is difficult to set up locally.

**⚠️ Using Mamba is for advanced users and is not recommended for local installation.**

#### The Docker Method (Strongly Recommended)

For a hassle-free experience with Mamba, the **Docker environment is the recommended approach**. It comes pre-configured with all the necessary CUDA and C++ build tools, avoiding all local compilation issues.

- **Guaranteed Compatibility**: The Docker image has `mamba-ssm` and `causal-conv1d` pre-compiled against a compatible version of the CUDA toolkit.
- **No Local Setup**: No need to install the CUDA toolkit or C++ compilers on your machine.

See the [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for instructions on how to use the Mamba-ready Docker image.

#### The Local Installation Method (Advanced)

If you cannot use Docker, you can attempt a local installation. This is **not guaranteed to work** and depends heavily on your system configuration.

1.  **Prerequisites**:
    *   A compatible NVIDIA GPU (e.g., A100).
    *   NVIDIA CUDA Toolkit (version 12.1 is recommended) installed, with `nvcc` in your system's `PATH`.
    *   A C++ compiler (like `g++`) installed.

2.  **Run the Installation Script**:

    ```bash
    bash scripts/install_mamba.sh
    ```

    This script will first check for the required build tools and then attempt to install the Mamba dependencies. If the script fails, please refer to the error messages and consider using the Docker environment.

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

## Project Structure

```
iterative-co-design/
├── src/                          # Source code
│   ├── co_design/               # Core optimization algorithms
│   │   ├── hds.py              # Hardware-Native Differentiable Sparsity
│   │   ├── iasp.py             # IO-Aware Scan Permutation
│   │   └── modularity.py       # Modularity calculation
│   ├── models/                  # Model wrappers and utilities
│   │   ├── wrapper.py          # ModelWrapper for transformations
│   │   └── utils.py            # Model loading utilities
│   └── utils/                   # Utilities and profiling
│       ├── profiler.py         # Hardware profiling (L2 cache, latency)
│       ├── evaluation.py       # Task-specific metrics
│       ├── config.py           # Configuration management
│       └── logging.py          # Logging utilities
├── configs/                     # Hydra configuration files
│   ├── model/                  # Model configurations (bert_base.yaml, mamba_3b.yaml)
│   ├── dataset/                # Dataset configurations (sst2.yaml, wikitext103.yaml)
│   ├── config.yaml             # Main configuration
│   └── defaults.yaml           # Default settings
├── scripts/                     # Experiment and utility scripts
│   ├── run_experiment.py       # Main experiment runner
│   ├── run_quant_test.py       # Quantization experiments
│   ├── generate_all_figures.py # Paper figure generation
│   └── install_mamba.sh        # Mamba installation script
├── tests/                       # Comprehensive test suite
├── notebooks/                   # Jupyter analysis notebooks
├── outputs/                     # Experiment results (auto-generated)
└── results/                     # Cached profiling results
```

## Hardware Profiling & Performance

This project includes advanced **hardware profiling capabilities** for measuring real-world performance improvements:

### GPU Profiling with NVIDIA Nsight Compute

The profiler (`src/utils/profiler.py`) provides:

- **L2 Cache Hit Rate Measurement**: Uses NVIDIA Nsight Compute (NCU) to measure L2 texture cache hit rates
- **Latency Profiling**: Precise GPU/CPU latency measurement using CUDA events
- **Memory Usage Tracking**: Peak and delta memory consumption analysis
- **Deterministic Caching**: Model-hash based result caching for reproducible measurements

### Recent Profiler Improvements

✅ **Timeout Resolution**: Reduced NCU timeout from 180s to 60s with optimized metric collection  
✅ **Targeted Metrics**: Focus on L2 cache metrics (`lts__t_sectors_hit_rate.pct`) for faster profiling  
✅ **Fallback Handling**: Automatic miss-rate to hit-rate conversion and typical values for failed measurements  
✅ **Enhanced Parsing**: Robust CSV output parsing with multiple metric format support  
✅ **Error Recovery**: Graceful handling of compilation failures and CUDA environment issues  

### Hardware Requirements for Full Profiling

**Minimum Requirements:**
- NVIDIA GPU (Turing, Ampere, or newer recommended)
- CUDA Toolkit 11.8+ or 12.x
- NVIDIA Nsight Compute (ncu) in PATH
- `sudo` access for GPU profiling (Linux/Docker environments)

**Optimal Setup:**
- NVIDIA A100, A6000, or RTX 3090/4090
- CUDA 12.1+ with Nsight Compute 2023.3+
- 16GB+ GPU memory for large models
- Fast NVMe storage for dataset caching

### Performance Baseline Expectations

**BERT-base on A100 (reference values):**
- **Latency**: ~2-5ms per inference (batch_size=1, seq_len=512)
- **L2 Cache Hit Rate**: 75-85% (dense model), 70-80% (post-optimization)
- **Memory Usage**: ~1.5GB GPU memory

**Performance Improvements:**
- **Permutation-only**: 10-25% latency reduction
- **Sparsity-only**: 15-30% speedup (depending on sparsity ratio)
- **Iterative Co-Design**: 25-40% combined improvement
- **Cache Efficiency**: 2-5% hit rate improvement through modularity optimization

### Profiling Without GPU/NCU

For environments without GPU profiling capabilities:

```bash
# Disable GPU profiling via environment variable
export DISABLE_GPU_PROFILING=1
python scripts/run_experiment.py dataset=sst2 method=dense model=bert_base
```

**Fallback Behavior:**
- Uses typical cache hit rate values (75% L2 hit rate)
- CPU-only latency measurement
- Basic memory profiling via psutil
- All optimization algorithms still function normally

## Configuration Options

### Model Configuration (`configs/model/`)

**bert_base.yaml:**
```yaml
name: "bert-base-uncased"
hidden_size: 768
vocab_size: 30522
task: "sequence_classification"
iasp:
  target_layer_name: "bert.encoder.layers.0.attention.self.query"
  cluster_size_range: [16, 96]
```

**mamba_3b.yaml:**
```yaml
name: "state-spaces/mamba-2.8b-hf"
hidden_size: 2560
vocab_size: 50277
task: "language_modeling"
iasp:
  target_layer_name: "model.backbone.layers.0.mixer.x_proj"
  cluster_size_range: [16, 128]
```

### Dataset Configuration (`configs/dataset/`)

**Key Parameters:**
- `sample_size`: Number of validation samples to use (default: 1000)
- `batch_size`: Inference batch size (default: 8)
- `text_column`: Dataset column containing text ("text", "sentence", etc.)

### HDS (Sparsity) Configuration (`configs/defaults.yaml`)

```yaml
hds:
  target_layers: ["*.query", "*.key", "*.value", "*.dense", "*linear*"]
  n: 2                    # Keep 2 out of every M weights (N:M sparsity)
  m: 4                    # Block size (2:4 = 50% sparsity)
  fine_tuning_epochs: 1   # Fine-tuning iterations
  gumbel_temp: 1.0       # Gumbel sampling temperature
```

### Experiment Configuration

**Available Settings:**
- `method`: Optimization method (dense, sparsity_only, permute_only, linear_pipeline, iterative)
- `num_iterations`: Number of co-design iterations (default: 3)
- `seed`: Random seed for reproducibility (default: 42)
- `dry_run`: Show planned operations without execution

**W&B Integration:**
```yaml
wandb:
  mode: "offline"  # or "online", "disabled"
  project: "iterative-co-design"
```

## Performance Metrics & Results Interpretation

### Output Files Structure (`outputs/YYYY-MM-DD/HH-MM-SS/`)

**Main Results (`{method}_metrics.json`):**
```json
{
  "perplexity": 15.23,           // Lower = better (language modeling)
  "accuracy": 0.85,              // Higher = better (classification)
  "latency_ms": 4.2,            // Lower = better
  "l2_cache_hit_rate_pct": 78.5, // Higher = better (cache efficiency)
  "modularity": 0.42,           // Higher = better (memory locality)
  "memory_delta_mb": 1.8        // GPU memory usage
}
```

**Configuration (`config.yaml`):**
- Complete Hydra configuration used for the run
- Model, dataset, and optimization parameters
- Reproducible experiment settings

### Metric Interpretation Guide

**Latency (ms):**
- **Excellent**: <2ms (highly optimized)
- **Good**: 2-5ms (well optimized)
- **Baseline**: 5-10ms (dense model)
- **Poor**: >10ms (needs optimization)

**L2 Cache Hit Rate (%):**
- **Excellent**: >85% (optimal memory access patterns)
- **Good**: 75-85% (efficient caching)
- **Typical**: 65-75% (baseline performance)
- **Poor**: <65% (inefficient memory access)

**Modularity Score:**
- **Range**: 0.0 to 1.0 (higher = more modular)
- **Dense baseline**: ~0.0 (no structure)
- **Optimized**: 0.3-0.6 (good clustering)
- **Interpretation**: Measures how well the permutation groups related neurons together

### Statistical Significance

**Measurement Reliability:**
- Latency: Average of 100 runs with warmup
- Cache rates: Hardware counters (highly accurate)
- Task metrics: Full validation set evaluation
- Memory: Peak allocation tracking

**Expected Variance:**
- Latency: ±5% (thermal and system load effects)
- Cache rates: ±2% (measurement precision)
- Task metrics: ±1% (dataset sampling effects)

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

---

## Troubleshooting

### Installation Issues

#### 🔧 Build Backend Errors

```bash
ERROR: Project has a 'pyproject.toml' and its build backend is missing the 'build_editable' hook
```

**Solutions:**
1. **Use legacy setup.py**: Project includes `setup.py` for compatibility
2. **Upgrade setuptools**: `pip install --upgrade setuptools>=64.0.0`
3. **Non-editable install**: `pip install .` instead of `pip install -e .`
4. **Modern backend**: Uses Hatchling for PEP 660 compliance

#### 🐍 Python Development Headers Missing

```bash
fatal error: Python.h: No such file or directory
```

**Solutions (Linux/Ubuntu):**
```bash
sudo apt-get update && sudo apt-get install -y python3-dev python3.10-dev build-essential
```

**Alternative (no sudo):**
```bash
export DISABLE_GPU_PROFILING=1  # Skip compilation-heavy dependencies
pip install -e .  # Install without Mamba support
```

#### 🔗 CUDA/Mamba Compilation Issues

```bash
ImportError: undefined symbol: _ZN3c104cuda9SetDeviceEi
```

**Root Cause**: PyTorch version mismatch with mamba-ssm compiled extensions

**Solutions:**
1. **Use Docker** (recommended): `docker compose run mamba-trainer`
2. **CPU mode**: `export MAMBA_FORCE_CPU=1`
3. **Compatible PyTorch**: 
   ```bash
   pip uninstall torch torchvision torchaudio -y
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
   ```
4. **BERT models only**: Use `model=bert_base` instead of `model=mamba_3b`

### Runtime Issues

#### 🚫 Permission Denied (GPU Profiling)

```bash
NCU profiling failed: Permission denied
```

**Solutions:**
```bash
# Option 1: Run with sudo (Docker environments)
sudo -E python scripts/run_experiment.py dataset=sst2 method=dense model=bert_base

# Option 2: Disable GPU profiling
export DISABLE_GPU_PROFILING=1
python scripts/run_experiment.py dataset=sst2 method=dense model=bert_base

# Option 3: Docker environment (pre-configured)
docker compose run trainer
```

#### ⏱️ GPU Profiling Timeout

```bash
WARNING: GPU profiling timed out.
```

**Recent Fixes Applied:**
- ✅ Reduced timeout from 180s to 60s
- ✅ Optimized metric collection for L2 cache only
- ✅ Simplified profiling script for faster execution

**If still occurring:**
```bash
export DISABLE_GPU_PROFILING=1  # Use fallback values
```

#### 📊 Dataset Download Issues

```bash
FileNotFoundError: Dataset 'wikitext103' not found locally
```

**Solutions:**
```bash
# Manual download
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-raw-v1')"
python -c "from datasets import load_dataset; load_dataset('glue', 'sst2')"

# Or use download script
bash data/download_datasets.sh --install-aria2
```

#### 🧠 Out of Memory Issues

```bash
CUDA out of memory
```

**Solutions:**
1. **Reduce batch size**: Edit `configs/dataset/your_dataset.yaml`
   ```yaml
   batch_size: 1  # Reduce from default 8
   sample_size: 100  # Reduce dataset size for testing
   ```

2. **Use CPU mode**:
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   python scripts/run_experiment.py dataset=sst2 method=dense model=bert_base
   ```

3. **Smaller model**: Use `model=bert_base` instead of `model=mamba_3b`

### Environment Issues

#### 🔄 Import Path Problems

```bash
ModuleNotFoundError: No module named 'src'
```

**Solution**: Install in development mode
```bash
pip install -e .  # Creates proper import paths
```

#### 🐳 Docker Issues

**Docker Build Failures:**
```bash
# Clean build (remove cached layers)
docker system prune -f
docker compose build --no-cache base

# Use pre-built image
docker pull mrcha033/iterative-co-design:latest
docker compose -f docker-compose.hub.yml run base
```

**GPU Access in Docker:**
```bash
# Verify NVIDIA Docker
docker run --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# If fails, install nvidia-container-toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Performance Issues

#### 🐌 Slow Experiment Execution

**Quick Testing:**
```bash
# Use dry run to verify setup
python scripts/run_experiment.py dataset=sst2 method=dense model=bert_base dry_run=true

# Minimal dataset for testing
python scripts/run_experiment.py dataset=sst2 method=dense model=bert_base experiment=debug
```

**Optimization:**
- **Disable W&B**: Set `wandb.mode: "disabled"` in config
- **Reduce sample size**: Edit dataset config `sample_size: 100`
- **Skip profiling**: `export DISABLE_GPU_PROFILING=1`

#### 📈 Unexpected Performance Results

**Verification Steps:**
1. **Check baseline**: Run `method=dense` first for reference
2. **Verify GPU usage**: `nvidia-smi` during execution
3. **Compare cache hit rates**: Should be 70-85% for typical models
4. **Check modularity**: Should increase from ~0.0 (dense) to >0.3 (optimized)

### Getting Help

#### 📝 Reporting Issues

When reporting problems, include:

```bash
# System information
python --version
pip list | grep -E "(torch|transformers|datasets)"
nvidia-smi  # If applicable

# Error reproduction
python scripts/run_experiment.py dataset=sst2 method=dense model=bert_base dry_run=true 2>&1 | tee error.log
```

#### 🔧 Debug Mode

Enable verbose logging:
```bash
export PYTHONPATH=$(pwd)
python -v scripts/run_experiment.py dataset=sst2 method=dense model=bert_base 2>&1 | tee debug.log
```

#### 📚 Additional Resources

- **Docker Guide**: [DOCKER_GUIDE.md](DOCKER_GUIDE.md)
- **Dataset Licenses**: [data/LICENSES.md](data/LICENSES.md)
- **Test Suite**: Run `pytest -v` for comprehensive verification
- **Paper**: See citation for technical details and expected performance ranges

---

## Citation & Contributing

If you use this code in your research, please cite our paper:

```bibtex
@article{cha2024orthogonality,
  title={The Orthogonality Fallacy: Iterative Co-Design as a First-Class Principle for Efficient AI},
  author={Cha, Yunmin and others},
  journal={arXiv preprint},
  year={2024}
}
```

**Contributions Welcome:**
- 🐛 Bug reports and fixes
- 📈 Performance improvements
- 🔧 New optimization algorithms
- 📚 Documentation improvements
- 🧪 Additional model support

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
