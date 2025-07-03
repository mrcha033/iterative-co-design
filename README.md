# Iterative Co-Design of Sparsity and Permutation

This repository contains the official implementation for the paper, **"The Orthogonality Fallacy: Iterative Co-Design as a First-Class Principle for Efficient AI."** This work dismantles the assumption that algorithmic optimizations (like sparsity) and hardware-level optimizations (like memory layout) are separable problems. It introduces an **Iterative Co-Design** framework that alternates between algorithmic state changes and memory layout optimization to find a more efficient Pareto-optimal model.

---

## 🚀 Key Features

- **Production-Ready IASP**: Robust permutation system with automatic rollback and safety monitoring
- **Hardware-Native Sparsity (HDS)**: N:M structured sparsity that maps directly to tensor cores
- **Multi-Objective Optimization**: Pareto-optimal tradeoffs between accuracy, latency, and memory
- **Distributed Training Support**: Full compatibility with DDP, FSDP, and mixed precision
- **Comprehensive Safety**: Layer-type registry, optimizer state tracking, and perplexity watchdogs

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mrcha033/iterative-co-design.git
cd iterative-co-design
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create and activate the virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

For basic installation with core features:
```bash
pip install -e .
```

For full development setup with all safety features:
```bash
pip install -e ".[dev]"
```

For production deployment with advanced IASP clustering:
```bash
pip install -e ".[iasp]"
```

**Note:** For Mamba models, you'll need `mamba-ssm` which requires CUDA. Use Docker for the easiest setup.

### 4. (Optional) Docker for Reproducibility

For guaranteed reproducibility and to avoid local compilation issues (especially with Mamba), we strongly recommend using the provided Docker environment.

```bash
# Build and run the development environment
docker-compose up --build
```

This will set up a container with all dependencies pre-installed and configured.

---

## Usage

All experiments are run through the `scripts/run_experiment.py` script, which is configured using [Hydra](https://hydra.cc/).

### Example Commands

Here are a few examples of how to run the different co-design methods.

**Run the `dense` baseline with BERT on the SST-2 dataset:**
```bash
python scripts/run_experiment.py model=bert_base dataset=sst2 method=dense
```

**Run the `iterative` co-design method with Mamba on the WikiText-103 dataset:**
```bash
python scripts/run_experiment.py model=mamba_370m dataset=wikitext103 method=iterative \
    iasp.max_ppl_increase=0.03  # 3% safety threshold
```

**Perform a "dry run" to see the experiment plan without executing:**
```bash
python scripts/run_experiment.py model=bert_base dataset=sst2 method=iterative dry_run=true
```

### Available Methods

- `dense`: The original, unmodified model.
- `sparsity_only`: Applies HDS (sparsity) to the dense model.
- `permute_only`: Applies IASP (permutation) to the dense model.
- `linear_pipeline`: Applies IASP, then HDS.
- `iterative`: Applies HDS and IASP in a feedback loop.

### Production Safety Features

The IASP system now includes multiple safety layers:

```python
# Example: Using IASP with production safety
from src.co_design.iasp_rollback import create_safe_iasp_wrapper

result = create_safe_iasp_wrapper(
    run_iasp_on_mamba,
    model=model,
    dataloader=train_dl,
    eval_dataloader=val_dl,
    config=iasp_config,
    max_ppl_increase=0.05  # 5% perplexity tolerance
)
```

---

## Project Structure

```
iterative-co-design/
├── src/                          # Source code
│   ├── co_design/               # Core optimization algorithms
│   │   ├── hds.py              # Hardware-Native Differentiable Sparsity
│   │   ├── iasp.py             # IO-Aware Scan Permutation
│   │   ├── iasp_rollback.py    # Safety rollback system
│   │   ├── pareto_watchdog.py  # Multi-objective monitoring
│   │   └── modularity.py       # Modularity calculation
│   ├── models/                  # Model wrappers and utilities
│   └── utils/                   # Utilities and profiling
├── configs/                     # Hydra configuration files
├── scripts/                     # Experiment and utility scripts
├── tests/                       # Comprehensive test suite
│   ├── test_iasp_safety.py     # Production safety tests
│   ├── test_distributed_iasp.py # Multi-GPU tests
│   └── test_permutation_training.py # Training continuity
├── notebooks/                   # Jupyter analysis notebooks
├── outputs/                     # Experiment results (auto-generated)
└── results/                     # Cached profiling results
```

---
## Testing

The project includes a comprehensive test suite using `pytest`.

To run all tests:
```bash
pytest
```

To run tests for a specific file:
```bash
pytest tests/test_hds.py
```

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{cha2024orthogonality,
  title={The Orthogonality Fallacy: Iterative Co-Design as a First-Class Principle for Efficient AI},
  author={Cha, Yunmin and others},
  journal={arXiv preprint},
  year={2024}
}
```

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.