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

Install the required Python packages:

```bash
pip install -r requirements.txt
```
*Note: The experiments may require a recent version of PyTorch (`>=2.3.0`) and other libraries. If you encounter issues, consider upgrading the packages.*

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

---

## Running Experiments

All experiments are orchestrated through the main runner script, `scripts/run_experiment.py`.

### Basic Usage

The script takes two main arguments:
- `--config`: The path to the experiment configuration file (e.g., `configs/model/mamba_3b.yaml`).
- `--method`: The experimental condition to run.

### Available Methods

- `dense`: (Baseline 1) The original, unmodified model.
- `permute_only`: (Baseline 3) Applies IASP to the dense model.
- `sparsity_only`: (Baseline 2) Applies HDS to the dense model.
- `linear_pipeline`: (Baseline 4) Applies IASP, then HDS.
- `iterative`: (Ours) Applies HDS and IASP in a loop.

### Example Command

To run the "dense" baseline experiment with the Mamba-3B configuration:

```bash
python scripts/run_experiment.py --config configs/model/mamba_3b.yaml --method dense
```

### Model Configuration

Model YAML files under `configs/model` include an `iasp` section that controls
the permutation search. The key `cluster_size_range` defines the minimum and
maximum cluster sizes (in neurons) considered when searching for an optimal
permutation.

### Dry Run

To see the sequence of operations for a method without executing the full, expensive run, use the `--dry_run` flag:

```bash
python scripts/run_experiment.py --config configs/model/mamba_3b.yaml --method iterative --dry_run
```

### Results

All results, including measured metrics (perplexity, latency, modularity, etc.) and any generated artifacts, are saved as JSON files in the `results/` directory, organized by model name. 

### Testing

To run the unit tests, first install the additional test dependencies and then execute `pytest`:

```bash
pip install -r requirements.txt -r tests/requirements.txt
pytest
```

Alternatively, you can use the helper script:

```bash
./scripts/run_tests.sh
```
