Project Structure for Replicating "The Orthogonality Fallacy"

This document outlines a project structure designed to systematically implement and reproduce the key experiments from the paper, "The Orthogonality Fallacy: Iterative Co-Design as a First-Class Principle for Efficient AI." The structure emphasizes modularity, reproducibility, and extensibility.

1. Top-Level Directory Structure

The overall project layout is as follows:

iterative-co-design/
├── configs/                  # Experiment hyperparameter configuration files
│   ├── mamba_3b_wikitext103.yaml
│   └── bert_base_sst2.yaml
│
├── data/                     # Scripts for downloading and preprocessing datasets
│   └── download_datasets.sh
│
├── notebooks/                # Jupyter notebooks for analysis and visualization
│   ├── 1_explore_correlation.ipynb
│   └── 2_analyze_results.ipynb
│
├── results/                  # Storage for all experiment outputs (logs, checkpoints, metrics)
│   ├── mamba_3b/
│   └── bert_base/
│
├── src/                      # Core logic and reusable source code modules
│   ├── co_design/
│   │   ├── __init__.py
│   │   ├── iasp.py           # IO-Aware Scan Permutation (IASP) implementation
│   │   ├── hds.py            # Hardware-Native Differentiable Sparsity (HDS) implementation
│   │   └── modularity.py     # Modularity calculation logic
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── wrapper.py        # Wrapper class for loading models and applying permutations
│   │   └── utils.py          # Utilities for model state_dict manipulation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── profiler.py       # Utility for measuring Latency and L2 Cache Hit Rate
│       └── evaluation.py     # Utilities for calculating Perplexity, Accuracy, etc.
│
├── scripts/                  # Executable scripts for running experiments
│   ├── run_experiment.py     # Main experiment runner (for Mamba and BERT)
│   └── run_quant_test.py     # Runner for the critical quantization experiment
│
├── README.md                 # Project description and setup instructions
└── requirements.txt          # List of required Python libraries

2. Detailed Description of Directories and Files
📝 configs/

This directory decouples hyperparameters from the code, enhancing reproducibility. YAML format is used for readability.

mamba_3b_wikitext103.yaml: Contains settings for the Mamba-3B model, such as learning rate, epochs, sparsity target, etc., based on Table 4 of the paper.

bert_base_sst2.yaml: Contains settings for the BERT-base model experiments.

📦 data/

Scripts to prepare the necessary datasets.

download_datasets.sh: A shell script that uses the Hugging Face datasets library to download WikiText-103 and SST-2 and save them to a designated path.

🔬 src/ (Source Code)

The core logic of the project resides here.

src/co_design/: Modules for the novel co-design techniques proposed in the paper.

iasp.py: Implements IO-Aware Scan Permutation (IASP).

get_activation_correlation(): A function to compute the correlation matrix C between dimensions based on model activations.

find_optimal_permutation(): The main function that takes the correlation matrix C and uses the logic from modularity.py to find the optimal permutation π*. This should be implemented using spectral clustering as mentioned in the paper.

hds.py: Implements Hardware-Native Differentiable Sparsity (HDS).

Defines a PyTorch nn.Module or related functions to learn an N:M structured sparsity mask using the Gumbel-Top-K reparameterization trick.

modularity.py: A function to calculate the Modularity score of a given permutation. This serves as the objective function that IASP optimizes.

src/models/: Modules for model loading and management.

wrapper.py: A wrapper class for Hugging Face transformers models. It will contain a crucial method, permute_model_weights(), which takes a permutation π from IASP and physically reorders the model's weights in its state_dict.

utils.py: Utility functions for tensor manipulation required for applying permutations.

src/utils/: Other essential utilities.

profiler.py: The core utility for performance measurement.

measure_latency(): Measures the average wall-clock inference time over many runs to get a stable latency (ms) value.

measure_cache_hits(): A critical function that uses a subprocess to call NVIDIA Nsight Compute (ncu), profiles a kernel, and parses the output to extract the L2 Cache Hit Rate. This is essential for proving the paper's "mechanistic link".

evaluation.py: Functions to calculate model performance metrics.

calculate_perplexity(): For the Mamba language model.

calculate_accuracy(): For the BERT classification model.

🚀 scripts/

The entry points for running all experiments via the command line.

run_experiment.py: The main script to reproduce Table 1 and Table 2.

Accepts a --config argument to specify the YAML file.

Accepts a --method argument to select the experiment scenario:

dense: (1) Dense Baseline

sparsity_only: (2) Sparsity-Only (HDS)

permute_only: (3) Permutation-Only (IASP)

linear_pipeline: (4) IASP-then-HDS

iterative: (5) Iterative Co-Design. The core claim of the paper. This mode would orchestrate the feedback loop between HDS and IASP.

run_quant_test.py: A dedicated script to reproduce the quantization experiment from Figure 2.

Accepts a --method argument:

quant_then_permute (Linear Pipeline 1)

permute_then_quant (Linear Pipeline 2)

permute_quant_repermute (Iterative Co-Design)

3. Experimental Workflow Example (Reproducing Table 1)

This structure enables a clear and reproducible workflow.

Setup:

Install dependencies: pip install -r requirements.txt.

Download data: ./data/download_datasets.sh.

Run Baselines (Methods 1-4): Execute the main script with different method flags.

# (1) Dense Baseline
python scripts/run_experiment.py --config configs/mamba_3b_wikitext103.yaml --method dense

# (2) Sparsity-Only
python scripts/run_experiment.py --config configs/mamba_3b_wikitext103.yaml --method sparsity_only

# (3) Permutation-Only & (4) Linear Pipeline
# ... and so on for permute_only and linear_pipeline
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Each run saves its measured metrics (Perplexity, Latency, Modularity, L2 Cache Hit Rate) to a unique file in results/mamba_3b/, e.g., dense_metrics.json.

Run Iterative Co-Design (Method 5):

python scripts/run_experiment.py --config configs/mamba_3b_wikitext103.yaml --method iterative --num_iterations 2
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Internal Logic:

Load the initial model.

Iteration 1:
a. Apply HDS to perturb the algorithmic state (fine-tuning).
b. Run iasp.find_optimal_permutation() on the new state to find π_1.
c. Apply π_1 to the model using models.wrapper.permute_model_weights().

Iteration 2:
a. Apply another round of HDS fine-tuning.
b. Run iasp.find_optimal_permutation() again to find a new permutation π_2.
c. Apply π_2.

Profile the final model state using utils.profiler and utils.evaluation and save the results.

Analysis and Visualization:

Open notebooks/2_analyze_results.ipynb.

Load all the .json files from the results/ directory into a Pandas DataFrame.

Generate a table that directly mirrors Table 1 from the paper.

Use Matplotlib/Seaborn to plot the results, recreating Figure 3 (metrics vs. iteration) and Figure 4 (Pareto Frontier).

This project structure provides a robust framework for validating the paper's claims and serves as a solid foundation for future research in co-design.