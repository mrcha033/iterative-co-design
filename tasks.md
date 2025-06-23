tasks.md: A Step-by-Step Implementation Plan

This document breaks down the implementation of the "Iterative Co-Design" project into granular, verifiable tasks. Each task is designed to be a single, indivisible unit of work with a clear verification step.

Phase 1: Foundational Setup & Utilities

This phase establishes the project backbone, including data handling, configuration management, and core measurement tools.

[x] Task 1.1: Initialize Project Structure

Description: Create the complete directory structure as outlined in the project proposal (configs/, data/, src/, etc.). Initialize a Git repository.

Verification: All directories exist. git status shows a clean initial commit.

[x] Task 1.2: Setup Dependency Management

Description: Create a requirements.txt file and add initial core dependencies: torch, transformers, datasets, pyyaml, numpy, scikit-learn, pandas.

Verification: Run pip install -r requirements.txt in a clean virtual environment. The command completes successfully.

[x] Task 1.3: Implement Configuration Loader

Description: Create a utility function (src/utils/config.py) that loads a specified .yaml file and returns a configuration object (e.g., a dictionary or an argparse.Namespace).

Verification: Write a unit test that loads a dummy test.yaml and asserts that the returned object contains the correct keys and values.

[x] Task 1.4: Create Data Loading Script

Description: Implement data/download_datasets.sh. This script will use the Hugging Face datasets library to download and cache 'wikitext-103-raw-v1' and 'sst2'.

Verification: Run the script. Check that the datasets are downloaded to the default Hugging Face cache directory (~/.cache/huggingface/datasets).

[x] Task 1.5: Implement Latency Profiler

Description: Implement the measure_latency() function in src/utils/profiler.py. It should take a model and dummy input, perform warm-up runs, and then average the wall-clock time over N subsequent runs. Use torch.cuda.synchronize() for accurate GPU timing.

Verification: Test with a pre-trained bert-base-uncased. The function should return a positive float value (e.g., 15.2). Running it twice should yield similar, stable results.

[x] Task 1.6: Implement L2 Cache Hit Rate Profiler

Description: Implement measure_cache_hits() in src/utils/profiler.py. This function will construct and execute an ncu command line string using Python's subprocess module to profile a model's forward pass. It must parse the text output from ncu to extract the l2_tex_hit_rate.pct metric.

Verification: Manually run ncu --metrics l2_tex_hit_rate.pct python my_simple_model_run.py on a tiny script. Then run the measure_cache_hits() function on the same script. Assert that the parsed value from the function matches the value from the manual run. This is a critical and potentially tricky task.

[x] Task 1.7: Implement Model Evaluation Utilities

Description: Implement calculate_perplexity() and calculate_accuracy() in src/utils/evaluation.py for Mamba and BERT respectively.

Verification: Load a pre-trained Mamba and BERT model. Run the evaluation functions on their respective validation sets. The results should be reasonable and match known baseline scores for those models.

Phase 2: Core Co-Design Algorithm Implementation

This phase focuses on implementing the novel components of the paper: IASP and HDS.

[x] Task 2.1: Implement Model Wrapper and Permutation Logic

Description: Create the ModelWrapper in src/models/wrapper.py. Implement the permute_model_weights(permutation) method. This method will take a permutation array (e.g., [0, 2, 1, 3]) and reorder the rows and columns of the relevant weight matrices in the model's state_dict.

Verification:

Create a tiny 4x4 linear layer.

Apply a permutation [0, 2, 1, 3].

Manually check that the second row of the original weight matrix is now the third row, and the third row is now the second. Do the same for columns.

Assert that torch.equal() confirms the manual and programmatic permutations are identical.

[x] Task 2.2: Implement Activation Correlation Matrix Calculation

Description: In src/co_design/iasp.py, implement get_activation_correlation(). This function should perform forward passes on a sample of data, hook into a target layer to extract activations, and compute the Pearson correlation matrix of the activation dimensions.

Verification: Use a simple 2-layer MLP. Feed it perfectly correlated input data for two dimensions (e.g., dimension i is always 2 * j). The resulting correlation matrix entry C[i, j] should be very close to 1.0. For uncorrelated (random) input, it should be close to 0.0.

[x] Task 2.3: Implement Modularity Calculation

Description: Implement the calculate_modularity() function in src/co_design/modularity.py. It takes a correlation matrix C and a proposed clustering (or partition) of nodes and computes the modularity score as defined by Newman.

Verification: Create a toy 4x4 correlation matrix representing two perfect clusters (e.g., C[0,1] and C[2,3] are high, all others are low). Test two partitions: [[0,1], [2,3]] and [[0,2], [1,3]]. Assert that the modularity score for the first (correct) partition is significantly higher.

[x] Task 2.4: Implement IASP (IO-Aware Scan Permutation)

Description: Implement find_optimal_permutation() in src/co_design/iasp.py. This function orchestrates the process:

Calls get_activation_correlation() to get matrix C.

Uses sklearn.cluster.SpectralClustering on C to find optimal clusters.

Constructs a final permutation array by concatenating the indices within each cluster.

Verification: Use the same toy 4x4 correlation matrix from Task 2.3. The function should return a permutation that groups the clustered nodes together, e.g., [0, 1, 2, 3] or [2, 3, 0, 1]. The modularity score for this output permutation should be maximal.

[x] Task 2.5: Implement HDS (Hardware-Native Differentiable Sparsity)

Description: In src/co_design/hds.py, implement a layer or hook that applies the Gumbel-Top-K trick to learn a 2:4 structured sparsity mask during a model's fine-tuning phase.

Verification: Create a small model and run a few training steps with an HDS layer. After training, inspect the learned sparsity mask. It should have exactly two non-zero values for every block of four values, and these values should be derived from the underlying continuous parameters.

Phase 3: Experiment Execution and Analysis

This phase brings all the components together to run the full experiments.

[x] Task 3.1: Implement the Main Experiment Runner

Description: Build scripts/run_experiment.py. It should parse command-line arguments (--config, --method), load the specified configuration, and call the appropriate sequence of functions from src to execute one of the five experimental conditions (Dense, Sparsity-Only, etc.).

Verification: Run a "dry run" for each method: python scripts/run_experiment.py --config configs/mamba_3b_wikitext103.yaml --method <method_name> --dry_run. The script should print the sequence of operations it would perform without actually running them (e.g., "1. Loading model. 2. Applying HDS. 3. Measuring latency...").

[x] Task 3.2: Run Baseline Experiments (Dense, Permute-Only)

Description: Execute run_experiment.py for the dense and permute_only methods on the Mamba-3B model.

Verification: The script should complete without errors and generate result files (e.g., results/mamba_3b/dense_metrics.json). The JSON file should contain all measured metrics: perplexity, latency, l2_cache_hit_rate, and modularity. The values should be reasonable.

[x] Task 3.3: Run Sparsity and Linear Pipeline Experiments

Description: Execute run_experiment.py for the sparsity_only and linear_pipeline methods. This involves fine-tuning with HDS.

Verification: The fine-tuning process should show a decreasing loss. The final result files should be generated correctly. We expect latency for linear_pipeline to be lower than for the individual baselines.

[x] Task 3.4: Run the Full Iterative Co-Design Experiment

Description: Execute run_experiment.py for the iterative method. This is the core experiment.

Verification: The script should successfully complete multiple iterations of HDS -> IASP. The log output should show that the modularity score and L2 cache hit rate increase with iterations, while latency decreases, directly verifying the paper's central claim. A final iterative_metrics.json file is produced.

[x] Task 3.5: Implement and Run the Quantization Experiment

Description: Build scripts/run_quant_test.py. Implement the three strategies (Quant-then-Permute, Permute-then-Quant, Permute-Quant-RePermute) using a standard PTQ library (like PyTorch's native quantization).

Verification: Run all three methods. The results should show that the Permute-Quant-RePermute method achieves the lowest latency, demonstrating the value of the final "re-permutation" step, as highlighted by the red arrow in Figure 2.

[x] Task 3.6: Create Analysis Notebook

Description: Create notebooks/2_analyze_results.ipynb. Write code to load all generated JSON files into a single Pandas DataFrame.

Verification: The notebook should correctly display a summary table that looks like Table 1. It should also generate plots that visually replicate Figure 3 (metrics vs. iteration) and Figure 4 (Pareto frontier). The final plots should qualitatively match the conclusions of the paper.

## Phase 4: Figure Generation and Visualization ??
This phase focuses on creating publication-quality figures that replicate the paper's key visualizations.

[x] Task 4.1: Implement Comprehensive Figure Generation Suite

Description: Create a unified scripts/generate_all_figures.py that handles all paper figures:
- **Figure 1**: Random vs. Optimized Permutation Latency (20 random permutations + IASP optimization)
- **Figure 2**: Quantization Co-Design Strategies comparison
- **Figure 3**: Metrics vs. Iteration (causal chain visualization)  
- **Figure 4**: Pareto Frontier (all methods comparison)
- Support for individual figure generation (--figure N)
- Quick mode for testing (--quick)
- Publication-quality PDF and PNG outputs

Verification: Run the script and verify it produces all figures showing expected improvements (~25-35% latency gains, iterative outperforming linear approaches).

[x] Task 4.2: Integrate Interactive Figure Generation

Description: Update notebooks/1_explore_correlation.ipynb to include interactive figure generation functionality for all paper figures.

Verification: Open the notebook and run all cells. It should successfully generate all figures and save them to the figures/ directory.

[x] Task 4.3: Update Documentation and Usage

Description: Update README.md, structure.md, and tasks.md to reflect the unified figure generation system with clear usage examples.

Verification: Documentation shows comprehensive usage examples for all figure generation modes (all figures, specific figure, quick mode, interactive).

