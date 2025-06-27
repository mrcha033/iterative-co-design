# Configuring Experiments

This project uses [Hydra](https://hydra.cc/) to manage complex configurations for experiments. This guide explains the structure of the configuration files and how to customize them for your own use cases.

## Configuration Structure

The main configuration is located in the `configs/` directory. Here's an overview of the key files:

- **`config.yaml`**: The main entry point for configuration. It sets the defaults for the model, dataset, and method, and configures Hydra's behavior.
- **`defaults.yaml`**: Contains the default parameters for the core algorithms (`hds`, `iasp`), as well as for logging (`wandb`) and storage.
- **`model/`**: Contains YAML files for each supported model (e.g., `bert_base.yaml`, `mamba_370m.yaml`). These files specify the model's name, task, and any model-specific parameters for HDS and IASP.
- **`dataset/`**: Contains YAML files for each dataset (e.g., `sst2.yaml`, `wikitext103.yaml`), specifying the dataset name, path, and batch size.

## How to Run a Custom Experiment

To run an experiment with a specific configuration, you can override the defaults from the command line.

For example, to run the `iterative` method with the `bert_base` model on the `sst2` dataset, you would run:

```bash
python scripts/run_experiment.py model=bert_base dataset=sst2 method=iterative
```

Hydra automatically composes the final configuration by merging the defaults with your command-line overrides.

## Key Configuration Parameters

Here are some of the most important parameters you might want to change:

### In `config.yaml`:

- `method`: The co-design method to run.
  - **Options:** `dense`, `sparsity_only`, `permute_only`, `linear_pipeline`, `iterative`.
- `seed`: The random seed for reproducibility.
- `num_iterations`: The number of iterations for the `iterative` method.

### In `defaults.yaml`:

- **`hds`**:
  - `target_layers`: A list of wildcard patterns for which layers to apply HDS to.
  - `n`, `m`: The N:M ratio for structured sparsity (e.g., 2:4 for 50% sparsity).
  - `fine_tuning_epochs`: The number of epochs to fine-tune the sparsity masks.
- **`iasp`**:
  - `cluster_size_range`: The range of cluster sizes to search for the optimal permutation.
  - `max_samples`: The number of activation samples to collect for correlation analysis.
  - `spectral_n_init`, `spectral_random_state`: Parameters for the `SpectralClustering` algorithm.

### In `model/*.yaml`:

- `name`: The name of the model on the Hugging Face Hub.
- `task`: The type of task the model is used for.
  - **Options:** `language_modeling`, `sequence_classification`.
- `iasp.target_layer_name`: The specific layer to analyze for activation correlation. This is highly model-specific.

## Creating a New Model Configuration

To add a new model, you can create a new YAML file in the `configs/model/` directory. For example, to add a DistilBERT model, you could create `configs/model/distilbert.yaml`:

```yaml
# configs/model/distilbert.yaml
name: "distilbert-base-uncased"
type: "bert" # Important for dispatching to the correct IASP function
task: "sequence_classification"

hds:
  target_layers:
    - "*.attention.q_lin"
    - "*.attention.k_lin"
    - "*.attention.v_lin"
    - "*.ffn.lin1"
    - "*.ffn.lin2"

iasp:
  target_layer_name: "distilbert.transformer.layer.0.attention.q_lin"
  cluster_size_range: [16, 64]
```

You could then run an experiment with this model:

```bash
python scripts/run_experiment.py model=distilbert dataset=sst2 method=dense
```
