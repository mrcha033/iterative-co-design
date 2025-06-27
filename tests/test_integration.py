"""
Integration tests for the main experiment pipelines in run_experiment.py.

These tests ensure that the different co-design methods (dense, sparse-only, etc.)
can run end-to-end without crashing, using a minimal model and dataset.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add the 'scripts' directory to the path to allow importing run_experiment
scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

# Now we can import the functions from the script
import run_experiment

# --- Test Fixtures ---

@pytest.fixture
def tiny_model():
    """A very small model for testing purposes."""
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(16, 16)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(16, 4)
            # Mock the config attribute that the script expects
            self.config = MagicMock()
            self.config.name = "tiny-test-model"
            self.config.vocab_size = 100
            self.config._name_or_path = "tiny-test-model"
            self.config.to_dict.return_value = {"architectures": ["TestModel"]}


        def forward(self, input_ids, attention_mask=None, labels=None):
            # A dummy forward pass that returns a loss if labels are provided
            out = self.linear2(self.relu(self.linear1(input_ids.float())))
            loss = None
            if labels is not None:
                loss = torch.nn.functional.mse_loss(out, labels.float())
            return MagicMock(loss=loss, logits=out)

    return TestModel()

@pytest.fixture
def tiny_data():
    """A minimal tokenizer, dataloader, and dataset."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "[PAD]"
    
    # Create a dummy dataloader that yields batches with the correct keys
    input_ids = torch.randint(0, 100, (10, 16))
    labels = torch.rand(10, 4)
    dataset = TensorDataset(input_ids, labels)
    
    def collate_fn(batch):
        return {
            "input_ids": batch[0][0].unsqueeze(0),
            "attention_mask": torch.ones_like(batch[0][0].unsqueeze(0)),
            "labels": batch[0][1].unsqueeze(0)
        }

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    eval_dataset = MagicMock() # Not used in the functions, but returned
    
    return tokenizer, dataloader, eval_dataset

@pytest.fixture
def base_config():
    """A base OmegaConf object for testing."""
    return OmegaConf.create({
        "seed": 42,
        "model": {
            "name": "test-model",
            "task": "sequence_classification",
            "vocab_size": 100,
            "hds": {
                "target_layers": ["linear1"],
                "n": 2,
                "m": 4,
                "fine_tuning_epochs": 1,
            },
            "iasp": {
                "cluster_size_range": [2, 8],
            }
        },
        "dataset": {
            "name": "test-dataset",
            "batch_size": 2,
        },
        "method_configs": {
            "iterative": {
                "iterations": 2,
            }
        },
        "wandb": {"log": False},
    })

# --- Mocks for External Dependencies ---

@patch("run_experiment.save_results")
@patch("run_experiment.LatencyProfiler")
@patch("run_experiment.get_model_and_data")
def run_test(
    method_name, mock_get_data, mock_profiler, mock_save,
    tiny_model, tiny_data, base_config
):
    """Helper function to run a test for a given method."""
    # Configure mocks
    mock_get_data.return_value = (tiny_model, tiny_data[0], tiny_data[1], tiny_data[2])
    mock_profiler.return_value.measure_latency.return_value = 10.0
    mock_profiler.return_value.measure_cache_hits.return_value = {"lts__t_sector_hit_rate.pct": 95.0}

    # Get the actual function from the script
    method_to_test = getattr(run_experiment, f"run_{method_name}")
    
    # Run the experiment method
    method_to_test(base_config)

    # Assert that results were saved
    mock_save.assert_called_once()
    
    # Assert that the model was loaded
    mock_get_data.assert_called_once()

# --- Tests for each experiment pipeline ---

def test_run_dense(tiny_model, tiny_data, base_config):
    run_test("dense", tiny_model=tiny_model, tiny_data=tiny_data, base_config=base_config)

def test_run_sparsity_only(tiny_model, tiny_data, base_config):
    run_test("sparsity_only", tiny_model=tiny_model, tiny_data=tiny_data, base_config=base_config)

def test_run_permute_only(tiny_model, tiny_data, base_config):
    # IASP can be slow, so we mock it to just return a dummy score
    with patch("run_experiment._run_iasp") as mock_run_iasp:
        mock_run_iasp.return_value = 0.5 # Dummy modularity score
        run_test("permute_only", tiny_model=tiny_model, tiny_data=tiny_data, base_config=base_config)
        mock_run_iasp.assert_called_once()

def test_run_linear_pipeline(tiny_model, tiny_data, base_config):
    with patch("run_experiment._run_iasp") as mock_run_iasp:
        mock_run_iasp.return_value = 0.5
        run_test("linear_pipeline", tiny_model=tiny_model, tiny_data=tiny_data, base_config=base_config)
        mock_run_iasp.assert_called_once()

def test_run_iterative(tiny_model, tiny_data, base_config):
    with patch("run_experiment._run_iasp") as mock_run_iasp:
        mock_run_iasp.return_value = 0.5
        run_test("iterative", tiny_model=tiny_model, tiny_data=tiny_data, base_config=base_config)
        # Should be called once per iteration
        assert mock_run_iasp.call_count == base_config.method_configs.iterative.iterations
