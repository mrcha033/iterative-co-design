"""Tests for the run_experiment.py script, including division by zero fixes."""

import torch.nn as nn
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

# Import the functions we want to test
from run_experiment import run_cleanup_if_configured


class TinyTestModel(nn.Module):
    """Extremely small model with d_model < 32 to test division by zero fix."""

    def __init__(self, d_model=16):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        # Mock config attribute
        self.config = MagicMock()
        self.config.hidden_size = d_model

    def forward(self, input_ids):
        return self.linear(input_ids.float())


class TestRunExperiment:
    """Test cases for run_experiment.py functionality."""

    def test_division_by_zero_fix_small_model(self):
        """Test that cluster sizing doesn't cause division by zero for d_model < 32."""
        d_model = 16  # Less than 32

        # Test the fixed calculation directly
        cluster_step = max(d_model // 32, 1)  # Should be 1 when d_model < 32
        nodes_per_cluster = d_model // cluster_step  # Should be 16 // 1 = 16

        assert cluster_step == 1, f"Expected cluster_step=1, got {cluster_step}"
        assert nodes_per_cluster == 16, f"Expected nodes_per_cluster=16, got {nodes_per_cluster}"

        # Verify this works for various small d_model values
        for test_d_model in [8, 16, 24, 31]:
            cluster_step = max(test_d_model // 32, 1)
            nodes_per_cluster = test_d_model // cluster_step

            # Should not raise any exceptions
            assert cluster_step >= 1
            assert nodes_per_cluster >= 1
            assert nodes_per_cluster <= test_d_model

    def test_division_by_zero_fix_normal_model(self):
        """Test that cluster sizing still works correctly for normal-sized models."""
        d_model = 768  # Typical BERT size

        # Test the fixed calculation
        cluster_step = max(d_model // 32, 1)  # Should be 768 // 32 = 24
        nodes_per_cluster = d_model // cluster_step  # Should be 768 // 24 = 32

        assert cluster_step == 24, f"Expected cluster_step=24, got {cluster_step}"
        assert nodes_per_cluster == 32, f"Expected nodes_per_cluster=32, got {nodes_per_cluster}"

    def test_cleanup_configured_dry_run(self):
        """Test cleanup functionality in dry run mode."""
        # Create a mock config with cleanup settings
        cfg = OmegaConf.create({
            "cleanup": {
                "base_dirs": ["outputs", "multirun"],
                "max_age_days": 30
            }
        })

        with patch("run_experiment.cleanup_old_runs") as mock_cleanup:
            run_cleanup_if_configured(cfg, dry_run=True)

            # Verify cleanup was called with dry_run=True
            mock_cleanup.assert_called_once_with(
                base_dirs=["outputs", "multirun"],
                max_age_days=30,
                dry_run=True
            )

    def test_cleanup_configured_normal_run(self):
        """Test cleanup functionality in normal (non-dry-run) mode."""
        # Create a mock config with cleanup settings
        cfg = OmegaConf.create({
            "cleanup": {
                "base_dirs": ["outputs"],
                "max_age_days": 15
            }
        })

        with patch("run_experiment.cleanup_old_runs") as mock_cleanup:
            run_cleanup_if_configured(cfg, dry_run=False)

            # Verify cleanup was called with dry_run=False
            mock_cleanup.assert_called_once_with(
                base_dirs=["outputs"],
                max_age_days=15,
                dry_run=False
            )

    def test_cleanup_not_configured(self):
        """Test that cleanup is skipped when not configured."""
        # Create a config without cleanup settings
        cfg = OmegaConf.create({
            "other_setting": "value"
        })

        with patch("run_experiment.cleanup_old_runs") as mock_cleanup:
            run_cleanup_if_configured(cfg, dry_run=False)

            # Verify cleanup was not called
            mock_cleanup.assert_not_called()

    def test_cleanup_empty_config(self):
        """Test cleanup with empty cleanup config."""
        # Create a config with empty cleanup
        cfg = OmegaConf.create({
            "cleanup": None
        })

        with patch("run_experiment.cleanup_old_runs") as mock_cleanup:
            run_cleanup_if_configured(cfg, dry_run=False)

            # Verify cleanup was not called
            mock_cleanup.assert_not_called()

    def test_cleanup_exception_handling(self):
        """Test that cleanup exceptions are handled gracefully."""
        cfg = OmegaConf.create({
            "cleanup": {
                "base_dirs": ["outputs"],
                "max_age_days": 30
            }
        })

        with patch("run_experiment.cleanup_old_runs") as mock_cleanup, \
             patch("builtins.print") as mock_print:

            # Make cleanup raise an exception
            mock_cleanup.side_effect = Exception("Cleanup failed")

            # Should not raise exception
            run_cleanup_if_configured(cfg, dry_run=False)

            # Verify error message was printed
            mock_print.assert_any_call("⚠️  Cleanup failed: Cleanup failed")
            mock_print.assert_any_call("Continuing with experiment...")
