"""
Integration tests for the experiment pipeline and CLI.

These tests verify that all experiment strategies execute correctly
and produce expected outputs.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
import torch.nn as nn
import numpy as np

from scripts.run_experiment import ExperimentRunner
from src.utils.config import load_config, merge_config_with_args
from src.models.permutable_model import PermutableModel
from src.utils.exceptions import IterativeCoDesignError


class TestExperimentPipeline:
    """Test the main experiment pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create temporary directory for test outputs
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create minimal test configuration
        self.test_config = {
            'model': {
                'name': 'mamba-3b',
                'hf_model_id': 'state-spaces/mamba-3b',
                'pretrained_path': None,
                'precision': 'float16'
            },
            'dataset': {
                'name': 'wikitext-103',
                'path': './data/',
                'sequence_length': 512,  # Smaller for testing
                'batch_size': 1,
                'num_samples': 10,  # Much smaller for testing
                'num_benchmark_samples': 5
            },
            'iasp': {
                'layer_name': 'layers.0.mixer',
                'num_clusters': 4,  # Smaller for testing
                'correlation_threshold': 0.1,
                'method': 'spectral',
                'precomputed_path': './data/correlation_matrices/'
            },
            'hds': {
                'pattern': '2:4',
                'learning_rate': 1e-5,
                'num_epochs': 1,  # Minimal for testing
                'gumbel_temperature': 1.0,
                'sparsity_ratio': 0.5
            },
            'ptq': {
                'bits': 8,
                'scheme': 'symmetric',
                'calibration_samples': 5  # Minimal for testing
            },
            'experiment': {
                'strategy': 'baseline',
                'num_iterations': 1,
                'output_dir': str(self.temp_dir),
                'seed': 42,
                'save_intermediate': True
            },
            'benchmark': {
                'warmup_runs': 2,  # Minimal for testing
                'num_runs': 3,     # Minimal for testing
                'use_cuda_events': False,  # Avoid CUDA dependency in tests
                'cuda_sync': False
            },
            'profiling': {
                'enabled': False,  # Disable for testing
                'tool': 'pytorch_profiler',
                'metrics': []
            },
            'hardware': {
                'device': 'cpu',  # Use CPU for testing
                'gpu_id': 0,
                'mixed_precision': False
            },
            'logging': {
                'level': 'WARNING',  # Reduce noise in tests
                'file': str(self.temp_dir / 'test.log'),
                'console': False,
                'rich_formatting': False
            },
            'reproducibility': {
                'deterministic': True,
                'cuda_deterministic': False,  # Not needed for CPU
                'warn_non_deterministic': False
            }
        }
        
        # Create mock model and dataloader
        self.mock_model = self._create_mock_model()
        self.mock_dataloader = self._create_mock_dataloader()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_model(self):
        """Create a mock model for testing."""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(16, 8)
                self.linear2 = nn.Linear(8, 4)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.linear1(x))
                x = self.linear2(x)
                return x
        
        model = MockModel()
        return PermutableModel(model, 'test-model', 'test-task')
    
    def _create_mock_dataloader(self):
        """Create a mock dataloader for testing."""
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([
            torch.randn(1, 16) for _ in range(5)
        ]))
        mock_dataloader.__len__ = Mock(return_value=5)
        return mock_dataloader
    
    @patch('scripts.run_experiment.ModelManager')
    @patch('scripts.run_experiment.DatasetManager')
    def test_baseline_strategy(self, mock_dataset_manager, mock_model_manager):
        """Test baseline strategy execution."""
        # Setup mocks
        mock_model_manager.return_value.load_model.return_value = self.mock_model
        mock_dataset_manager.return_value.get_dataloader.return_value = self.mock_dataloader
        
        # Configure for baseline strategy
        config = self.test_config.copy()
        config['experiment']['strategy'] = 'baseline'
        
        # Run experiment
        runner = ExperimentRunner(config)
        results = runner.run()
        
        # Verify results
        assert results['status'] == 'completed'
        assert 'baseline_benchmark' in results
        assert 'final_benchmark' in results
        assert results['strategy_results']['type'] == 'baseline'
        
        # Verify output files
        output_dir = Path(results['output_dir'])
        assert (output_dir / 'config.json').exists()
        assert (output_dir / 'results.json').exists()
        assert (output_dir / 'summary.txt').exists()
    
    @patch('scripts.run_experiment.ModelManager')
    @patch('scripts.run_experiment.DatasetManager')
    @patch('scripts.run_experiment.IASPPermutationOptimizer')
    def test_permute_only_strategy(self, mock_iasp, mock_dataset_manager, mock_model_manager):
        """Test permute-only strategy execution."""
        # Setup mocks
        mock_model_manager.return_value.load_model.return_value = self.mock_model
        mock_dataset_manager.return_value.get_dataloader.return_value = self.mock_dataloader
        
        mock_iasp_instance = Mock()
        mock_iasp_instance.compute_permutation.return_value = (
            torch.randperm(16),  # Mock permutation
            {'modularity': 0.75}  # Mock results
        )
        mock_iasp.return_value = mock_iasp_instance
        
        # Configure for permute-only strategy
        config = self.test_config.copy()
        config['experiment']['strategy'] = 'permute_only'
        
        # Run experiment
        runner = ExperimentRunner(config)
        results = runner.run()
        
        # Verify results
        assert results['status'] == 'completed'
        assert results['strategy_results']['type'] == 'permute_only'
        assert 'iasp_results' in results['strategy_results']
        
        # Verify IASP was called
        mock_iasp_instance.compute_permutation.assert_called_once()
        mock_iasp_instance.apply_permutation.assert_called_once()
    
    @patch('scripts.run_experiment.ModelManager')
    @patch('scripts.run_experiment.DatasetManager')
    @patch('scripts.run_experiment.apply_hds_to_model')
    def test_sparsity_only_strategy(self, mock_hds, mock_dataset_manager, mock_model_manager):
        """Test sparsity-only strategy execution."""
        # Setup mocks
        mock_model_manager.return_value.load_model.return_value = self.mock_model
        mock_dataset_manager.return_value.get_dataloader.return_value = self.mock_dataloader
        
        mock_hds.return_value = (
            self.mock_model,  # Mock sparse model
            {'final_sparsity': {'avg_sparsity': 0.5}}  # Mock results
        )
        
        # Configure for sparsity-only strategy
        config = self.test_config.copy()
        config['experiment']['strategy'] = 'sparsity_only'
        
        # Run experiment
        runner = ExperimentRunner(config)
        results = runner.run()
        
        # Verify results
        assert results['status'] == 'completed'
        assert results['strategy_results']['type'] == 'sparsity_only'
        assert 'hds_results' in results['strategy_results']
        
        # Verify HDS was called
        mock_hds.assert_called_once()
    
    @patch('scripts.run_experiment.ModelManager')
    @patch('scripts.run_experiment.DatasetManager')
    @patch('scripts.run_experiment.apply_hds_to_model')
    @patch('scripts.run_experiment.IASPPermutationOptimizer')
    def test_linear_sparsity_strategy(self, mock_iasp, mock_hds, mock_dataset_manager, mock_model_manager):
        """Test linear sparsity strategy execution."""
        # Setup mocks
        mock_model_manager.return_value.load_model.return_value = self.mock_model
        mock_dataset_manager.return_value.get_dataloader.return_value = self.mock_dataloader
        
        mock_hds.return_value = (
            self.mock_model,
            {'final_sparsity': {'avg_sparsity': 0.5}}
        )
        
        mock_iasp_instance = Mock()
        mock_iasp_instance.compute_permutation.return_value = (
            torch.randperm(16),
            {'modularity': 0.75}
        )
        mock_iasp.return_value = mock_iasp_instance
        
        # Configure for linear sparsity strategy
        config = self.test_config.copy()
        config['experiment']['strategy'] = 'linear_sparsity'
        
        # Run experiment
        runner = ExperimentRunner(config)
        results = runner.run()
        
        # Verify results
        assert results['status'] == 'completed'
        assert results['strategy_results']['type'] == 'linear_sparsity'
        assert 'hds_results' in results['strategy_results']
        assert 'iasp_results' in results['strategy_results']
        
        # Verify both HDS and IASP were called
        mock_hds.assert_called_once()
        mock_iasp_instance.compute_permutation.assert_called_once()
    
    @patch('scripts.run_experiment.ModelManager')
    @patch('scripts.run_experiment.DatasetManager')
    @patch('scripts.run_experiment.apply_hds_to_model')
    @patch('scripts.run_experiment.IASPPermutationOptimizer')
    def test_iterative_sparsity_strategy(self, mock_iasp, mock_hds, mock_dataset_manager, mock_model_manager):
        """Test iterative sparsity strategy execution."""
        # Setup mocks
        mock_model_manager.return_value.load_model.return_value = self.mock_model
        mock_dataset_manager.return_value.get_dataloader.return_value = self.mock_dataloader
        
        mock_hds.return_value = (
            self.mock_model,
            {'final_sparsity': {'avg_sparsity': 0.5}}
        )
        
        mock_iasp_instance = Mock()
        mock_iasp_instance.compute_permutation.return_value = (
            torch.randperm(16),
            {'modularity': 0.75}
        )
        mock_iasp.return_value = mock_iasp_instance
        
        # Configure for iterative sparsity strategy
        config = self.test_config.copy()
        config['experiment']['strategy'] = 'iterative_sparsity'
        config['experiment']['num_iterations'] = 1
        
        # Run experiment
        runner = ExperimentRunner(config)
        results = runner.run()
        
        # Verify results
        assert results['status'] == 'completed'
        assert results['strategy_results']['type'] == 'iterative_sparsity'
        assert 'iterations' in results['strategy_results']
        assert len(results['strategy_results']['iterations']) == 1
        
        iteration = results['strategy_results']['iterations'][0]
        assert 'initial_iasp' in iteration
        assert 'hds' in iteration
        assert 'final_iasp' in iteration
        
        # Verify both HDS and IASP were called multiple times
        mock_hds.assert_called_once()
        assert mock_iasp_instance.compute_permutation.call_count == 2  # Initial and final IASP
    
    @patch('scripts.run_experiment.ModelManager')
    @patch('scripts.run_experiment.DatasetManager')
    @patch('scripts.run_experiment.quantize_model')
    @patch('scripts.run_experiment.IASPPermutationOptimizer')
    def test_iterative_quant_strategy(self, mock_iasp, mock_quantize, mock_dataset_manager, mock_model_manager):
        """Test iterative quantization strategy execution."""
        # Setup mocks
        mock_model_manager.return_value.load_model.return_value = self.mock_model
        mock_dataset_manager.return_value.get_dataloader.return_value = self.mock_dataloader
        
        mock_quantize.return_value = (
            self.mock_model,
            {'num_quantized_layers': 2}
        )
        
        mock_iasp_instance = Mock()
        mock_iasp_instance.compute_permutation.return_value = (
            torch.randperm(16),
            {'modularity': 0.75}
        )
        mock_iasp.return_value = mock_iasp_instance
        
        # Configure for iterative quantization strategy
        config = self.test_config.copy()
        config['experiment']['strategy'] = 'iterative_quant'
        
        # Run experiment
        runner = ExperimentRunner(config)
        results = runner.run()
        
        # Verify results
        assert results['status'] == 'completed'
        assert results['strategy_results']['type'] == 'iterative_quant'
        assert 'initial_iasp' in results['strategy_results']
        assert 'ptq' in results['strategy_results']
        assert 'final_iasp' in results['strategy_results']
        
        # Verify both quantization and IASP were called
        mock_quantize.assert_called_once()
        assert mock_iasp_instance.compute_permutation.call_count == 2  # Initial and final IASP
    
    def test_invalid_strategy(self):
        """Test handling of invalid strategy."""
        config = self.test_config.copy()
        config['experiment']['strategy'] = 'invalid_strategy'
        
        runner = ExperimentRunner(config)
        
        with pytest.raises(IterativeCoDesignError):
            runner.run()
    
    @patch('scripts.run_experiment.ModelManager')
    @patch('scripts.run_experiment.DatasetManager')
    def test_experiment_reproducibility(self, mock_dataset_manager, mock_model_manager):
        """Test that experiments are reproducible with same seed."""
        # Setup mocks
        mock_model_manager.return_value.load_model.return_value = self.mock_model
        mock_dataset_manager.return_value.get_dataloader.return_value = self.mock_dataloader
        
        config = self.test_config.copy()
        config['experiment']['strategy'] = 'baseline'
        config['experiment']['seed'] = 12345
        
        # Run experiment twice with same seed
        runner1 = ExperimentRunner(config)
        results1 = runner1.run()
        
        # Create new temp directory for second run
        temp_dir2 = Path(tempfile.mkdtemp())
        config['experiment']['output_dir'] = str(temp_dir2)
        
        runner2 = ExperimentRunner(config)
        results2 = runner2.run()
        
        # Verify reproducibility info is consistent
        assert results1['reproducibility_info']['seed'] == results2['reproducibility_info']['seed']
        
        # Clean up
        shutil.rmtree(temp_dir2, ignore_errors=True)
    
    @patch('scripts.run_experiment.ModelManager')
    @patch('scripts.run_experiment.DatasetManager')
    def test_experiment_output_structure(self, mock_dataset_manager, mock_model_manager):
        """Test that experiment outputs have correct structure."""
        # Setup mocks
        mock_model_manager.return_value.load_model.return_value = self.mock_model
        mock_dataset_manager.return_value.get_dataloader.return_value = self.mock_dataloader
        
        config = self.test_config.copy()
        config['experiment']['strategy'] = 'baseline'
        
        # Run experiment
        runner = ExperimentRunner(config)
        results = runner.run()
        
        # Verify results structure
        required_keys = [
            'config', 'timestamp', 'experiment_id', 'reproducibility_info',
            'output_dir', 'baseline_benchmark', 'final_benchmark',
            'strategy_results', 'improvements', 'status', 'duration_seconds'
        ]
        
        for key in required_keys:
            assert key in results, f"Missing required key: {key}"
        
        # Verify improvements calculation
        improvements = results['improvements']
        assert 'baseline_latency_ms' in improvements
        assert 'final_latency_ms' in improvements
        assert 'improvement_ms' in improvements
        assert 'improvement_pct' in improvements
        assert 'speedup_factor' in improvements
        
        # Verify output files
        output_dir = Path(results['output_dir'])
        assert output_dir.exists()
        assert (output_dir / 'config.json').exists()
        assert (output_dir / 'results.json').exists()
        assert (output_dir / 'summary.txt').exists()
        
        # Verify config.json is valid JSON
        with open(output_dir / 'config.json') as f:
            saved_config = json.load(f)
            assert saved_config == config
        
        # Verify results.json is valid JSON
        with open(output_dir / 'results.json') as f:
            saved_results = json.load(f)
            assert saved_results['experiment_id'] == results['experiment_id']
    
    def test_config_loading_and_merging(self):
        """Test configuration loading and CLI argument merging."""
        # Create temporary config file
        config_file = self.temp_dir / 'test_config.yaml'
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Load config
        loaded_config = load_config(str(config_file))
        assert loaded_config == self.test_config
        
        # Test argument merging
        class MockArgs:
            strategy = 'iterative_sparsity'
            model = 'bert-large'
            num_iterations = 3
            seed = 999
            layer_name = 'encoder.layer.0'
            
            # Attributes that might not exist
            def __getattr__(self, name):
                return None
        
        args = MockArgs()
        merged_config = merge_config_with_args(loaded_config, args)
        
        # Verify CLI args override config
        assert merged_config['experiment']['strategy'] == 'iterative_sparsity'
        assert merged_config['model']['name'] == 'bert-large'
        assert merged_config['experiment']['num_iterations'] == 3
        assert merged_config['experiment']['seed'] == 999
        assert merged_config['iasp']['layer_name'] == 'encoder.layer.0'
        
        # Verify other values preserved
        assert merged_config['dataset']['name'] == self.test_config['dataset']['name']
    
    @patch('scripts.run_experiment.ModelManager')
    @patch('scripts.run_experiment.DatasetManager')
    def test_save_intermediate_results(self, mock_dataset_manager, mock_model_manager):
        """Test saving intermediate results."""
        # Setup mocks
        mock_model_manager.return_value.load_model.return_value = self.mock_model
        mock_dataset_manager.return_value.get_dataloader.return_value = self.mock_dataloader
        
        config = self.test_config.copy()
        config['experiment']['strategy'] = 'baseline'
        config['experiment']['save_intermediate'] = True
        
        # Run experiment
        runner = ExperimentRunner(config)
        results = runner.run()
        
        # For baseline strategy, no intermediate files should be saved
        output_dir = Path(results['output_dir'])
        permutation_files = list(output_dir.glob('permutation_*.pt'))
        assert len(permutation_files) == 0  # Baseline shouldn't save permutations


class TestExperimentCLI:
    """Test the CLI interface."""
    
    def test_cli_argument_parsing(self):
        """Test CLI argument parsing works correctly."""
        # This would require actually calling main.py with subprocess
        # For now, we'll test the argument parser directly
        from main import parse_args
        
        # Mock command line arguments
        import sys
        original_argv = sys.argv
        
        try:
            sys.argv = [
                'main.py',
                '--strategy', 'iterative_sparsity',
                '--model', 'mamba-3b',
                '--dataset', 'wikitext-103',
                '--num-iterations', '2',
                '--seed', '42',
                '--dry-run'
            ]
            
            args = parse_args()
            
            assert args.strategy == 'iterative_sparsity'
            assert args.model == 'mamba-3b'
            assert args.dataset == 'wikitext-103'
            assert args.num_iterations == 2
            assert args.seed == 42
            assert args.dry_run is True
            
        finally:
            sys.argv = original_argv


if __name__ == '__main__':
    pytest.main([__file__, '-v'])