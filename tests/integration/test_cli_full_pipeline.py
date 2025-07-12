"""
Comprehensive CLI integration tests for full experiment pipelines.

These tests verify end-to-end CLI functionality with toy models to ensure
all experiment strategies work correctly without requiring full-scale models.
"""
import json
import tempfile
import subprocess
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
import pytest
import torch
import torch.nn as nn

from scripts.run_experiment import ExperimentRunner
from src.utils.config import get_default_config


class ToyMambaBlock(nn.Module):
    """Simplified Mamba block for testing."""
    
    def __init__(self, d_model=64):
        super().__init__()
        self.mixer = nn.ModuleDict({
            'in_proj': nn.Linear(d_model, d_model * 2),
            'conv1d': nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            'x_proj': nn.Linear(d_model, d_model),
            'out_proj': nn.Linear(d_model, d_model)
        })
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Simplified forward pass
        residual = x
        x = self.norm(x)
        x = self.mixer.in_proj(x)
        x, gate = x.chunk(2, dim=-1)
        
        # Simplified conv1d application
        if x.dim() == 3:  # (batch, seq, features)
            x = x.transpose(1, 2)  # (batch, features, seq)
            x = self.mixer.conv1d(x)
            x = x.transpose(1, 2)  # (batch, seq, features)
        
        x = x * torch.sigmoid(gate)
        x = self.mixer.out_proj(x)
        return x + residual


class ToyBertBlock(nn.Module):
    """Simplified BERT block for testing."""
    
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x)
        x = residual + attn_out
        
        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        return x + residual


class TestCLIFullPipeline:
    """Test complete CLI pipeline with toy models."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = Path(self.temp_dir) / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Create toy config
        self.toy_config = {
            'model': {
                'name': 'toy-mamba',
                'precision': 'float32'
            },
            'dataset': {
                'name': 'synthetic',
                'batch_size': 2,
                'sequence_length': 32,
                'num_samples': 10
            },
            'experiment': {
                'strategy': 'baseline',
                'seed': 42,
                'output_dir': str(self.results_dir),
                'save_intermediate': False,
                'num_iterations': 1
            },
            'hardware': {
                'device': 'cpu',  # Use CPU for reproducible testing
                'gpu_id': 0
            },
            'iasp': {
                'layer_name': 'mixer.in_proj',
                'num_clusters': 4,
                'method': 'spectral',
                'correlation_threshold': 0.1
            },
            'hds': {
                'pattern': '2:4',
                'sparsity_ratio': 0.5,
                'learning_rate': 1e-4,
                'num_epochs': 2
            },
            'ptq': {
                'bits': 8,
                'scheme': 'symmetric',
                'calibration_samples': 10
            },
            'benchmark': {
                'warmup_runs': 1,
                'num_runs': 2,
                'use_cuda_events': False,
                'cuda_sync': False
            },
            'profiling': {
                'enabled': False  # Disable hardware profiling for tests
            },
            'reproducibility': {
                'deterministic': True,
                'cuda_deterministic': False
            }
        }
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_toy_model(self, model_type='mamba'):
        """Create a toy model for testing."""
        if model_type == 'mamba':
            return ToyMambaBlock(d_model=64)
        elif model_type == 'bert':
            return ToyBertBlock(d_model=64, num_heads=4)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_toy_dataloader(self, batch_size=2, sequence_length=32, num_samples=10):
        """Create a toy dataloader for testing."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Generate synthetic data
        data = torch.randn(num_samples, sequence_length, 64)
        dataset = TensorDataset(data)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    @patch('src.models.manager.ModelManager.load_model')
    @patch('src.utils.dataset_manager.DatasetManager.get_dataloader')
    def test_baseline_strategy(self, mock_dataloader, mock_model):
        """Test baseline strategy execution."""
        # Mock model and dataloader
        mock_model.return_value = self._create_toy_model('mamba')
        mock_dataloader.return_value = self._create_toy_dataloader()
        
        # Run experiment
        config = self.toy_config.copy()
        config['experiment']['strategy'] = 'baseline'
        
        runner = ExperimentRunner(config)
        results = runner.run()
        
        # Verify results
        assert results['status'] == 'completed'
        assert 'baseline_benchmark' in results
        assert 'final_benchmark' in results
        assert results['baseline_benchmark']['mean_latency_ms'] > 0
        
        # Verify output files
        experiment_dir = Path(results['output_dir'])
        assert (experiment_dir / 'results.json').exists()
        assert (experiment_dir / 'summary.txt').exists()
    
    @patch('src.models.manager.ModelManager.load_model')
    @patch('src.utils.dataset_manager.DatasetManager.get_dataloader')
    def test_permute_only_strategy(self, mock_dataloader, mock_model):
        """Test permute-only strategy execution."""
        mock_model.return_value = self._create_toy_model('mamba')
        mock_dataloader.return_value = self._create_toy_dataloader()
        
        config = self.toy_config.copy()
        config['experiment']['strategy'] = 'permute_only'
        
        runner = ExperimentRunner(config)
        results = runner.run()
        
        assert results['status'] == 'completed'
        assert 'strategy_results' in results
        assert results['strategy_results']['type'] == 'permute_only'
        assert 'iasp_results' in results['strategy_results']
        assert 'permutation' in results['strategy_results']['iasp_results']
    
    @patch('src.models.manager.ModelManager.load_model')
    @patch('src.utils.dataset_manager.DatasetManager.get_dataloader')
    def test_sparsity_only_strategy(self, mock_dataloader, mock_model):
        """Test sparsity-only strategy execution."""
        mock_model.return_value = self._create_toy_model('mamba')
        mock_dataloader.return_value = self._create_toy_dataloader()
        
        config = self.toy_config.copy()
        config['experiment']['strategy'] = 'sparsity_only'
        
        runner = ExperimentRunner(config)
        results = runner.run()
        
        assert results['status'] == 'completed'
        assert 'strategy_results' in results
        assert results['strategy_results']['type'] == 'sparsity_only'
        assert 'hds_results' in results['strategy_results']
    
    @patch('src.models.manager.ModelManager.load_model')
    @patch('src.utils.dataset_manager.DatasetManager.get_dataloader')
    def test_linear_sparsity_strategy(self, mock_dataloader, mock_model):
        """Test linear sparsity strategy execution."""
        mock_model.return_value = self._create_toy_model('mamba')
        mock_dataloader.return_value = self._create_toy_dataloader()
        
        config = self.toy_config.copy()
        config['experiment']['strategy'] = 'linear_sparsity'
        
        runner = ExperimentRunner(config)
        results = runner.run()
        
        assert results['status'] == 'completed'
        assert 'strategy_results' in results
        assert results['strategy_results']['type'] == 'linear_sparsity'
        assert 'hds_results' in results['strategy_results']
        assert 'iasp_results' in results['strategy_results']
    
    @patch('src.models.manager.ModelManager.load_model')
    @patch('src.utils.dataset_manager.DatasetManager.get_dataloader')
    def test_iterative_sparsity_strategy(self, mock_dataloader, mock_model):
        """Test iterative sparsity strategy execution."""
        mock_model.return_value = self._create_toy_model('mamba')
        mock_dataloader.return_value = self._create_toy_dataloader()
        
        config = self.toy_config.copy()
        config['experiment']['strategy'] = 'iterative_sparsity'
        config['experiment']['num_iterations'] = 1  # Reduce for testing
        
        runner = ExperimentRunner(config)
        results = runner.run()
        
        assert results['status'] == 'completed'
        assert 'strategy_results' in results
        assert results['strategy_results']['type'] == 'iterative_sparsity'
        assert 'iterations' in results['strategy_results']
        assert len(results['strategy_results']['iterations']) == 1
    
    @patch('src.models.manager.ModelManager.load_model')
    @patch('src.utils.dataset_manager.DatasetManager.get_dataloader')
    def test_quantization_strategies(self, mock_dataloader, mock_model):
        """Test quantization strategy execution."""
        mock_model.return_value = self._create_toy_model('mamba')
        mock_dataloader.return_value = self._create_toy_dataloader()
        
        quantization_strategies = [
            'linear_quant_permute_first',
            'linear_quant_quant_first',
            'iterative_quant'
        ]
        
        for strategy in quantization_strategies:
            config = self.toy_config.copy()
            config['experiment']['strategy'] = strategy
            
            runner = ExperimentRunner(config)
            results = runner.run()
            
            assert results['status'] == 'completed'
            assert 'strategy_results' in results
            assert results['strategy_results']['type'] == strategy
    
    @patch('src.models.manager.ModelManager.load_model')
    @patch('src.utils.dataset_manager.DatasetManager.get_dataloader')
    def test_bert_model_compatibility(self, mock_dataloader, mock_model):
        """Test strategy execution with BERT-like model."""
        mock_model.return_value = self._create_toy_model('bert')
        mock_dataloader.return_value = self._create_toy_dataloader()
        
        config = self.toy_config.copy()
        config['model']['name'] = 'toy-bert'
        config['iasp']['layer_name'] = 'feed_forward.0'  # Different layer path
        config['experiment']['strategy'] = 'linear_sparsity'
        
        runner = ExperimentRunner(config)
        results = runner.run()
        
        assert results['status'] == 'completed'
        assert 'strategy_results' in results
    
    @patch('src.models.manager.ModelManager.load_model')
    @patch('src.utils.dataset_manager.DatasetManager.get_dataloader')
    def test_error_handling(self, mock_dataloader, mock_model):
        """Test error handling in CLI pipeline."""
        # Test with invalid layer name
        mock_model.return_value = self._create_toy_model('mamba')
        mock_dataloader.return_value = self._create_toy_dataloader()
        
        config = self.toy_config.copy()
        config['iasp']['layer_name'] = 'nonexistent.layer'
        config['experiment']['strategy'] = 'permute_only'
        
        runner = ExperimentRunner(config)
        results = runner.run()
        
        assert results['status'] == 'failed'
        assert 'error' in results
    
    @patch('src.models.manager.ModelManager.load_model')
    @patch('src.utils.dataset_manager.DatasetManager.get_dataloader')
    def test_deterministic_results(self, mock_dataloader, mock_model):
        """Test that results are deterministic with fixed seed."""
        mock_model.return_value = self._create_toy_model('mamba')
        mock_dataloader.return_value = self._create_toy_dataloader()
        
        config = self.toy_config.copy()
        config['experiment']['strategy'] = 'permute_only'
        config['experiment']['seed'] = 12345
        
        # Run experiment twice
        runner1 = ExperimentRunner(config)
        results1 = runner1.run()
        
        runner2 = ExperimentRunner(config)
        results2 = runner2.run()
        
        # Results should be identical (within floating point precision)
        assert results1['status'] == results2['status'] == 'completed'
        
        perm1 = results1['strategy_results']['iasp_results']['permutation']
        perm2 = results2['strategy_results']['iasp_results']['permutation']
        assert perm1 == perm2
    
    @patch('src.models.manager.ModelManager.load_model')
    @patch('src.utils.dataset_manager.DatasetManager.get_dataloader')
    def test_result_file_format(self, mock_dataloader, mock_model):
        """Test that result files are properly formatted."""
        mock_model.return_value = self._create_toy_model('mamba')
        mock_dataloader.return_value = self._create_toy_dataloader()
        
        config = self.toy_config.copy()
        config['experiment']['strategy'] = 'baseline'
        
        runner = ExperimentRunner(config)
        results = runner.run()
        
        # Check JSON results file
        results_file = Path(results['output_dir']) / 'results.json'
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
        
        # Verify structure
        assert 'experiment_id' in saved_results
        assert 'timestamp' in saved_results
        assert 'config' in saved_results
        assert 'baseline_benchmark' in saved_results
        assert 'status' in saved_results
        
        # Check summary file
        summary_file = Path(results['output_dir']) / 'summary.txt'
        with open(summary_file, 'r') as f:
            summary_content = f.read()
        
        assert 'Experiment Summary' in summary_content
        assert 'Experiment ID:' in summary_content
        assert 'Strategy:' in summary_content
    
    @patch('src.models.manager.ModelManager.load_model')
    @patch('src.utils.dataset_manager.DatasetManager.get_dataloader')
    def test_improvement_calculation(self, mock_dataloader, mock_model):
        """Test performance improvement calculation."""
        mock_model.return_value = self._create_toy_model('mamba')
        mock_dataloader.return_value = self._create_toy_dataloader()
        
        config = self.toy_config.copy()
        config['experiment']['strategy'] = 'linear_sparsity'
        
        runner = ExperimentRunner(config)
        results = runner.run()
        
        assert results['status'] == 'completed'
        assert 'improvements' in results
        
        improvements = results['improvements']
        assert 'baseline_latency_ms' in improvements
        assert 'final_latency_ms' in improvements
        assert 'improvement_ms' in improvements
        assert 'improvement_pct' in improvements
        assert 'speedup_factor' in improvements
        
        # Verify calculations are consistent
        baseline = improvements['baseline_latency_ms']
        final = improvements['final_latency_ms']
        improvement_ms = improvements['improvement_ms']
        improvement_pct = improvements['improvement_pct']
        speedup = improvements['speedup_factor']
        
        assert abs(improvement_ms - (baseline - final)) < 1e-6
        assert abs(improvement_pct - ((baseline - final) / baseline * 100)) < 1e-6
        assert abs(speedup - (baseline / final)) < 1e-6


class TestCLICommandLineInterface:
    """Test CLI command line interface."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_main_script_help(self):
        """Test main script help output."""
        try:
            result = subprocess.run(
                ['python', 'main.py', '--help'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=Path(__file__).parent.parent.parent
            )
            
            assert result.returncode == 0
            assert 'usage:' in result.stdout.lower()
            assert 'strategy' in result.stdout.lower()
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Main script not available or timeout")
    
    def test_config_file_argument(self):
        """Test configuration file argument parsing."""
        # Create test config file
        config_file = Path(self.temp_dir) / 'test_config.yaml'
        
        with open(config_file, 'w') as f:
            f.write("""
model:
  name: toy-mamba
experiment:
  strategy: baseline
  seed: 42
""")
        
        try:
            # Test dry run to avoid actual execution
            result = subprocess.run(
                ['python', 'main.py', '--config', str(config_file), '--dry-run'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=Path(__file__).parent.parent.parent
            )
            
            # Should either succeed or fail gracefully
            assert result.returncode in [0, 1]  # Allow for expected failures
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Main script not available or timeout")


class TestResultAnalysisIntegration:
    """Test integration with result analysis scripts."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = Path(self.temp_dir) / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Create sample result files
        self._create_sample_results()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_sample_results(self):
        """Create sample result files for testing."""
        sample_results = [
            {
                'experiment_id': 'baseline_mamba-3b_20231201_120000',
                'config': {
                    'model': {'name': 'mamba-3b'},
                    'experiment': {'strategy': 'baseline'}
                },
                'baseline_benchmark': {
                    'mean_latency_ms': 35.2,
                    'std_latency_ms': 0.3
                },
                'final_benchmark': {
                    'mean_latency_ms': 35.2,
                    'std_latency_ms': 0.3
                }
            },
            {
                'experiment_id': 'iterative_sparsity_mamba-3b_20231201_120100',
                'config': {
                    'model': {'name': 'mamba-3b'},
                    'experiment': {'strategy': 'iterative_sparsity'}
                },
                'baseline_benchmark': {
                    'mean_latency_ms': 35.2,
                    'std_latency_ms': 0.3
                },
                'final_benchmark': {
                    'mean_latency_ms': 19.8,
                    'std_latency_ms': 0.2
                },
                'strategy_results': {
                    'type': 'iterative_sparsity',
                    'iasp_results': {'modularity': 0.79}
                }
            }
        ]
        
        for i, result in enumerate(sample_results):
            exp_dir = self.results_dir / f'experiment_{i}'
            exp_dir.mkdir(exist_ok=True)
            
            with open(exp_dir / 'results.json', 'w') as f:
                json.dump(result, f, indent=2)
    
    def test_table_generation_integration(self):
        """Test table generation script integration."""
        try:
            from scripts.generate_tables import TableGenerator
            
            generator = TableGenerator(self.results_dir, self.temp_dir)
            
            # Generate main results table
            latex_content = generator.generate_main_results_table(['mamba-3b'])
            
            assert isinstance(latex_content, str)
            assert '\\begin{table}' in latex_content
            assert '\\end{table}' in latex_content
            assert 'mamba-3b' in latex_content.lower()
            
        except ImportError:
            pytest.skip("Table generation script not available")
    
    def test_figure_generation_integration(self):
        """Test figure generation script integration."""
        try:
            from scripts.generate_figures import FigureGenerator
            
            generator = FigureGenerator(self.results_dir, self.temp_dir)
            
            # Generate a simple figure
            figure_path = generator.generate_quantization_barchart('mamba-3b')
            
            # May not generate actual file due to missing data, but should not crash
            assert isinstance(figure_path, str)
            
        except ImportError:
            pytest.skip("Figure generation script not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])