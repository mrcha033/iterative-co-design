"""
Unit tests for configuration utilities.

These tests verify configuration loading, validation, and management.
"""
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from src.utils.config import (
    ConfigManager, load_config, validate_config, merge_configs,
    get_default_config, save_config
)


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config_manager = ConfigManager()
    
    def test_manager_initialization(self):
        """Test config manager initialization."""
        assert hasattr(self.config_manager, 'config')
        assert isinstance(self.config_manager.config, dict)
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        default_config = self.config_manager.get_default_config()
        
        assert isinstance(default_config, dict)
        assert 'model' in default_config
        assert 'dataset' in default_config
        assert 'experiment' in default_config
        assert 'hardware' in default_config
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            'model': {
                'name': 'mamba-3b',
                'precision': 'float16'
            },
            'dataset': {
                'name': 'wikitext-103',
                'batch_size': 4,
                'num_samples': 1000
            },
            'experiment': {
                'strategy': 'iterative_sparsity',
                'seed': 42,
                'output_dir': './results'
            },
            'hardware': {
                'device': 'cuda',
                'gpu_id': 0
            }
        }
        
        self.config_manager.validate_config(valid_config)  # Should not raise
    
    def test_config_validation_missing_sections(self):
        """Test configuration validation with missing sections."""
        incomplete_config = {
            'model': {
                'name': 'mamba-3b'
            }
            # Missing dataset, experiment, hardware sections
        }
        
        with pytest.raises(ValueError, match="Missing required section"):
            self.config_manager.validate_config(incomplete_config)
    
    def test_config_validation_invalid_values(self):
        """Test configuration validation with invalid values."""
        invalid_configs = [
            {
                'model': {'name': ''},  # Empty model name
                'dataset': {'name': 'wikitext-103', 'batch_size': 4},
                'experiment': {'strategy': 'baseline', 'seed': 42},
                'hardware': {'device': 'cuda'}
            },
            {
                'model': {'name': 'mamba-3b'},
                'dataset': {'name': 'wikitext-103', 'batch_size': 0},  # Invalid batch size
                'experiment': {'strategy': 'baseline', 'seed': 42},
                'hardware': {'device': 'cuda'}
            },
            {
                'model': {'name': 'mamba-3b'},
                'dataset': {'name': 'wikitext-103', 'batch_size': 4},
                'experiment': {'strategy': 'invalid_strategy', 'seed': 42},  # Invalid strategy
                'hardware': {'device': 'cuda'}
            }
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ValueError):
                self.config_manager.validate_config(invalid_config)
    
    def test_merge_configs(self):
        """Test configuration merging."""
        base_config = {
            'model': {
                'name': 'mamba-3b',
                'precision': 'float16'
            },
            'dataset': {
                'name': 'wikitext-103',
                'batch_size': 4
            }
        }
        
        override_config = {
            'model': {
                'precision': 'float32'  # Override existing
            },
            'dataset': {
                'num_samples': 2000  # Add new field
            },
            'experiment': {  # Add new section
                'strategy': 'baseline'
            }
        }
        
        merged = self.config_manager.merge_configs(base_config, override_config)
        
        assert merged['model']['name'] == 'mamba-3b'  # Preserved
        assert merged['model']['precision'] == 'float32'  # Overridden
        assert merged['dataset']['batch_size'] == 4  # Preserved
        assert merged['dataset']['num_samples'] == 2000  # Added
        assert merged['experiment']['strategy'] == 'baseline'  # Added section
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        test_config = {
            'model': {
                'name': 'test-model',
                'precision': 'float16'
            },
            'experiment': {
                'strategy': 'baseline',
                'seed': 123
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_file = f.name
        
        try:
            # Save config
            self.config_manager.save_config(test_config, config_file)
            
            # Load config
            loaded_config = self.config_manager.load_config(config_file)
            
            assert loaded_config == test_config
            
        finally:
            Path(config_file).unlink(missing_ok=True)


class TestConfigUtilityFunctions:
    """Test standalone configuration utility functions."""
    
    def test_load_config_yaml(self):
        """Test loading YAML configuration file."""
        test_config = {
            'model': {'name': 'test'},
            'dataset': {'name': 'test-data', 'batch_size': 8},
            'experiment': {'strategy': 'baseline'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_file = f.name
        
        try:
            loaded_config = load_config(config_file)
            assert loaded_config == test_config
        finally:
            Path(config_file).unlink(missing_ok=True)
    
    def test_load_config_json(self):
        """Test loading JSON configuration file."""
        import json
        
        test_config = {
            'model': {'name': 'test'},
            'dataset': {'name': 'test-data', 'batch_size': 8}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_file = f.name
        
        try:
            loaded_config = load_config(config_file)
            assert loaded_config == test_config
        finally:
            Path(config_file).unlink(missing_ok=True)
    
    def test_load_config_nonexistent(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/config.yaml')
    
    def test_load_config_invalid_format(self):
        """Test loading invalid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")  # Invalid YAML
            config_file = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                load_config(config_file)
        finally:
            Path(config_file).unlink(missing_ok=True)
    
    def test_validate_config_function(self):
        """Test standalone validate_config function."""
        valid_config = {
            'model': {'name': 'mamba-3b'},
            'dataset': {'name': 'wikitext-103', 'batch_size': 4},
            'experiment': {'strategy': 'baseline', 'seed': 42},
            'hardware': {'device': 'cuda'}
        }
        
        validate_config(valid_config)  # Should not raise
    
    def test_merge_configs_function(self):
        """Test standalone merge_configs function."""
        config1 = {'a': 1, 'b': {'x': 10}}
        config2 = {'b': {'y': 20}, 'c': 3}
        
        merged = merge_configs(config1, config2)
        
        assert merged['a'] == 1
        assert merged['b']['x'] == 10
        assert merged['b']['y'] == 20
        assert merged['c'] == 3
    
    def test_get_default_config_function(self):
        """Test get_default_config function."""
        default = get_default_config()
        
        assert isinstance(default, dict)
        assert 'model' in default
        assert 'dataset' in default
        assert 'experiment' in default
        assert 'hardware' in default
    
    def test_save_config_function(self):
        """Test save_config function."""
        test_config = {'test': 'value'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_file = f.name
        
        try:
            save_config(test_config, config_file)
            
            # Verify file was created and contains correct data
            with open(config_file, 'r') as f:
                loaded = yaml.safe_load(f)
            
            assert loaded == test_config
        finally:
            Path(config_file).unlink(missing_ok=True)


class TestConfigValidationRules:
    """Test specific configuration validation rules."""
    
    def test_model_validation(self):
        """Test model configuration validation."""
        valid_models = ['mamba-3b', 'bert-large', 'resnet-50', 'gcn']
        
        for model_name in valid_models:
            config = {
                'model': {'name': model_name},
                'dataset': {'name': 'wikitext-103', 'batch_size': 4},
                'experiment': {'strategy': 'baseline', 'seed': 42},
                'hardware': {'device': 'cuda'}
            }
            validate_config(config)  # Should not raise
    
    def test_strategy_validation(self):
        """Test experiment strategy validation."""
        valid_strategies = [
            'baseline', 'permute_only', 'sparsity_only', 'linear_sparsity',
            'iterative_sparsity', 'linear_quant_permute_first',
            'linear_quant_quant_first', 'iterative_quant'
        ]
        
        for strategy in valid_strategies:
            config = {
                'model': {'name': 'mamba-3b'},
                'dataset': {'name': 'wikitext-103', 'batch_size': 4},
                'experiment': {'strategy': strategy, 'seed': 42},
                'hardware': {'device': 'cuda'}
            }
            validate_config(config)  # Should not raise
    
    def test_dataset_validation(self):
        """Test dataset configuration validation."""
        valid_datasets = ['wikitext-103', 'imagenet', 'ogbn-arxiv']
        
        for dataset_name in valid_datasets:
            config = {
                'model': {'name': 'mamba-3b'},
                'dataset': {'name': dataset_name, 'batch_size': 4},
                'experiment': {'strategy': 'baseline', 'seed': 42},
                'hardware': {'device': 'cuda'}
            }
            validate_config(config)  # Should not raise
    
    def test_hardware_validation(self):
        """Test hardware configuration validation."""
        valid_devices = ['cuda', 'cpu']
        
        for device in valid_devices:
            config = {
                'model': {'name': 'mamba-3b'},
                'dataset': {'name': 'wikitext-103', 'batch_size': 4},
                'experiment': {'strategy': 'baseline', 'seed': 42},
                'hardware': {'device': device}
            }
            validate_config(config)  # Should not raise
    
    def test_batch_size_validation(self):
        """Test batch size validation."""
        valid_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        
        for batch_size in valid_batch_sizes:
            config = {
                'model': {'name': 'mamba-3b'},
                'dataset': {'name': 'wikitext-103', 'batch_size': batch_size},
                'experiment': {'strategy': 'baseline', 'seed': 42},
                'hardware': {'device': 'cuda'}
            }
            validate_config(config)  # Should not raise
        
        # Invalid batch sizes
        invalid_batch_sizes = [0, -1, 0.5, 'invalid']
        
        for batch_size in invalid_batch_sizes:
            config = {
                'model': {'name': 'mamba-3b'},
                'dataset': {'name': 'wikitext-103', 'batch_size': batch_size},
                'experiment': {'strategy': 'baseline', 'seed': 42},
                'hardware': {'device': 'cuda'}
            }
            with pytest.raises(ValueError):
                validate_config(config)


class TestConfigEnvironmentOverrides:
    """Test environment variable overrides."""
    
    @patch.dict('os.environ', {'ITERATIVE_MODEL_NAME': 'bert-large'})
    def test_environment_model_override(self):
        """Test model name override from environment."""
        config_manager = ConfigManager()
        
        # If the implementation supports environment overrides
        if hasattr(config_manager, 'apply_environment_overrides'):
            config = {'model': {'name': 'mamba-3b'}}
            overridden = config_manager.apply_environment_overrides(config)
            assert overridden['model']['name'] == 'bert-large'
    
    @patch.dict('os.environ', {'ITERATIVE_BATCH_SIZE': '16'})
    def test_environment_batch_size_override(self):
        """Test batch size override from environment."""
        config_manager = ConfigManager()
        
        if hasattr(config_manager, 'apply_environment_overrides'):
            config = {'dataset': {'batch_size': 4}}
            overridden = config_manager.apply_environment_overrides(config)
            assert overridden['dataset']['batch_size'] == 16


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_config(self):
        """Test handling of empty configuration."""
        with pytest.raises(ValueError):
            validate_config({})
    
    def test_none_config(self):
        """Test handling of None configuration."""
        with pytest.raises((ValueError, TypeError)):
            validate_config(None)
    
    def test_config_with_extra_fields(self):
        """Test configuration with extra unknown fields."""
        config_with_extra = {
            'model': {'name': 'mamba-3b'},
            'dataset': {'name': 'wikitext-103', 'batch_size': 4},
            'experiment': {'strategy': 'baseline', 'seed': 42},
            'hardware': {'device': 'cuda'},
            'unknown_section': {'unknown_field': 'value'}
        }
        
        # Should either accept gracefully or raise appropriate error
        try:
            validate_config(config_with_extra)
        except ValueError as e:
            assert 'unknown' in str(e).lower()
    
    def test_deeply_nested_config_merge(self):
        """Test merging deeply nested configurations."""
        base = {
            'a': {
                'b': {
                    'c': {
                        'd': 1,
                        'e': 2
                    }
                }
            }
        }
        
        override = {
            'a': {
                'b': {
                    'c': {
                        'e': 3,  # Override
                        'f': 4   # Add
                    }
                }
            }
        }
        
        merged = merge_configs(base, override)
        
        assert merged['a']['b']['c']['d'] == 1  # Preserved
        assert merged['a']['b']['c']['e'] == 3  # Overridden
        assert merged['a']['b']['c']['f'] == 4  # Added


if __name__ == '__main__':
    pytest.main([__file__, '-v'])