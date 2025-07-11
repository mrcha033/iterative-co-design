"""
Integration tests for model and dataset loading.
"""
import pytest
import tempfile
import torch
from pathlib import Path

from src.models.manager import ModelManager
from src.utils.dataset_manager import DatasetManager
from src.utils.config import Config, create_default_config


class TestModelIntegration:
    """Integration tests for model loading and usage."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_manager = ModelManager(cache_dir=self.temp_dir)
        self.dataset_manager = DatasetManager(data_dir=self.temp_dir)
    
    @pytest.mark.slow
    def test_gcn_model_loading(self):
        """Test loading and using GCN model."""
        # Load GCN model (this should work without external dependencies)
        model = self.model_manager.load_model('gcn', device='cpu')
        
        # Test basic properties
        assert model.model_name == 'gcn'
        assert model.model_type == 'gcn'
        
        # Test layer access
        layer_names = model.get_layer_names()
        assert len(layer_names) > 0
        
        # Test forward pass with dummy data
        x = torch.randn(100, 128)  # 100 nodes, 128 features
        edge_index = torch.randint(0, 100, (2, 200))  # 200 edges
        
        try:
            output = model(x, edge_index)
            assert output.shape[0] == 100  # Should have 100 node outputs
        except Exception as e:
            # If torch_geometric is not available, this is expected
            assert "torch_geometric" in str(e) or "GCNConv" in str(e)
    
    def test_config_integration(self):
        """Test configuration integration."""
        config = create_default_config()
        
        # Test configuration validation
        assert config.model.name == 'mamba-3b'
        assert config.dataset.name == 'wikitext-103'
        assert config.iasp.layer_name == 'layers.0.mixer'
        
        # Test serialization
        config_path = Path(self.temp_dir) / 'test_config.yaml'
        config.to_yaml(str(config_path))
        assert config_path.exists()
        
        # Test loading from file
        loaded_config = Config.from_yaml(str(config_path))
        assert loaded_config.model.name == config.model.name
        assert loaded_config.dataset.name == config.dataset.name
    
    def test_dataset_manager_integration(self):
        """Test dataset manager integration."""
        # Test supported datasets
        datasets = self.dataset_manager.list_supported_datasets()
        assert 'wikitext-103' in datasets
        assert 'imagenet' in datasets
        assert 'ogbn-arxiv' in datasets
        
        # Test dataset info
        info = self.dataset_manager.get_dataset_info('wikitext-103')
        assert info['type'] == 'text'
        assert info['task'] == 'language_modeling'
        
        # Test dataset stats (should show not downloaded)
        stats = self.dataset_manager.get_dataset_stats('wikitext-103')
        assert stats['status'] == 'not_downloaded'
        assert stats['size_mb'] == 0
    
    def test_model_layer_validation(self):
        """Test model layer validation."""
        # Test with GCN model (since it doesn't require external downloads)
        try:
            model = self.model_manager.load_model('gcn', device='cpu')
            layer_names = model.get_layer_names()
            
            if layer_names:
                # Test valid layer
                first_layer = layer_names[0]
                info = model.get_layer_info(first_layer)
                assert 'name' in info
                assert 'type' in info
                
                # Test invalid layer
                with pytest.raises(ValueError, match="Layer 'nonexistent' not found"):
                    model.get_layer('nonexistent')
        except Exception as e:
            # If torch_geometric is not available, skip this test
            if "torch_geometric" in str(e):
                pytest.skip("torch_geometric not available")
            else:
                raise
    
    def test_permutation_integration(self):
        """Test permutation functionality integration."""
        try:
            model = self.model_manager.load_model('gcn', device='cpu')
            
            # Get permutable layers
            permutable_layers = model.get_permutable_layers()
            
            if permutable_layers:
                layer_name = permutable_layers[0]
                
                # Test dimension size
                try:
                    input_size = model.get_dimension_size(layer_name, 'input')
                    assert input_size > 0
                    
                    # Create and apply permutation
                    perm = torch.randperm(input_size).numpy()
                    model.apply_permutation(layer_name, perm, 'input')
                    
                    # Check that permutation was applied
                    assert model.has_permutation(layer_name)
                    
                except Exception as e:
                    # Some layers might not support permutation
                    if "no weight" in str(e).lower():
                        pytest.skip(f"Layer {layer_name} doesn't support permutation")
                    else:
                        raise
        except Exception as e:
            if "torch_geometric" in str(e):
                pytest.skip("torch_geometric not available")
            else:
                raise
    
    def test_model_summary(self):
        """Test model summary functionality."""
        try:
            model = self.model_manager.load_model('gcn', device='cpu')
            
            summary = model.get_model_summary()
            assert 'model_name' in summary
            assert 'model_type' in summary
            assert 'total_parameters' in summary
            assert 'trainable_parameters' in summary
            assert summary['total_parameters'] >= 0
            
        except Exception as e:
            if "torch_geometric" in str(e):
                pytest.skip("torch_geometric not available")
            else:
                raise
    
    def test_error_handling(self):
        """Test error handling in integration scenarios."""
        # Test unsupported model
        with pytest.raises(ValueError, match="Model invalid-model not supported"):
            self.model_manager.load_model('invalid-model')
        
        # Test unsupported dataset
        with pytest.raises(ValueError, match="Dataset invalid-dataset not supported"):
            self.dataset_manager.get_dataset_info('invalid-dataset')
    
    def test_cache_functionality(self):
        """Test caching functionality."""
        # Test model cache
        cache_info = self.model_manager.get_cache_info()
        assert isinstance(cache_info, dict)
        assert 'cache_dir' in cache_info
        
        # Test dataset cache
        dataset_stats = self.dataset_manager.get_dataset_stats('wikitext-103')
        assert isinstance(dataset_stats, dict)
        assert 'status' in dataset_stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])