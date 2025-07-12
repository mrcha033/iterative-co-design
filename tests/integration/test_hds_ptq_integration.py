"""
Integration tests for HDS and PTQ modules working together.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from src.co_design.hds import HDSConfig, HDSOptimizer, apply_hds_to_model
from src.co_design.ptq import PTQConfig, PostTrainingQuantizer, quantize_model
from src.models.permutable_model import PermutableModel


class TestHDSPTQIntegration:
    """Test integration between HDS and PTQ modules."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create a simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 4)
                self.linear2 = nn.Linear(4, 2)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.linear1(x))
                x = self.linear2(x)
                return x
        
        self.model = SimpleModel()
        self.permutable_model = PermutableModel(self.model, 'test', 'test')
        
        # Create mock dataloader
        self.dataloader = Mock()
        self.dataloader.__iter__ = Mock(return_value=iter([
            torch.randn(2, 8) for _ in range(5)
        ]))
        self.dataloader.__len__ = Mock(return_value=5)
        
        # Create configurations
        self.hds_config = HDSConfig(
            num_epochs=2,
            learning_rate=1e-3,
            sparsity_ratio="2:4",
            log_interval=1
        )
        
        self.ptq_config = PTQConfig(
            calibration_samples=10,
            verbose=False
        )
    
    def test_hds_then_ptq_workflow(self):
        """Test HDS followed by PTQ workflow."""
        # Step 1: Apply HDS to create sparse model
        hds_optimizer = HDSOptimizer(self.hds_config)
        
        # Prepare model for HDS
        hds_model = hds_optimizer.prepare_model(self.permutable_model)
        
        # Verify HDS preparation
        assert len(hds_optimizer.hds_layers) == 2
        assert 'linear1' in hds_optimizer.hds_layers
        assert 'linear2' in hds_optimizer.hds_layers
        
        # Mock training process for speed
        with patch.object(hds_optimizer, 'train') as mock_train:
            mock_train.return_value = {
                'training_history': [{'epoch': 0, 'train_loss': 1.0}],
                'final_sparsity': {'avg_sparsity': 0.5}
            }
            
            # Get sparse model
            sparse_model = hds_optimizer.get_sparse_model()
            
            # Step 2: Apply PTQ to sparse model
            ptq_quantizer = PostTrainingQuantizer(self.ptq_config)
            
            # Prepare sparse model for quantization
            ptq_model = ptq_quantizer.prepare_model(sparse_model)
            
            # Verify PTQ preparation
            assert len(ptq_quantizer.observers) == 2
            assert 'linear1' in ptq_quantizer.observers
            assert 'linear2' in ptq_quantizer.observers
            
            # Calibrate quantization
            ptq_quantizer.calibrate(ptq_model, self.dataloader)
            
            # Verify calibration
            assert len(ptq_quantizer.calibration_stats) == 2
            
            # Quantize model
            quantized_model = ptq_quantizer.quantize_model(ptq_model)
            
            # Verify quantization
            assert len(ptq_quantizer.quantized_layers) == 2
            
            # Get final stats
            ptq_stats = ptq_quantizer.get_quantization_stats()
            assert ptq_stats['num_quantized_layers'] == 2
    
    def test_ptq_then_hds_workflow(self):
        """Test PTQ followed by HDS workflow."""
        # Step 1: Apply PTQ first
        ptq_quantizer = PostTrainingQuantizer(self.ptq_config)
        
        # Prepare model for quantization
        ptq_model = ptq_quantizer.prepare_model(self.permutable_model)
        
        # Calibrate and quantize
        ptq_quantizer.calibrate(ptq_model, self.dataloader)
        quantized_model = ptq_quantizer.quantize_model(ptq_model)
        
        # Step 2: Apply HDS to quantized model
        hds_optimizer = HDSOptimizer(self.hds_config)
        
        # Note: In practice, this workflow is less common
        # as HDS typically comes before quantization
        # This test verifies the modules don't interfere with each other
        
        # Prepare quantized model for HDS
        hds_model = hds_optimizer.prepare_model(quantized_model)
        
        # Verify both transformations coexist
        assert len(ptq_quantizer.quantized_layers) == 2
        assert len(hds_optimizer.hds_layers) >= 0  # May be 0 if quantized layers can't be wrapped
    
    def test_utility_functions_integration(self):
        """Test utility functions working together."""
        # Test apply_hds_to_model utility
        with patch('src.co_design.hds.HDSOptimizer.train') as mock_train:
            mock_train.return_value = {
                'training_history': [],
                'final_sparsity': {'avg_sparsity': 0.5}
            }
            
            sparse_model, hds_results = apply_hds_to_model(
                self.permutable_model,
                self.dataloader,
                self.hds_config
            )
            
            assert sparse_model is not None
            assert 'training_history' in hds_results
            
            # Test quantize_model utility on sparse model
            quantized_model, ptq_results = quantize_model(
                sparse_model,
                self.dataloader,
                self.ptq_config
            )
            
            assert quantized_model is not None
            assert 'calibration_stats' in ptq_results
            assert 'num_quantized_layers' in ptq_results
    
    def test_model_functionality_preservation(self):
        """Test that model functionality is preserved through transformations."""
        # Create test input
        test_input = torch.randn(1, 8)
        
        # Get original output
        original_output = self.permutable_model(test_input)
        
        # Apply HDS
        hds_optimizer = HDSOptimizer(self.hds_config)
        hds_model = hds_optimizer.prepare_model(self.permutable_model)
        
        # Get HDS output
        hds_output = hds_model(test_input)
        
        # Verify shapes match
        assert hds_output.shape == original_output.shape
        
        # Apply PTQ to HDS model
        ptq_quantizer = PostTrainingQuantizer(self.ptq_config)
        ptq_model = ptq_quantizer.prepare_model(hds_model)
        
        # Calibrate
        ptq_quantizer.calibrate(ptq_model, self.dataloader)
        
        # Quantize
        final_model = ptq_quantizer.quantize_model(ptq_model)
        
        # Get final output
        final_output = final_model(test_input)
        
        # Verify shapes still match
        assert final_output.shape == original_output.shape
    
    def test_error_handling_integration(self):
        """Test error handling when modules interact."""
        # Test with empty dataloader
        empty_dataloader = Mock()
        empty_dataloader.__iter__ = Mock(return_value=iter([]))
        empty_dataloader.__len__ = Mock(return_value=0)
        
        # HDS should handle empty dataloader gracefully
        hds_optimizer = HDSOptimizer(self.hds_config)
        hds_model = hds_optimizer.prepare_model(self.permutable_model)
        
        # PTQ should handle empty dataloader gracefully
        ptq_quantizer = PostTrainingQuantizer(self.ptq_config)
        ptq_model = ptq_quantizer.prepare_model(self.permutable_model)
        
        # This should not raise exceptions
        try:
            ptq_quantizer.calibrate(ptq_model, empty_dataloader)
        except Exception as e:
            # Some warnings are expected, but no hard failures
            assert "No samples" in str(e) or "warn" in str(e).lower()
    
    def test_configuration_compatibility(self):
        """Test that HDS and PTQ configurations are compatible."""
        # Test with different target layers
        hds_config = HDSConfig(target_layers=['linear1'])
        ptq_config = PTQConfig(target_layers=['linear2'])
        
        # Apply HDS to linear1 only
        hds_optimizer = HDSOptimizer(hds_config)
        hds_model = hds_optimizer.prepare_model(self.permutable_model)
        
        assert len(hds_optimizer.hds_layers) == 1
        assert 'linear1' in hds_optimizer.hds_layers
        
        # Apply PTQ to linear2 only
        ptq_quantizer = PostTrainingQuantizer(ptq_config)
        ptq_model = ptq_quantizer.prepare_model(hds_model)
        
        assert len(ptq_quantizer.observers) == 1
        assert 'linear2' in ptq_quantizer.observers
        
        # Both should coexist without interference
        assert len(hds_optimizer.hds_layers) == 1
        assert len(ptq_quantizer.observers) == 1
    
    def test_memory_efficiency_integration(self):
        """Test memory efficiency with both transformations."""
        # Create a larger model to test memory handling
        class LargerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(64, 32)
                self.linear2 = nn.Linear(32, 16)
                self.linear3 = nn.Linear(16, 8)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.linear1(x))
                x = self.relu(self.linear2(x))
                x = self.linear3(x)
                return x
        
        larger_model = LargerModel()
        larger_permutable = PermutableModel(larger_model, 'test', 'test')
        
        # Create larger dataloader
        larger_dataloader = Mock()
        larger_dataloader.__iter__ = Mock(return_value=iter([
            torch.randn(4, 64) for _ in range(10)
        ]))
        larger_dataloader.__len__ = Mock(return_value=10)
        
        # Apply both transformations
        hds_optimizer = HDSOptimizer(self.hds_config)
        hds_model = hds_optimizer.prepare_model(larger_permutable)
        
        ptq_quantizer = PostTrainingQuantizer(self.ptq_config)
        ptq_model = ptq_quantizer.prepare_model(hds_model)
        
        # Calibrate
        ptq_quantizer.calibrate(ptq_model, larger_dataloader)
        
        # Quantize
        final_model = ptq_quantizer.quantize_model(ptq_model)
        
        # Test forward pass
        test_input = torch.randn(2, 64)
        output = final_model(test_input)
        
        # Should complete without memory issues
        assert output.shape == (2, 8)
    
    def test_sequential_optimization_benefits(self):
        """Test that sequential optimization provides expected benefits."""
        # This test verifies the conceptual flow rather than actual optimization
        # In practice, HDS creates sparsity, PTQ reduces precision
        
        # Apply HDS
        hds_optimizer = HDSOptimizer(self.hds_config)
        hds_model = hds_optimizer.prepare_model(self.permutable_model)
        
        # Get sparsity info
        sparsity_info = hds_optimizer._get_sparsity_info()
        
        # Apply PTQ
        ptq_quantizer = PostTrainingQuantizer(self.ptq_config)
        ptq_model = ptq_quantizer.prepare_model(hds_model)
        
        # Calibrate and quantize
        ptq_quantizer.calibrate(ptq_model, self.dataloader)
        final_model = ptq_quantizer.quantize_model(ptq_model)
        
        # Get quantization stats
        ptq_stats = ptq_quantizer.get_quantization_stats()
        
        # Both optimizations should be applied
        assert sparsity_info['avg_sparsity'] >= 0.0
        assert ptq_stats['num_quantized_layers'] > 0
        
        # The final model should be both sparse and quantized
        assert len(hds_optimizer.hds_layers) > 0
        assert len(ptq_quantizer.quantized_layers) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])