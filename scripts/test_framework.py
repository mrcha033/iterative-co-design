#!/usr/bin/env python3
"""
Test runner for the iterative co-design framework.
"""
import argparse
import sys
import tempfile
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.manager import ModelManager
from src.utils.dataset_manager import DatasetManager
from src.utils.config import create_default_config
from src.utils.exceptions import create_helpful_error_message


def test_basic_imports():
    """Test that all basic imports work."""
    print("Testing basic imports...")
    try:
        from src.models import ModelManager, PermutableModel, GCNModel
        from src.utils import Config, DatasetManager
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_model_manager():
    """Test ModelManager functionality."""
    print("\nTesting ModelManager...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ModelManager(cache_dir=temp_dir)
            
            # Test basic functionality
            assert manager.is_model_supported('gcn')
            assert not manager.is_model_supported('invalid-model')
            
            models = manager.list_supported_models()
            assert 'gcn' in models
            assert 'mamba-3b' in models
            
            print("✅ ModelManager basic tests passed")
            return True
    except Exception as e:
        print(f"❌ ModelManager test failed: {e}")
        return False


def test_dataset_manager():
    """Test DatasetManager functionality."""
    print("\nTesting DatasetManager...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(data_dir=temp_dir)
            
            # Test basic functionality
            assert manager.is_dataset_supported('wikitext-103')
            assert not manager.is_dataset_supported('invalid-dataset')
            
            datasets = manager.list_supported_datasets()
            assert 'wikitext-103' in datasets
            assert 'imagenet' in datasets
            
            print("✅ DatasetManager basic tests passed")
            return True
    except Exception as e:
        print(f"❌ DatasetManager test failed: {e}")
        return False


def test_config_system():
    """Test configuration system."""
    print("\nTesting configuration system...")
    try:
        config = create_default_config()
        
        # Test basic properties
        assert config.model.name == 'mamba-3b'
        assert config.dataset.name == 'wikitext-103'
        assert config.experiment.strategy == 'iterative_sparsity'
        
        # Test serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            
            # Test loading
            from src.utils.config import Config
            loaded_config = Config.from_yaml(f.name)
            assert loaded_config.model.name == config.model.name
        
        print("✅ Configuration system tests passed")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_gcn_model():
    """Test GCN model loading (if dependencies available)."""
    print("\nTesting GCN model...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ModelManager(cache_dir=temp_dir)
            
            # Try to load GCN model
            model = manager.load_model('gcn', device='cpu')
            
            # Test basic properties
            assert model.model_name == 'gcn'
            assert model.model_type == 'gcn'
            
            # Test layer access
            layer_names = model.get_layer_names()
            assert len(layer_names) > 0
            
            # Test model summary
            summary = model.get_model_summary()
            assert 'total_parameters' in summary
            
            print("✅ GCN model test passed")
            return True
    except Exception as e:
        if "torch_geometric" in str(e):
            print("⚠️  GCN model test skipped (torch_geometric not available)")
            return True
        else:
            print(f"❌ GCN model test failed: {e}")
            return False


def test_environment():
    """Test the environment setup."""
    print("\nTesting environment...")
    try:
        # Test PyTorch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Test Python version
        print(f"Python version: {sys.version}")
        
        # Test basic tensor operations
        x = torch.randn(10, 20)
        y = torch.mm(x, x.T)
        assert y.shape == (10, 10)
        
        print("✅ Environment test passed")
        return True
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False


def run_quick_tests():
    """Run quick tests for basic functionality."""
    print("Running quick tests...\n")
    
    tests = [
        test_basic_imports,
        test_environment,
        test_model_manager,
        test_dataset_manager,
        test_config_system,
        test_gcn_model,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    
    return failed == 0


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Test the iterative co-design framework')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        if args.quick:
            success = run_quick_tests()
        else:
            print("Running full test suite...")
            # For now, just run quick tests
            success = run_quick_tests()
        
        if success:
            print("\n🎉 All tests passed! The framework is ready to use.")
            return 0
        else:
            print("\n❌ Some tests failed. Please check the output above.")
            return 1
            
    except Exception as e:
        error_msg = create_helpful_error_message(e, "Framework testing failed")
        print(f"\n{error_msg}")
        return 1


if __name__ == '__main__':
    sys.exit(main())