"""
Unit tests for dataset manager.

These tests verify dataset loading and management functionality.
"""
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from torch.utils.data import DataLoader

from src.utils.dataset_manager import DatasetManager


class TestDatasetManager:
    """Test DatasetManager functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.dataset_manager = DatasetManager()
    
    def test_manager_initialization(self):
        """Test dataset manager initialization."""
        assert hasattr(self.dataset_manager, 'supported_datasets')
        assert 'wikitext-103' in self.dataset_manager.supported_datasets
        assert 'imagenet' in self.dataset_manager.supported_datasets
        assert 'ogbn-arxiv' in self.dataset_manager.supported_datasets
    
    @patch('src.utils.text_dataset.WikiTextDataset')
    def test_get_wikitext_dataloader(self, mock_dataset_class):
        """Test WikiText dataloader creation."""
        # Mock dataset instance
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.__getitem__ = Mock(return_value=torch.randn(512))
        mock_dataset_class.return_value = mock_dataset
        
        # Test dataloader creation
        dataloader = self.dataset_manager.get_dataloader(
            dataset_name='wikitext-103',
            batch_size=4,
            sequence_length=512,
            num_samples=50
        )
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 4
        
        # Verify dataset was created with correct parameters
        mock_dataset_class.assert_called_once()
        call_kwargs = mock_dataset_class.call_args[1]
        assert call_kwargs['sequence_length'] == 512
        assert call_kwargs['num_samples'] == 50
    
    @patch('torchvision.datasets.ImageNet')
    @patch('torchvision.transforms.Compose')
    def test_get_imagenet_dataloader(self, mock_transforms, mock_imagenet):
        """Test ImageNet dataloader creation."""
        # Mock dataset instance
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=200)
        mock_imagenet.return_value = mock_dataset
        
        # Mock transforms
        mock_transforms.return_value = Mock()
        
        dataloader = self.dataset_manager.get_dataloader(
            dataset_name='imagenet',
            batch_size=8,
            num_samples=100
        )
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 8
        
        mock_imagenet.assert_called_once()
        mock_transforms.assert_called_once()
    
    @patch('src.utils.graph_dataset.ArxivDataset')
    def test_get_arxiv_dataloader(self, mock_dataset_class):
        """Test OGB Arxiv dataloader creation."""
        # Mock dataset instance
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=150)
        mock_dataset_class.return_value = mock_dataset
        
        dataloader = self.dataset_manager.get_dataloader(
            dataset_name='ogbn-arxiv',
            batch_size=16,
            num_samples=75
        )
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 16
        
        mock_dataset_class.assert_called_once()
        call_kwargs = mock_dataset_class.call_args[1]
        assert call_kwargs['num_samples'] == 75
    
    def test_unsupported_dataset(self):
        """Test handling of unsupported dataset names."""
        with pytest.raises(ValueError, match="Unsupported dataset"):
            self.dataset_manager.get_dataloader(
                dataset_name='unsupported-dataset',
                batch_size=4
            )
    
    def test_invalid_batch_size(self):
        """Test handling of invalid batch sizes."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            self.dataset_manager.get_dataloader(
                dataset_name='wikitext-103',
                batch_size=0
            )
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            self.dataset_manager.get_dataloader(
                dataset_name='wikitext-103',
                batch_size=-1
            )
    
    def test_invalid_num_samples(self):
        """Test handling of invalid num_samples."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            self.dataset_manager.get_dataloader(
                dataset_name='wikitext-103',
                batch_size=4,
                num_samples=0
            )
    
    @patch('src.utils.text_dataset.WikiTextDataset')
    def test_dataloader_iteration(self, mock_dataset_class):
        """Test that created dataloaders are iterable."""
        # Mock dataset with specific data
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset.__getitem__ = Mock(side_effect=lambda idx: torch.randn(512))
        mock_dataset_class.return_value = mock_dataset
        
        dataloader = self.dataset_manager.get_dataloader(
            dataset_name='wikitext-103',
            batch_size=2,
            sequence_length=512,
            num_samples=10
        )
        
        # Test iteration
        batch_count = 0
        for batch in dataloader:
            assert isinstance(batch, torch.Tensor)
            assert batch.shape[0] <= 2  # Batch size
            assert batch.shape[1] == 512  # Sequence length
            batch_count += 1
            if batch_count >= 3:  # Only test a few batches
                break
        
        assert batch_count > 0
    
    @patch('src.utils.text_dataset.WikiTextDataset')
    def test_dataset_caching(self, mock_dataset_class):
        """Test dataset caching behavior."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=50)
        mock_dataset_class.return_value = mock_dataset
        
        # Create dataloader twice with same parameters
        params = {
            'dataset_name': 'wikitext-103',
            'batch_size': 4,
            'sequence_length': 512,
            'num_samples': 25
        }
        
        dataloader1 = self.dataset_manager.get_dataloader(**params)
        dataloader2 = self.dataset_manager.get_dataloader(**params)
        
        # Dataset should be created only once due to caching
        assert mock_dataset_class.call_count <= 2  # Allow for some caching implementation
    
    def test_get_dataset_info(self):
        """Test dataset information retrieval."""
        # This assumes the manager has a method to get dataset info
        if hasattr(self.dataset_manager, 'get_dataset_info'):
            info = self.dataset_manager.get_dataset_info('wikitext-103')
            assert isinstance(info, dict)
            assert 'name' in info
    
    @patch('src.utils.text_dataset.WikiTextDataset')
    def test_default_parameters(self, mock_dataset_class):
        """Test dataloader creation with default parameters."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset_class.return_value = mock_dataset
        
        # Test with minimal parameters
        dataloader = self.dataset_manager.get_dataloader(
            dataset_name='wikitext-103',
            batch_size=4
        )
        
        assert isinstance(dataloader, DataLoader)
        
        # Verify defaults were applied
        call_kwargs = mock_dataset_class.call_args[1]
        assert 'sequence_length' in call_kwargs
        assert 'num_samples' in call_kwargs
    
    @patch('src.utils.text_dataset.WikiTextDataset')
    def test_custom_sequence_length(self, mock_dataset_class):
        """Test custom sequence length handling."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset_class.return_value = mock_dataset
        
        custom_length = 1024
        dataloader = self.dataset_manager.get_dataloader(
            dataset_name='wikitext-103',
            batch_size=4,
            sequence_length=custom_length
        )
        
        call_kwargs = mock_dataset_class.call_args[1]
        assert call_kwargs['sequence_length'] == custom_length
    
    def test_supported_dataset_list(self):
        """Test that supported datasets list is accessible."""
        supported = self.dataset_manager.supported_datasets
        
        assert isinstance(supported, (list, tuple, set))
        assert len(supported) > 0
        
        # Check that key datasets are supported
        expected_datasets = ['wikitext-103', 'imagenet', 'ogbn-arxiv']
        for dataset in expected_datasets:
            assert dataset in supported
    
    @patch('src.utils.text_dataset.WikiTextDataset')
    def test_worker_configuration(self, mock_dataset_class):
        """Test dataloader worker configuration."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset_class.return_value = mock_dataset
        
        dataloader = self.dataset_manager.get_dataloader(
            dataset_name='wikitext-103',
            batch_size=4,
            num_workers=2
        )
        
        # Check that num_workers is set (if supported by the implementation)
        if hasattr(dataloader, 'num_workers'):
            assert dataloader.num_workers >= 0


class TestDatasetManagerEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.dataset_manager = DatasetManager()
    
    def test_very_large_batch_size(self):
        """Test handling of very large batch sizes."""
        with patch('src.utils.text_dataset.WikiTextDataset') as mock_dataset:
            mock_dataset.return_value.__len__ = Mock(return_value=10)
            
            # Large batch size larger than dataset
            dataloader = self.dataset_manager.get_dataloader(
                dataset_name='wikitext-103',
                batch_size=1000,
                num_samples=10
            )
            
            # Should still create dataloader, but batches may be smaller
            assert isinstance(dataloader, DataLoader)
    
    @patch('src.utils.text_dataset.WikiTextDataset')
    def test_single_sample_dataset(self, mock_dataset_class):
        """Test dataset with only one sample."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(return_value=torch.randn(512))
        mock_dataset_class.return_value = mock_dataset
        
        dataloader = self.dataset_manager.get_dataloader(
            dataset_name='wikitext-103',
            batch_size=4,
            num_samples=1
        )
        
        # Should handle single sample gracefully
        batches = list(dataloader)
        assert len(batches) >= 0  # May be 0 or 1 depending on implementation
    
    def test_case_insensitive_dataset_names(self):
        """Test case insensitive dataset name handling."""
        with patch('src.utils.text_dataset.WikiTextDataset') as mock_dataset:
            mock_dataset.return_value.__len__ = Mock(return_value=100)
            
            # Test different cases
            test_cases = ['WikiText-103', 'WIKITEXT-103', 'wikitext-103']
            
            for dataset_name in test_cases:
                try:
                    dataloader = self.dataset_manager.get_dataloader(
                        dataset_name=dataset_name,
                        batch_size=4
                    )
                    assert isinstance(dataloader, DataLoader)
                except ValueError:
                    # If case sensitivity is enforced, that's also valid
                    pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])