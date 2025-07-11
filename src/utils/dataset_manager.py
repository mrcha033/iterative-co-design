"""
Dataset management utilities for downloading and loading datasets.
"""
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig


class DatasetManager:
    """
    Manages dataset downloading, caching, and loading for the iterative co-design framework.
    Supports WikiText-103, ImageNet, and ogbn-arxiv datasets.
    """
    
    SUPPORTED_DATASETS = {
        'wikitext-103': {
            'type': 'text',
            'task': 'language_modeling',
            'loader': 'torchtext',
            'splits': ['train', 'valid', 'test'],
        },
        'imagenet': {
            'type': 'vision',
            'task': 'image_classification',
            'loader': 'torchvision',
            'splits': ['train', 'val'],
        },
        'ogbn-arxiv': {
            'type': 'graph',
            'task': 'node_classification',
            'loader': 'ogb',
            'splits': ['train', 'valid', 'test'],
        }
    }
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the DatasetManager.
        
        Args:
            data_dir: Directory to store datasets. If None, uses default.
        """
        self.data_dir = Path(data_dir) if data_dir else Path('./data')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_datasets: Dict[str, Dataset] = {}
    
    def is_dataset_supported(self, dataset_name: str) -> bool:
        """Check if a dataset is supported."""
        return dataset_name.lower() in self.SUPPORTED_DATASETS
    
    def list_supported_datasets(self) -> list:
        """Get list of supported dataset names."""
        return list(self.SUPPORTED_DATASETS.keys())
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get information about a supported dataset."""
        if not self.is_dataset_supported(dataset_name):
            raise ValueError(f"Dataset {dataset_name} not supported. "
                           f"Supported datasets: {self.list_supported_datasets()}")
        return self.SUPPORTED_DATASETS[dataset_name.lower()]
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> Path:
        """
        Download a dataset if not already cached.
        
        Args:
            dataset_name: Name of the dataset to download
            force_download: Force re-download even if cached
            
        Returns:
            Path to the downloaded dataset
        """
        if not self.is_dataset_supported(dataset_name):
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        dataset_info = self.get_dataset_info(dataset_name)
        dataset_path = self.data_dir / dataset_name
        
        if dataset_path.exists() and not force_download:
            print(f"Dataset {dataset_name} already exists at {dataset_path}")
            return dataset_path
        
        print(f"Downloading {dataset_name}...")
        
        if dataset_info['loader'] == 'torchtext':
            return self._download_torchtext_dataset(dataset_name, dataset_path)
        elif dataset_info['loader'] == 'torchvision':
            return self._download_torchvision_dataset(dataset_name, dataset_path)
        elif dataset_info['loader'] == 'ogb':
            return self._download_ogb_dataset(dataset_name, dataset_path)
        else:
            raise NotImplementedError(f"Loader {dataset_info['loader']} not implemented")
    
    def _download_torchtext_dataset(self, dataset_name: str, dataset_path: Path) -> Path:
        """Download datasets using torchtext."""
        try:
            from torchtext.datasets import WikiText103
            
            if dataset_name == 'wikitext-103':
                # Download WikiText-103
                train_iter, valid_iter, test_iter = WikiText103(
                    root=str(dataset_path.parent),
                    split=('train', 'valid', 'test')
                )
                
                # Save iterators to files for later use
                dataset_path.mkdir(parents=True, exist_ok=True)
                
                # Convert iterators to text files
                for split_name, iterator in [('train', train_iter), ('valid', valid_iter), ('test', test_iter)]:
                    split_file = dataset_path / f"{split_name}.txt"
                    with open(split_file, 'w', encoding='utf-8') as f:
                        for line in iterator:
                            f.write(line + '\n')
                
                return dataset_path
            else:
                raise NotImplementedError(f"torchtext dataset {dataset_name} not implemented")
                
        except ImportError:
            raise ImportError("torchtext is required for text datasets. Install with: pip install torchtext")
    
    def _download_torchvision_dataset(self, dataset_name: str, dataset_path: Path) -> Path:
        """Download datasets using torchvision."""
        try:
            from torchvision import datasets, transforms
            
            if dataset_name == 'imagenet':
                # ImageNet requires manual download and setup
                # We'll create a placeholder structure
                dataset_path.mkdir(parents=True, exist_ok=True)
                
                # Create README with instructions
                readme_path = dataset_path / 'README.md'
                with open(readme_path, 'w') as f:
                    f.write("""# ImageNet Dataset
                    
ImageNet dataset requires manual download due to licensing restrictions.

## Setup Instructions:
1. Download ImageNet from https://image-net.org/
2. Extract the dataset to this directory
3. Ensure the structure is:
   - train/
   - val/
   - test/ (optional)

## Expected Structure:
```
imagenet/
├── train/
│   ├── n01440764/
│   └── ...
├── val/
│   ├── n01440764/
│   └── ...
└── README.md
```
""")
                
                print(f"ImageNet requires manual setup. See {readme_path} for instructions.")
                return dataset_path
            else:
                raise NotImplementedError(f"torchvision dataset {dataset_name} not implemented")
                
        except ImportError:
            raise ImportError("torchvision is required for vision datasets. Install with: pip install torchvision")
    
    def _download_ogb_dataset(self, dataset_name: str, dataset_path: Path) -> Path:
        """Download datasets using OGB."""
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
            
            if dataset_name == 'ogbn-arxiv':
                # Download ogbn-arxiv dataset
                dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=str(dataset_path.parent))
                
                # The dataset is automatically downloaded and cached
                return dataset_path.parent / 'ogbn_arxiv'
            else:
                raise NotImplementedError(f"OGB dataset {dataset_name} not implemented")
                
        except ImportError:
            raise ImportError("ogb is required for graph datasets. Install with: pip install ogb")
    
    def load_dataset(
        self,
        dataset_name: str,
        split: str = 'train',
        batch_size: int = 1,
        sequence_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        shuffle: bool = True,
        **kwargs
    ) -> DataLoader:
        """
        Load a dataset and return a DataLoader.
        
        Args:
            dataset_name: Name of the dataset
            split: Dataset split ('train', 'valid', 'test')
            batch_size: Batch size for DataLoader
            sequence_length: Sequence length for text datasets
            num_samples: Limit number of samples (for debugging)
            shuffle: Whether to shuffle the dataset
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            DataLoader for the dataset
        """
        if not self.is_dataset_supported(dataset_name):
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        # Ensure dataset is downloaded
        dataset_path = self.download_dataset(dataset_name)
        
        # Load dataset based on type
        dataset_info = self.get_dataset_info(dataset_name)
        
        if dataset_info['type'] == 'text':
            dataset = self._load_text_dataset(dataset_name, dataset_path, split, sequence_length, num_samples)
        elif dataset_info['type'] == 'vision':
            dataset = self._load_vision_dataset(dataset_name, dataset_path, split, num_samples)
        elif dataset_info['type'] == 'graph':
            dataset = self._load_graph_dataset(dataset_name, dataset_path, split, num_samples)
        else:
            raise NotImplementedError(f"Dataset type {dataset_info['type']} not implemented")
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
        
        return dataloader
    
    def _load_text_dataset(self, dataset_name: str, dataset_path: Path, split: str, sequence_length: Optional[int], num_samples: Optional[int]) -> Dataset:
        """Load text dataset."""
        from .text_dataset import TextDataset
        
        split_file = dataset_path / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file {split_file} not found")
        
        return TextDataset(
            file_path=str(split_file),
            sequence_length=sequence_length or 1024,
            num_samples=num_samples
        )
    
    def _load_vision_dataset(self, dataset_name: str, dataset_path: Path, split: str, num_samples: Optional[int]) -> Dataset:
        """Load vision dataset."""
        try:
            from torchvision import datasets, transforms
            
            if dataset_name == 'imagenet':
                # Check if ImageNet is properly set up
                split_path = dataset_path / split
                if not split_path.exists():
                    raise FileNotFoundError(f"ImageNet {split} split not found at {split_path}. "
                                          f"Please follow setup instructions in {dataset_path}/README.md")
                
                # Define transforms
                if split == 'train':
                    transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                
                dataset = datasets.ImageFolder(str(split_path), transform=transform)
                
                # Limit samples if requested
                if num_samples:
                    dataset = torch.utils.data.Subset(dataset, range(min(num_samples, len(dataset))))
                
                return dataset
            else:
                raise NotImplementedError(f"Vision dataset {dataset_name} not implemented")
                
        except ImportError:
            raise ImportError("torchvision is required for vision datasets")
    
    def _load_graph_dataset(self, dataset_name: str, dataset_path: Path, split: str, num_samples: Optional[int]) -> Dataset:
        """Load graph dataset."""
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
            from .graph_dataset import GraphDataset
            
            if dataset_name == 'ogbn-arxiv':
                # Load OGB dataset
                dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=str(dataset_path.parent))
                
                # Wrap in our custom dataset class
                return GraphDataset(
                    dataset=dataset,
                    split=split,
                    num_samples=num_samples
                )
            else:
                raise NotImplementedError(f"Graph dataset {dataset_name} not implemented")
                
        except ImportError:
            raise ImportError("ogb and torch-geometric are required for graph datasets")
    
    def get_dataset_stats(self, dataset_name: str) -> Dict:
        """Get statistics about a dataset."""
        if not self.is_dataset_supported(dataset_name):
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        dataset_path = self.data_dir / dataset_name
        
        if not dataset_path.exists():
            return {'status': 'not_downloaded', 'size_mb': 0}
        
        # Calculate size
        size_bytes = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
        size_mb = size_bytes / (1024 * 1024)
        
        return {
            'status': 'downloaded',
            'path': str(dataset_path),
            'size_mb': round(size_mb, 2),
            'num_files': len(list(dataset_path.rglob('*')))
        }
    
    def clear_cache(self, dataset_name: Optional[str] = None):
        """Clear dataset cache."""
        if dataset_name:
            dataset_path = self.data_dir / dataset_name
            if dataset_path.exists():
                import shutil
                shutil.rmtree(dataset_path)
                print(f"Cleared cache for {dataset_name}")
        else:
            # Clear all datasets
            if self.data_dir.exists():
                import shutil
                shutil.rmtree(self.data_dir)
                self.data_dir.mkdir(parents=True, exist_ok=True)
                print("Cleared all dataset cache")