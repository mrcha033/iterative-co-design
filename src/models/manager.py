"""
Model Manager for loading and managing different model architectures.
"""
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .permutable_model import PermutableModel


class ModelManager:
    """
    Manages model loading, caching, and wrapping for the iterative co-design framework.
    Supports Mamba-3B, BERT-large, ResNet-50, and GCN architectures.
    """
    
    SUPPORTED_MODELS = {
        'mamba-3b': {
            'hf_id': 'state-spaces/mamba-3b',
            'type': 'mamba',
            'requires_transformers': True,
        },
        'bert-large': {
            'hf_id': 'bert-large-uncased',
            'type': 'bert',
            'requires_transformers': True,
        },
        'resnet-50': {
            'hf_id': 'microsoft/resnet-50',
            'type': 'resnet',
            'requires_transformers': True,
        },
        'gcn': {
            'hf_id': None,  # Custom implementation
            'type': 'gcn',
            'requires_transformers': False,
        }
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the ModelManager.
        
        Args:
            cache_dir: Directory to cache downloaded models. If None, uses default.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'iterative-co-design'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models: Dict[str, nn.Module] = {}
        
    def is_model_supported(self, model_name: str) -> bool:
        """Check if a model is supported by the framework."""
        return model_name.lower() in self.SUPPORTED_MODELS
    
    def list_supported_models(self) -> List[str]:
        """Get list of supported model names."""
        return list(self.SUPPORTED_MODELS.keys())
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a supported model."""
        if not self.is_model_supported(model_name):
            raise ValueError(f"Model {model_name} not supported. "
                           f"Supported models: {self.list_supported_models()}")
        return self.SUPPORTED_MODELS[model_name.lower()]
    
    def load_model(
        self,
        model_name: str,
        precision: str = 'float16',
        device: str = 'cuda',
        pretrained_path: Optional[str] = None,
        force_download: bool = False
    ) -> PermutableModel:
        """
        Load and wrap a model for the iterative co-design framework.
        
        Args:
            model_name: Name of the model to load
            precision: Model precision ('float32', 'float16', 'bfloat16')
            device: Device to load model on ('cuda', 'cpu')
            pretrained_path: Path to local pretrained model (optional)
            force_download: Force re-download even if cached
            
        Returns:
            PermutableModel: Wrapped model ready for co-design
        """
        if not self.is_model_supported(model_name):
            raise ValueError(f"Model {model_name} not supported. "
                           f"Supported models: {self.list_supported_models()}")
        
        # Use cache key to avoid reloading
        cache_key = f"{model_name}_{precision}_{device}"
        if cache_key in self._loaded_models and not force_download:
            return self._loaded_models[cache_key]
        
        model_info = self.get_model_info(model_name)
        
        # Load the raw model
        if pretrained_path:
            raw_model = self._load_from_path(pretrained_path, model_info)
        else:
            raw_model = self._load_from_hub(model_name, model_info, force_download)
        
        # Set precision and device
        raw_model = self._set_precision(raw_model, precision)
        raw_model = raw_model.to(device)
        
        # Wrap in PermutableModel
        wrapped_model = PermutableModel(
            model=raw_model,
            model_type=model_info['type'],
            model_name=model_name
        )
        
        # Cache the wrapped model
        self._loaded_models[cache_key] = wrapped_model
        
        return wrapped_model
    
    def _load_from_hub(self, model_name: str, model_info: Dict, force_download: bool) -> nn.Module:
        """Load model from Hugging Face Hub."""
        hf_id = model_info.get('hf_id')
        if not hf_id:
            return self._load_custom_model(model_name, model_info)
        
        if model_info['requires_transformers']:
            try:
                from transformers import AutoModel, AutoConfig
                
                # Download to cache directory
                cache_path = self.cache_dir / model_name
                if force_download or not cache_path.exists():
                    print(f"Downloading {model_name} from Hugging Face Hub...")
                    config = AutoConfig.from_pretrained(hf_id)
                    model = AutoModel.from_pretrained(
                        hf_id,
                        cache_dir=str(cache_path),
                        torch_dtype=torch.float32,  # We'll set precision later
                        trust_remote_code=True
                    )
                else:
                    print(f"Loading {model_name} from cache...")
                    model = AutoModel.from_pretrained(
                        str(cache_path),
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                
                return model
                
            except ImportError:
                raise ImportError("transformers library is required for this model. "
                                "Install with: pip install transformers")
        else:
            raise NotImplementedError(f"Loading {model_name} not yet implemented")
    
    def _load_from_path(self, path: str, model_info: Dict) -> nn.Module:
        """Load model from local path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model path {path} does not exist")
        
        if path.suffix == '.pt' or path.suffix == '.pth':
            # PyTorch checkpoint
            checkpoint = torch.load(path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Create model architecture and load weights
            model = self._create_model_architecture(model_info)
            model.load_state_dict(state_dict)
            return model
        else:
            # Try loading as transformers model
            try:
                from transformers import AutoModel
                return AutoModel.from_pretrained(str(path), trust_remote_code=True)
            except Exception as e:
                raise ValueError(f"Could not load model from {path}: {e}")
    
    def _load_custom_model(self, model_name: str, model_info: Dict) -> nn.Module:
        """Load custom model implementations."""
        if model_name == 'gcn':
            # Load GCN model for graph tasks
            from .gcn_model import GCNModel
            return GCNModel()
        else:
            raise NotImplementedError(f"Custom model {model_name} not implemented")
    
    def _create_model_architecture(self, model_info: Dict) -> nn.Module:
        """Create empty model architecture for loading checkpoints."""
        model_type = model_info['type']
        
        if model_type == 'mamba':
            # Create Mamba architecture
            raise NotImplementedError("Mamba architecture creation not implemented")
        elif model_type == 'bert':
            from transformers import BertModel, BertConfig
            config = BertConfig.from_pretrained('bert-large-uncased')
            return BertModel(config)
        elif model_type == 'resnet':
            from transformers import ResNetModel, ResNetConfig
            config = ResNetConfig.from_pretrained('microsoft/resnet-50')
            return ResNetModel(config)
        elif model_type == 'gcn':
            from .gcn_model import GCNModel
            return GCNModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _set_precision(self, model: nn.Module, precision: str) -> nn.Module:
        """Set model precision."""
        if precision == 'float32':
            return model.float()
        elif precision == 'float16':
            return model.half()
        elif precision == 'bfloat16':
            return model.bfloat16()
        else:
            raise ValueError(f"Unsupported precision: {precision}")
    
    def get_model_layers(self, model_name: str) -> List[str]:
        """
        Get list of available layers for a model.
        Useful for error messages and layer selection.
        """
        if not self.is_model_supported(model_name):
            raise ValueError(f"Model {model_name} not supported")
        
        # Load model if not already loaded
        if model_name not in self._loaded_models:
            wrapped_model = self.load_model(model_name)
        else:
            wrapped_model = self._loaded_models[model_name]
        
        return wrapped_model.get_layer_names()
    
    def validate_layer_name(self, model_name: str, layer_name: str) -> bool:
        """
        Validate that a layer name exists in the model.
        
        Args:
            model_name: Name of the model
            layer_name: Name of the layer to validate
            
        Returns:
            bool: True if layer exists, False otherwise
        """
        try:
            available_layers = self.get_model_layers(model_name)
            return layer_name in available_layers
        except Exception:
            return False
    
    def clear_cache(self):
        """Clear the model cache."""
        self._loaded_models.clear()
    
    def get_cache_info(self) -> Dict:
        """Get information about cached models."""
        return {
            'cache_dir': str(self.cache_dir),
            'loaded_models': list(self._loaded_models.keys()),
            'cache_size_mb': sum(
                f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file()
            ) / (1024 * 1024) if self.cache_dir.exists() else 0
        }