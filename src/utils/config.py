"""
Configuration loading utilities for YAML files.
Implements Task 1.3 from tasks.md.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Type, TypeVar
from dataclasses import dataclass, field, asdict, is_dataclass
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)

# Type variable for generic dataclass types
T = TypeVar('T')


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return as a dictionary.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration data

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML is malformed
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def load_omegaconf(config_path: Union[str, Path]) -> DictConfig:
    """
    Load a YAML configuration file using OmegaConf for Hydra compatibility.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        OmegaConf DictConfig object

    Raises:
        FileNotFoundError: If the config file doesn't exist
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    return OmegaConf.load(config_path)


def merge_configs(*config_paths: Union[str, Path]) -> DictConfig:
    """
    Load and merge multiple YAML configuration files.
    Later configs override earlier ones.

    Args:
        *config_paths: Variable number of paths to YAML files

    Returns:
        Merged OmegaConf DictConfig object
    """
    configs = []

    for path in config_paths:
        configs.append(load_omegaconf(path))

    # Merge all configs (later ones override earlier ones)
    merged = OmegaConf.merge(*configs)

    return merged


def save_config(
    config: Union[Dict[str, Any], DictConfig], output_path: Union[str, Path]
) -> None:
    """
    Save a configuration to a YAML file.

    Args:
        config: Configuration data (dict or DictConfig)
        output_path: Path where to save the YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(config, DictConfig):
        # Use OmegaConf for DictConfig
        with open(output_path, "w", encoding="utf-8") as f:
            OmegaConf.save(config, f)
    else:
        # Use standard YAML for regular dicts
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, indent=2)


def config_to_namespace(config: Dict[str, Any]) -> object:
    """
    Convert a configuration dictionary to an argparse.Namespace-like object.

    Args:
        config: Configuration dictionary

    Returns:
        Object with configuration keys as attributes
    """

    class ConfigNamespace:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigNamespace(**value))
                else:
                    setattr(self, key, value)

        def __repr__(self):
            attrs = []
            for key, value in self.__dict__.items():
                attrs.append(f"{key}={value}")
            return f"ConfigNamespace({', '.join(attrs)})"

    return ConfigNamespace(**config)


def validate_config(config: Union[Dict[str, Any], DictConfig], 
                   required_fields: Optional[List[str]] = None) -> bool:
    """
    Validate configuration structure and values.
    
    Args:
        config: Configuration to validate
        required_fields: List of required top-level fields
        
    Returns:
        True if valid, raises ValueError otherwise
        
    Raises:
        ValueError: If configuration is invalid
    """
    if required_fields is None:
        # Default required fields for experiments
        required_fields = ['model', 'dataset']
    
    # Convert DictConfig to dict for easier validation
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config
    
    # Check required fields
    for field in required_fields:
        if field not in config_dict:
            raise ValueError(f"Required field '{field}' missing in configuration")
    
    # Validate specific field types and values
    if 'seed' in config_dict and not isinstance(config_dict['seed'], int):
        raise ValueError("'seed' must be an integer")
        
    if 'num_iterations' in config_dict:
        if not isinstance(config_dict['num_iterations'], int) or config_dict['num_iterations'] < 1:
            raise ValueError("'num_iterations' must be a positive integer")
    
    # Validate model configuration
    if 'model' in config_dict:
        model_config = config_dict['model']
        if 'task' in model_config and model_config['task'] not in ['language_modeling', 'sequence_classification']:
            raise ValueError(f"Invalid model task: {model_config['task']}")
    
    # Validate dataset configuration
    if 'dataset' in config_dict:
        dataset_config = config_dict['dataset']
        if 'batch_size' in dataset_config:
            if not isinstance(dataset_config['batch_size'], int) or dataset_config['batch_size'] < 1:
                raise ValueError("'dataset.batch_size' must be a positive integer")
    
    # Validate IASP configuration
    if 'iasp' in config_dict:
        iasp_config = config_dict['iasp']
        
        # CRITICAL: Ensure permute_gate is False or not set
        if 'permute_gate' in iasp_config and iasp_config['permute_gate'] is not False:
            raise ValueError(
                "IASP configuration error: 'permute_gate' must be False. "
                "Gate permutation is explicitly forbidden for safety reasons."
            )
        
        # Validate cluster size range
        if 'cluster_size_range' in iasp_config:
            csr = iasp_config['cluster_size_range']
            if not isinstance(csr, (list, tuple)) or len(csr) != 2:
                raise ValueError("'iasp.cluster_size_range' must be a list/tuple of 2 integers")
            if csr[0] >= csr[1]:
                raise ValueError("'iasp.cluster_size_range' must have min < max")
            if csr[0] < 1:
                raise ValueError("'iasp.cluster_size_range' minimum must be positive")
        
        # Validate max_samples
        if 'max_samples' in iasp_config:
            if not isinstance(iasp_config['max_samples'], int) or iasp_config['max_samples'] < 1:
                raise ValueError("'iasp.max_samples' must be a positive integer")
        
        # Validate rollback parameters if present
        if 'max_ppl_increase' in iasp_config:
            max_ppl = iasp_config['max_ppl_increase']
            if not isinstance(max_ppl, (int, float)) or max_ppl <= 0 or max_ppl > 1:
                raise ValueError("'iasp.max_ppl_increase' must be between 0 and 1")
    
    # Validate HDS configuration
    if 'hds' in config_dict:
        hds_config = config_dict['hds']
        
        # Validate N:M sparsity pattern
        if 'n' in hds_config and 'm' in hds_config:
            n, m = hds_config['n'], hds_config['m']
            if not (isinstance(n, int) and isinstance(m, int)):
                raise ValueError("'hds.n' and 'hds.m' must be integers")
            if n >= m or n < 1 or m < 2:
                raise ValueError("HDS sparsity must satisfy 1 <= n < m, with m >= 2")
        
        # Validate learning rate
        if 'learning_rate' in hds_config:
            lr = hds_config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                raise ValueError("'hds.learning_rate' must be positive")
    
    logger.info("Configuration validation passed")
    return True


def load_and_validate_config(config_path: Union[str, Path], 
                           required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load and validate a configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        required_fields: List of required top-level fields
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config = load_yaml_config(config_path)
    validate_config(config, required_fields)
    return config


def get_config_value(config: Union[Dict[str, Any], DictConfig], 
                    key_path: str, 
                    default: Any = None) -> Any:
    """
    Safely get a value from nested configuration using dot notation.
    
    Args:
        config: Configuration dictionary or DictConfig
        key_path: Dot-separated path to the value (e.g., 'model.name')
        default: Default value if key doesn't exist
        
    Returns:
        Configuration value or default
        
    Example:
        >>> get_config_value(config, 'model.iasp.target_layer_name', 'default_layer')
    """
    if isinstance(config, DictConfig):
        return OmegaConf.select(config, key_path, default=default)
    
    # For regular dicts, implement dot notation access
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a configuration file. Convenience wrapper for load_yaml_config.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration data
    """
    return load_yaml_config(config_path)


# === Structured Configuration Extensions ===

@dataclass
class IaspConfig:
    """IO-Aware Scan Permutation configuration."""
    
    # Target layer patterns with model-family specific defaults
    # The actual default will be applied at runtime based on model family
    target_layers: Optional[List[str]] = None
    
    # Model family for default pattern selection
    model_family: str = "mamba"
    
    # Clustering parameters
    max_samples: int = 4096
    sample_stride: int = 2
    knn_k: int = 128
    cluster_size_range: List[int] = field(default_factory=lambda: [16, 128])
    
    # Performance tuning
    jobs: int = 1
    gpu_spectral: bool = True
    modularity_skip_threshold: float = 0.01
    
    # Spectral clustering parameters
    spectral_n_init: int = 10
    spectral_random_state: int = 42
    
    def __post_init__(self):
        """Apply model-family specific defaults if target_layers is None."""
        if self.target_layers is None:
            family_defaults = {
                "mamba": ["backbone.layers.*.mixer.in_proj"],
                "bert": ["*.intermediate.dense"],
            }
            self.target_layers = family_defaults.get(self.model_family, ["*.in_proj"])
            logger.debug(f"Applied default target_layers for {self.model_family}: {self.target_layers}")


@dataclass
class HdsConfig:
    """Hardware-Native Differentiable Sparsity configuration."""
    
    # Target layers for sparsification
    target_layers: List[str] = field(default_factory=list)
    
    # N:M sparsity parameters
    n: int = 2
    m: int = 4
    
    # Fine-tuning parameters
    fine_tuning_epochs: int = 1
    learning_rate: float = 0.0001
    
    # Layout-aware parameters
    layout_aware_lambda: float = 0.01
    penalty_fn: str = "inverse"


def initialize_structured_configs():
    """
    Initialize structured config schemas for use with OmegaConf.
    Call this before Hydra initialization.
    
    Note: This doesn't use global registration as that API is deprecated.
    Instead, it provides factory functions for creating properly structured configs.
    """
    # Modern approach - directly create structured configs when needed
    # No global registration, as that's deprecated
    
    # Register resolver using the modern approach
    try:
        # Modern approach (OmegaConf >= 2.1)
        from omegaconf.resolvers import register_resolver
        register_resolver("get_default_target_layers", get_default_target_layers)
    except ImportError:
        # Fall back to old API for backward compatibility
        from omegaconf import OmegaConf
        OmegaConf.register_resolver("get_default_target_layers", get_default_target_layers)
    
    logger.info("Structured config system initialized")


def create_iasp_config(cfg_dict: Optional[Dict[str, Any]] = None) -> Any:
    """
    Create a structured IaspConfig from a dictionary, with proper defaults.
    
    Args:
        cfg_dict: Dictionary with IASP configuration values
        
    Returns:
        Structured config instance with proper defaults
    """
    # Create base config with defaults
    base_cfg = IaspConfig()
    
    if cfg_dict is None:
        # Return a structured instance of the default config
        return OmegaConf.structured(base_cfg)
    
    # Convert dataclass to dict for safe merging
    base_dict = asdict(base_cfg)
    
    # Update with provided values 
    merged_dict = {**base_dict, **cfg_dict}
    
    # Create a config instance with the merged values
    config_instance = IaspConfig(**{k: v for k, v in merged_dict.items() 
                                 if k in base_dict})
    
    # Convert to structured config
    return OmegaConf.structured(config_instance)


def get_default_target_layers(model_family: str) -> list:
    """
    Get the default target layers for a specific model family.
    This can be used as a resolver in YAML configs.
    
    Args:
        model_family: The model family (e.g., "mamba", "bert")
        
    Returns:
        List of default target layer patterns
    """
    family_defaults = {
        "mamba": ["backbone.layers.*.mixer.in_proj"],
        "bert": ["*.intermediate.dense"],
        # Add more model families as needed
    }
    return family_defaults.get(model_family, ["*.in_proj"])


def create_hds_config(cfg_dict: Optional[Dict[str, Any]] = None) -> Any:
    """
    Create a structured HdsConfig from a dictionary, with proper defaults.
    
    Args:
        cfg_dict: Dictionary with HDS configuration values
        
    Returns:
        Structured config instance with proper defaults
    """
    # Create base config with defaults
    base_cfg = HdsConfig()
    
    if cfg_dict is None:
        # Return a structured instance of the default config
        return OmegaConf.structured(base_cfg)
    
    # Convert dataclass to dict for safe merging
    base_dict = asdict(base_cfg)
    
    # Update with provided values 
    merged_dict = {**base_dict, **cfg_dict}
    
    # Create a config instance with the merged values
    config_instance = HdsConfig(**{k: v for k, v in merged_dict.items() 
                                if k in base_dict})
    
    # Convert to structured config
    return OmegaConf.structured(config_instance)
