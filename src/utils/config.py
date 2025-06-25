"""
Configuration loading utilities for YAML files.
Implements Task 1.3 from tasks.md.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)


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
