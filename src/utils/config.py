"""
Configuration loading utilities for YAML files.
Implements Task 1.3 from tasks.md.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union
from omegaconf import DictConfig, OmegaConf


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
