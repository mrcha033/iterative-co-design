"""
Utility functions for safely handling OmegaConf/Hydra configs.
"""

from typing import Any, List, Sequence
from omegaconf import ListConfig, DictConfig


def to_plain_list(config_value: Any) -> List[str]:
    """
    Safely convert any config value (especially ListConfig) to a plain Python list of strings.
    
    This handles:
    - ListConfig -> list of strings
    - String -> single-element list
    - Sequence -> list
    - Any other value -> single-element list with string representation
    
    Args:
        config_value: Value from Hydra config, often a ListConfig
        
    Returns:
        A standard Python list of strings
    """
    if isinstance(config_value, ListConfig):
        return [str(item) for item in config_value]
    if isinstance(config_value, Sequence) and not isinstance(config_value, (str, bytes)):
        return [str(item) for item in config_value]
    if config_value is None:
        return []
    return [str(config_value)] 