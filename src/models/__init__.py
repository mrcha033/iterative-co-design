"""
Model wrappers and utilities for neural network manipulation.

This package provides wrapper classes and utilities for working with
different model architectures and handling model transformations.
"""

from .wrapper import ModelWrapper
from .utils import get_device

__all__ = [
    "ModelWrapper",
    "get_device",
]
