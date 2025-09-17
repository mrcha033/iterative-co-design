"""Utility helpers for dynamic dotted-path imports."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


def load_object(path: str) -> Any:
    """Load an object from ``module.attr`` or ``module:attr`` dotted syntax.

    Raises ``ValueError`` with a descriptive message when the module or
    attribute cannot be resolved.
    """

    if not isinstance(path, str) or not path:
        raise ValueError("import path must be a non-empty string")

    module_name: str
    attr_name: str

    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        parts = path.split(".")
        if len(parts) < 2:
            raise ValueError(
                "import specification must contain both module and attribute (module.attr or module:attr)"
            )
        module_name = ".".join(parts[:-1])
        attr_name = parts[-1]

    try:
        module: ModuleType = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        raise ValueError(f"module '{module_name}' could not be imported") from exc

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"attribute '{attr_name}' not found in module '{module_name}'") from exc


__all__ = ["load_object"]

