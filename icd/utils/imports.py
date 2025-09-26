"""Utility helpers for dynamic dotted-path imports."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


def _candidate_module_paths(module_name: str) -> Iterable[Path]:
    rel = Path(*module_name.split("."))
    candidates = [rel.with_suffix(".py"), rel / "__init__.py"]
    for base in [Path.cwd()] + [Path(p) for p in sys.path if p]:
        for cand in candidates:
            full = base / cand
            if full.is_file():
                yield full


def _load_module_from_path(module_name: str) -> ModuleType | None:
    for file_path in _candidate_module_paths(module_name):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[arg-type]
            return module
    return None


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
        module = _load_module_from_path(module_name)
        if module is None:
            raise ValueError(f"module '{module_name}' could not be imported") from exc

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"attribute '{attr_name}' not found in module '{module_name}'") from exc


__all__ = ["load_object"]
