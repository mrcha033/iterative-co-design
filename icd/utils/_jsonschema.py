"""Minimal JSON schema validator tailored for the environment fingerprint.

This tiny helper substitutes the external :mod:`jsonschema` dependency during
tests where installing third-party packages is undesirable.  Only the subset of
Draft 2020-12 that is required for ``docs/schema/env_fingerprint.json`` is
implemented which keeps the code compact while still providing meaningful
validation errors.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Dict


class ValidationError(ValueError):
    """Raised when an instance does not satisfy the expected structure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _ensure_object(instance: Any, *, name: str) -> Dict[str, Any]:
    _require(isinstance(instance, dict), f"{name} must be an object")
    return instance


def _ensure_string(value: Any, *, name: str) -> str:
    _require(isinstance(value, str), f"{name} must be a string")
    return value


def _ensure_number_mapping(value: Any, *, name: str) -> Dict[str, float]:
    mapping = _ensure_object(value, name=name)
    for key, item in mapping.items():
        _require(isinstance(key, str), f"{name} keys must be strings")
        _require(isinstance(item, Real), f"{name} values must be numbers")
    return {str(key): float(item) for key, item in mapping.items()}


def validate(instance: Any, schema: Dict[str, Any]) -> None:  # pragma: no cover - exercised indirectly
    """Validate ``instance`` against the fingerprint schema.

    The implementation assumes the provided ``schema`` is equivalent to
    ``docs/schema/env_fingerprint.json``.  ``schema`` is accepted to mirror the
    signature of :func:`jsonschema.validate` which keeps the public API
    compatible with the real library.
    """

    del schema  # schema structure is known ahead of time

    doc = _ensure_object(instance, name="fingerprint")
    allowed_top = {"host", "gpu", "packages"}
    _require(allowed_top.issuperset(doc), "unexpected top-level keys in fingerprint")
    for key in allowed_top:
        _require(key in doc, f"fingerprint missing required key: {key}")

    # host object -----------------------------------------------------
    host = _ensure_object(doc.get("host"), name="host")
    _require({"hostname", "driver"}.issubset(host), "host object missing fields")
    _require(set(host).issubset({"hostname", "driver"}), "host contains unknown fields")
    _ensure_string(host.get("hostname"), name="host.hostname")
    _ensure_string(host.get("driver"), name="host.driver")

    # gpu object ------------------------------------------------------
    gpu = _ensure_object(doc.get("gpu"), name="gpu")
    _require("name" in gpu, "gpu object missing name")
    _ensure_string(gpu.get("name"), name="gpu.name")
    allowed_gpu = {"name", "clocks", "temperatures"}
    _require(set(gpu).issubset(allowed_gpu), "gpu contains unknown fields")
    if "clocks" in gpu:
        _ensure_number_mapping(gpu["clocks"], name="gpu.clocks")
    if "temperatures" in gpu:
        _ensure_number_mapping(gpu["temperatures"], name="gpu.temperatures")

    # packages array --------------------------------------------------
    packages = doc.get("packages")
    _require(isinstance(packages, list), "packages must be an array")
    for idx, item in enumerate(packages):
        pkg = _ensure_object(item, name=f"packages[{idx}]")
        _require({"name", "version"}.issubset(pkg), "package entry missing fields")
        _require(set(pkg).issubset({"name", "version"}), "package entry contains unknown fields")
        _ensure_string(pkg.get("name"), name=f"packages[{idx}].name")
        _ensure_string(pkg.get("version"), name=f"packages[{idx}].version")


__all__ = ["ValidationError", "validate"]

