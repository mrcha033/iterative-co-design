from __future__ import annotations


class ConfigError(Exception):
    """Raised when configuration is invalid or incomplete."""


class MeasureError(Exception):
    """Raised when measurement tools fail in a non-recoverable way."""


__all__ = ["ConfigError", "MeasureError"]

