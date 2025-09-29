__version__ = "0.1.0"

__all__ = ["__version__"]

try:  # pragma: no cover - defensive import to stabilise test harness monkeypatching
    import importlib

    importlib.import_module("icd.graph")
except Exception:  # pragma: no cover - optional dependency failures are tolerated
    pass
