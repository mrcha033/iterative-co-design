"""Environment fingerprint collection utilities."""

from __future__ import annotations

# NOTE: The module-level docstring was condensed to avoid exceeding lint limits.

import json
import platform
import socket
from importlib import import_module
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:  # pragma: no cover - optional dependency guard
    import pynvml
except Exception:  # pragma: no cover - handled in callers
    pynvml = None  # type: ignore[assignment]

import jsonschema


SchemaDict = Dict[str, Any]
Fingerprint = Dict[str, Any]

DEFAULT_PACKAGES: Tuple[Tuple[str, str | None], ...] = (
    ("icd", "icd"),
    ("torch", "torch"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pynvml", "pynvml"),
)


def load_env_fingerprint_schema() -> SchemaDict:
    """Load the environment fingerprint JSON schema."""

    schema_path = Path(__file__).resolve().parents[2] / "docs" / "schema" / "env_fingerprint.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_host_info(driver_hint: str | None = None) -> Dict[str, str]:
    hostname = socket.gethostname()
    driver = driver_hint or platform.platform()
    return {"hostname": hostname, "driver": driver}


def _collect_gpu_info() -> Tuple[Dict[str, Any], str | None]:
    """Collect GPU metadata via NVML when available."""

    gpu: Dict[str, Any] = {"name": "unavailable", "clocks": {}, "temperatures": {}}
    driver_hint: str | None = None

    if pynvml is None:  # pragma: no cover - exercised when NVML missing
        return gpu, driver_hint

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
    except pynvml.NVMLError:  # pragma: no cover - missing NVML runtime
        return gpu, driver_hint

    try:
        try:
            driver_bytes = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_bytes, bytes):
                driver_hint = driver_bytes.decode("utf-8")
            else:  # pragma: no cover - defensive
                driver_hint = str(driver_bytes)
        except pynvml.NVMLError:  # pragma: no cover - fallback on host info
            driver_hint = None

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError:
            return gpu, driver_hint

        try:
            name_bytes = pynvml.nvmlDeviceGetName(handle)
            gpu["name"] = name_bytes.decode("utf-8") if isinstance(name_bytes, bytes) else str(name_bytes)
        except pynvml.NVMLError:  # pragma: no cover - fallback to default name
            gpu["name"] = "unknown"

        clock_domains: List[Tuple[str, int]] = []
        for attr, label in (
            ("NVML_CLOCK_GRAPHICS", "graphics"),
            ("NVML_CLOCK_SM", "sm"),
            ("NVML_CLOCK_MEM", "memory"),
            ("NVML_CLOCK_VIDEO", "video"),
        ):
            domain = getattr(pynvml, attr, None)
            if domain is not None:
                clock_domains.append((label, domain))

        clocks: Dict[str, float] = {}
        for label, domain in clock_domains:
            try:
                clocks[label] = float(pynvml.nvmlDeviceGetClockInfo(handle, domain))
            except pynvml.NVMLError:  # pragma: no cover - skip unavailable domains
                continue
        if clocks:
            gpu["clocks"] = clocks

        temperature_targets: List[Tuple[str, int]] = []
        for attr, label in (
            ("NVML_TEMPERATURE_GPU", "gpu"),
            ("NVML_TEMPERATURE_MEMORY", "memory"),
        ):
            target = getattr(pynvml, attr, None)
            if target is not None:
                temperature_targets.append((label, target))

        temperatures: Dict[str, float] = {}
        for label, target in temperature_targets:
            try:
                temperatures[label] = float(pynvml.nvmlDeviceGetTemperature(handle, target))
            except pynvml.NVMLError:  # pragma: no cover - skip missing readings
                continue
        if temperatures:
            gpu["temperatures"] = temperatures

    finally:
        if initialized:
            try:  # pragma: no cover - ensure shutdown
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

    return gpu, driver_hint


def _resolve_package_versions(packages: Iterable[Tuple[str, str | None]]) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for dist_name, module_name in packages:
        version: str | None = None
        module = None
        if module_name:
            try:
                module = import_module(module_name)
            except Exception:  # pragma: no cover - optional dependency
                module = None
        if module is not None:
            version = getattr(module, "__version__", None)
        if version is None:
            try:
                version = importlib_metadata.version(dist_name)
            except importlib_metadata.PackageNotFoundError:
                version = None
        if version is not None:
            results.append({"name": dist_name, "version": str(version)})
    return results


def collect_env_fingerprint(packages: Iterable[Tuple[str, str | None]] | None = None) -> Fingerprint:
    """Collect and validate the current environment fingerprint."""

    gpu, driver_hint = _collect_gpu_info()
    host = _collect_host_info(driver_hint)
    package_entries = _resolve_package_versions(packages or DEFAULT_PACKAGES)
    fingerprint: Fingerprint = {"host": host, "gpu": gpu, "packages": package_entries}
    validate_env_fingerprint(fingerprint)
    return fingerprint


def validate_env_fingerprint(doc: Fingerprint) -> Fingerprint:
    """Validate ``doc`` against the environment fingerprint schema."""

    schema = load_env_fingerprint_schema()
    jsonschema.validate(doc, schema)
    return doc


__all__ = ["collect_env_fingerprint", "validate_env_fingerprint", "load_env_fingerprint_schema"]

