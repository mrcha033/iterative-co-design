"""Cross-vendor profiling utilities.

This module orchestrates profiling runs across NVIDIA, AMD, and Intel hardware
using the vendor-specific wrappers defined in :mod:`icd.measure`.  It aligns the
resulting metrics into a unified schema that is convenient for downstream
analysis and reproduces the workflows documented in
``docs/Cross_Vendor_Profiling.md``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)

try:  # Torch is optional for command materialisation.
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in minimal envs
    torch = None  # type: ignore

from .profiling import run_with_ncu
from .rocm_profiler import ROCmProfiler, ROCmProfilerConfig
from .vtune_profiler import VTuneProfiler, VTuneProfilerConfig

try:
    from .ncu import is_ncu_available
except Exception:  # pragma: no cover - optional dependency
    def is_ncu_available() -> bool:  # type: ignore
        return False


VendorCollector = Callable[[Sequence[str], Sequence[str], Optional[MutableMapping[str, str]], Optional[str]], Mapping[str, Any]]


def _normalize_vendor(vendor: str) -> str:
    vendor = vendor.lower()
    if "nvidia" in vendor or "cuda" in vendor:
        return "nvidia"
    if "amd" in vendor or "rocm" in vendor:
        return "amd"
    if "intel" in vendor or "xeon" in vendor or "vtune" in vendor:
        return "intel"
    return vendor


def _has_command(cmd: str) -> bool:
    return bool(shutil.which(cmd))


def _detect_nvidia_device() -> Optional[str]:
    for args in (
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
    ):
        try:
            output = subprocess.check_output(args, text=True)
        except (OSError, subprocess.CalledProcessError):  # pragma: no cover - depends on env
            continue
        name = output.strip().splitlines()[0] if output.strip() else ""
        if name:
            return name
    return None


def _detect_amd_device() -> Optional[str]:
    for args in (
        ["rocm-smi", "--showproductname"],
        ["rocm-smi", "-P"],
        ["rocminfo"],
    ):
        try:
            output = subprocess.check_output(args, text=True)
        except (OSError, subprocess.CalledProcessError):  # pragma: no cover
            continue
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "Product Name" in stripped:
                return stripped.split(":", 1)[-1].strip()
            if "gfx" in stripped.lower():
                return stripped
        if output:
            return output.splitlines()[0].strip()
    return None


def _detect_intel_device() -> Optional[str]:
    for args in (
        ["lspci"],
        ["lscpu"],
    ):
        try:
            output = subprocess.check_output(args, text=True)
        except (OSError, subprocess.CalledProcessError):  # pragma: no cover
            continue
        for line in output.splitlines():
            stripped = line.strip()
            if "Intel" not in stripped:
                continue
            if "CPU" in stripped or "processor" in stripped.lower():
                return stripped
            if "Graphics" in stripped or "GPU" in stripped:
                return stripped
        if output:
            return output.splitlines()[0].strip()
    return None


def _strip_percent(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def _average_metric(rows: Iterable[Mapping[str, Any]], key_candidates: Sequence[str]) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        for key in key_candidates:
            if key not in row:
                continue
            value = _strip_percent(row.get(key))
            if value is not None:
                values.append(value)
                break
    if not values:
        return None
    return sum(values) / len(values)


def _search_nested(summary: Any, key_candidates: Sequence[str]) -> Optional[float]:
    if isinstance(summary, Mapping):
        for key in key_candidates:
            if key in summary:
                value = _strip_percent(summary[key])
                if value is not None:
                    return value
        for value in summary.values():
            found = _search_nested(value, key_candidates)
            if found is not None:
                return found
    elif isinstance(summary, (list, tuple)):
        for item in summary:
            found = _search_nested(item, key_candidates)
            if found is not None:
                return found
    return None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_env(env: Optional[Mapping[str, str]]) -> MutableMapping[str, str]:
    merged: MutableMapping[str, str] = dict(os.environ)
    if env:
        merged.update(env)
    return merged


def _serialize_inputs(inputs: Any) -> Any:
    if isinstance(inputs, dict):
        return {k: _serialize_inputs(v) for k, v in inputs.items()}
    if isinstance(inputs, (list, tuple)):
        return type(inputs)(_serialize_inputs(v) for v in inputs)
    if torch is not None and hasattr(inputs, "detach"):
        detached = inputs.detach()
        return detached.cpu()
    if torch is not None and hasattr(inputs, "cpu"):
        return inputs.cpu()
    return inputs


def _materialize_model_command(
    model: Any,
    example_inputs: Any,
    work_dir: Path,
    *,
    warmup: int,
    iterations: int,
) -> Sequence[str]:
    if torch is None:  # pragma: no cover - torch is expected in production, but optional for docs builds
        raise RuntimeError("PyTorch is required to materialize model commands")

    _ensure_dir(work_dir)
    model_path = work_dir / "profile_model.pt"
    inputs_path = work_dir / "profile_inputs.pt"
    script_path = work_dir / "profile_runner.py"

    device_str = "cpu"
    if hasattr(model, "parameters"):
        try:
            device_str = str(next(model.parameters()).device)
        except StopIteration:
            device_str = "cpu"
    if device_str == "cpu" and hasattr(model, "buffers"):
        try:
            device_str = str(next(model.buffers()).device)
        except StopIteration:
            device_str = "cpu"

    current_device = device_str
    if hasattr(model, "to"):
        try:
            model = model.to("cpu")
        except Exception as exc:  # pragma: no cover - depends on model
            raise RuntimeError(f"Failed to move model to CPU for serialization: {exc}") from exc
    torch.save(model, model_path)

    if hasattr(model, "to"):
        try:
            model.to(current_device)
        except Exception:
            pass

    if isinstance(example_inputs, dict):
        inputs_serialized = ()
        kwargs_serialized = _serialize_inputs(example_inputs)
        input_is_tuple = False
    elif isinstance(example_inputs, tuple):
        inputs_serialized = _serialize_inputs(example_inputs)
        kwargs_serialized = {}
        input_is_tuple = True
    elif isinstance(example_inputs, list):
        inputs_serialized = _serialize_inputs(tuple(example_inputs))
        kwargs_serialized = {}
        input_is_tuple = True
    else:
        inputs_serialized = _serialize_inputs((example_inputs,))
        kwargs_serialized = {}
        input_is_tuple = True

    payload = {
        "inputs": inputs_serialized,
        "kwargs": kwargs_serialized,
        "input_is_tuple": input_is_tuple,
        "device": current_device,
    }

    torch.save(payload, inputs_path)

    script = f"""
import argparse
import torch


def _move(value, device):
    if isinstance(value, dict):
        return {{k: _move(v, device) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return type(value)(_move(v, device) for v in value)
    if hasattr(value, "to"):
        return value.to(device)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--warmup", type=int, default={warmup})
    parser.add_argument("--iterations", type=int, default={iterations})
    args = parser.parse_args()

    model = torch.load(args.model, map_location="cpu", weights_only=False)
    payload = torch.load(args.inputs)

    device = torch.device(payload.get("device", "cpu"))
    inputs = payload.get("inputs", ())
    kwargs = payload.get("kwargs", {{}})
    if payload.get("input_is_tuple", False) and not isinstance(inputs, tuple):
        inputs = tuple(inputs)

    inputs = _move(inputs, device)
    kwargs = _move(kwargs, device)

    model = model.to(device)
    model.eval()

    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            model(*inputs, **kwargs)
            if use_cuda:
                torch.cuda.synchronize(device)
        for _ in range(max(1, args.iterations)):
            model(*inputs, **kwargs)
            if use_cuda:
                torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
"""

    script_path.write_text(script)
    return [sys.executable, str(script_path), "--model", str(model_path), "--inputs", str(inputs_path)]


VENDOR_METRIC_MAP: Dict[str, Dict[str, Sequence[str]]] = {
    "nvidia": {
        "gpu_utilization": ("sm__cycles_active.avg.pct_of_peak_sustained_elapsed",),
        "memory_bandwidth_pct": ("dram__throughput.avg.pct_of_peak_sustained_elapsed",),
        "cache_hit_rate": (
            "lts__t_sectors_hit_rate.pct",
            "l2_tex__t_sector_hit_rate.pct",
        ),
        "compute_efficiency": (
            "tensor_precision_fu_utilization",
            "smsp__sass_average_warp_registers_utilization.pct",
        ),
    },
    "amd": {
        "gpu_utilization": ("GPUBusy",),
        "memory_bandwidth_pct": ("MemUnit",),
        "cache_hit_rate": ("L2CacheHit",),
        "compute_efficiency": ("VALUUtil", "SALUUtil"),
    },
    "intel": {
        "gpu_utilization": ("GPU Utilization", "GPU Utilization (%)"),
        "memory_bandwidth_pct": ("Memory Bandwidth", "Memory Bandwidth (%)"),
        "cache_hit_rate": ("L2 Hit Rate", "L2 Cache Hit Rate"),
        "compute_efficiency": ("FP Efficiency", "Compute Utilization"),
    },
}


DEFAULT_NVIDIA_METRICS = (
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sectors_hit_rate.pct",
    "tensor_precision_fu_utilization",
)


DEFAULT_AMD_METRICS = (
    "SQ_WAVES",
    "VALUUtil",
    "SALUUtil",
    "MemUnit",
    "L2CacheHit",
    "GPUBusy",
)


DEFAULT_INTEL_ANALYSIS = "gpu-hotspots"


class CrossVendorProfiler:
    """Unified interface for vendor-specific profilers."""

    def __init__(
        self,
        vendor: str,
        *,
        nvidia_collector: Optional[VendorCollector] = None,
        amd_collector: Optional[Callable[[Sequence[str], MutableMapping[str, str], Path, Sequence[str]], Mapping[str, Any]]] = None,
        intel_collector: Optional[Callable[[Sequence[str], MutableMapping[str, str], Path, str, Sequence[str]], Mapping[str, Any]]] = None,
    ) -> None:
        self.vendor = _normalize_vendor(vendor)
        if self.vendor not in ("nvidia", "amd", "intel"):
            raise ValueError(f"Unsupported vendor: {vendor}")
        self._nvidia_collector = nvidia_collector or self._collect_nvidia
        self._amd_collector = amd_collector or self._collect_amd
        self._intel_collector = intel_collector or self._collect_intel

    # ------------------------------------------------------------------
    # Vendor detection helpers
    # ------------------------------------------------------------------
    @classmethod
    def available_vendors(cls) -> List[str]:
        vendors: List[str] = []
        if is_ncu_available() or _has_command("nvidia-smi"):
            vendors.append("nvidia")
        if _has_command("rocprof"):
            vendors.append("amd")
        if _has_command("vtune"):
            vendors.append("intel")
        return vendors

    @classmethod
    def auto_detect(cls, preferred_order: Optional[Sequence[str]] = None) -> "CrossVendorProfiler":
        candidates = preferred_order or ("nvidia", "amd", "intel")
        available = cls.available_vendors()
        LOGGER.debug("Detected vendor candidates: %s", available)
        for vendor in candidates:
            if _normalize_vendor(vendor) in available:
                return cls(vendor)
        if available:
            return cls(available[0])
        raise RuntimeError("No supported vendor profiler detected. Ensure Nsight Compute, rocprof, or VTune is installed.")

    @classmethod
    def for_vendor(cls, vendor: str) -> "CrossVendorProfiler":
        return cls(vendor)

    # ------------------------------------------------------------------
    # Metric collection
    # ------------------------------------------------------------------
    def collect_unified_metrics(
        self,
        *,
        command: Optional[Sequence[str]] = None,
        binary: Optional[Path] = None,
        model: Any = None,
        inputs: Any = None,
        env: Optional[Mapping[str, str]] = None,
        output_dir: Optional[Path] = None,
        keep_artifacts: bool = False,
        warmup: int = 5,
        iterations: int = 10,
        metrics: Optional[Sequence[str]] = None,
        analysis_type: Optional[str] = None,
        working_directory: Optional[Path] = None,
    ) -> Dict[str, Any]:
        if command is not None and binary is not None:
            raise ValueError("Provide either a command or a binary path, not both")

        temp_dir_created = False
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="icd_cross_vendor_"))
            temp_dir_created = True
        else:
            output_dir = Path(output_dir)
            _ensure_dir(output_dir)

        work_dir = working_directory or output_dir
        env_vars = _prepare_env(env)

        target_command: Sequence[str]
        if command is not None:
            target_command = list(command)
        elif binary is not None:
            target_command = [str(binary)]
        elif model is not None:
            if inputs is None:
                raise ValueError("`inputs` must be provided when profiling a model")
            target_command = _materialize_model_command(model, inputs, output_dir, warmup=warmup, iterations=iterations)
        else:
            raise ValueError("A command, binary, or model must be supplied for profiling")

        LOGGER.debug("Running %s profiling with command: %s", self.vendor, " ".join(target_command))

        if self.vendor == "nvidia":
            vendor_metrics = metrics or DEFAULT_NVIDIA_METRICS
            raw = self._nvidia_collector(target_command, vendor_metrics, env_vars, str(work_dir))
            device = _detect_nvidia_device()
        elif self.vendor == "amd":
            vendor_metrics = metrics or DEFAULT_AMD_METRICS
            raw = self._amd_collector(target_command, env_vars, output_dir, vendor_metrics)
            device = _detect_amd_device()
        else:
            vendor_metrics = metrics or ()
            raw = self._intel_collector(
                target_command,
                env_vars,
                output_dir,
                analysis_type or DEFAULT_INTEL_ANALYSIS,
                vendor_metrics,
            )
            device = _detect_intel_device()

        unified = self._map_metrics(raw)
        result = {
            "vendor": self.vendor,
            "device": device or os.environ.get("ICD_DEVICE", self.vendor),
            "metrics": unified,
            "raw": raw,
        }
        if keep_artifacts:
            result["artifacts"] = str(output_dir)
        elif temp_dir_created:
            shutil.rmtree(output_dir, ignore_errors=True)
        return result

    # ------------------------------------------------------------------
    # Internal collectors
    # ------------------------------------------------------------------
    def _collect_nvidia(
        self,
        command: Sequence[str],
        metrics: Sequence[str],
        env: Optional[MutableMapping[str, str]],
        cwd: Optional[str],
    ) -> Mapping[str, Any]:
        metric_list = list(metrics)
        return run_with_ncu(command, metric_list, env=dict(env or {}), cwd=cwd)

    def _collect_amd(
        self,
        command: Sequence[str],
        env: MutableMapping[str, str],
        output_dir: Path,
        metrics: Sequence[str],
    ) -> Mapping[str, Any]:
        if not command:
            raise ValueError("Command must not be empty for ROCm profiling")
        config = ROCmProfilerConfig(
            binary=Path(command[0]),
            args=list(command[1:]),
            output_dir=output_dir,
            metrics=list(metrics),
            working_dir=output_dir,
        )
        profiler = ROCmProfiler(config)
        return profiler.collect()

    def _collect_intel(
        self,
        command: Sequence[str],
        env: MutableMapping[str, str],
        output_dir: Path,
        analysis: str,
        metrics: Sequence[str],
    ) -> Mapping[str, Any]:
        if not command:
            raise ValueError("Command must not be empty for VTune profiling")
        config = VTuneProfilerConfig(
            binary=Path(command[0]),
            args=list(command[1:]),
            output_dir=output_dir,
            analysis_type=analysis,
            env=dict(env),
            working_dir=output_dir,
        )
        profiler = VTuneProfiler(config)
        data = profiler.collect()
        if metrics:
            data.setdefault("requested_metrics", list(metrics))
        return data

    # ------------------------------------------------------------------
    # Metric mapping
    # ------------------------------------------------------------------
    def _map_metrics(self, raw: Mapping[str, Any]) -> Dict[str, Optional[float]]:
        mapping = VENDOR_METRIC_MAP[self.vendor]
        unified: Dict[str, Optional[float]] = {}
        if self.vendor == "amd":
            rows = raw.get("metrics", []) if isinstance(raw, Mapping) else []
            for name, aliases in mapping.items():
                unified[name] = _average_metric(rows, aliases)
        elif self.vendor == "intel":
            summary = raw.get("summary") if isinstance(raw, Mapping) else None
            for name, aliases in mapping.items():
                unified[name] = _search_nested(summary, aliases) if summary else None
        else:
            for name, aliases in mapping.items():
                value: Optional[float] = None
                for alias in aliases:
                    if alias in raw:
                        value = _strip_percent(raw[alias])
                        if value is not None:
                            break
                unified[name] = value
        return unified


def _default_command_builder(
    vendor_key: str,
    model_path: Optional[Path],
    config_path: Optional[Path],
    out_root: Path,
) -> Sequence[str]:
    if config_path is None:
        raise ValueError("config_path is required when command_builder is not provided")

    vendor = _normalize_vendor(vendor_key)
    vendor_dir = _ensure_dir(out_root / vendor_key.replace("/", "_"))

    overrides = []
    if model_path:
        overrides.append(f"pipeline.model_path={model_path}")
    vendor_overrides = {
        "nvidia": ["measure.ncu_enable=true"],
        "amd": ["measure.rocm_enable=true"],
        "intel": ["measure.vtune_enable=true"],
    }
    for override in vendor_overrides.get(vendor, []):
        overrides.append(override)

    cmd: List[str] = [
        sys.executable,
        "-m",
        "icd.cli.main",
        "run",
        "-c",
        str(config_path),
        "--out",
        str(vendor_dir),
    ]
    for override in overrides:
        cmd.extend(["--override", override])
    return cmd


@dataclass
class CrossVendorValidator:
    """Coordinate profiling across multiple vendors."""

    vendors: Sequence[str]
    profiler_factory: Callable[[str], CrossVendorProfiler] = CrossVendorProfiler.for_vendor

    def validate_model(
        self,
        *,
        model_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        metrics: Optional[Sequence[str]] = None,
        command_builder: Optional[Callable[[str, Optional[Path], Optional[Path], Path], Sequence[str]]] = None,
        output_root: Optional[Path] = None,
        warmup: int = 5,
        iterations: int = 10,
        env: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        if not self.vendors:
            raise ValueError("At least one vendor must be specified")

        temp_dir_created = False
        if output_root is None:
            output_root = Path(tempfile.mkdtemp(prefix="icd_cross_vendor_validator_"))
            temp_dir_created = True
        else:
            output_root = Path(output_root)
            _ensure_dir(output_root)

        builder = command_builder or _default_command_builder
        results: Dict[str, Any] = {}

        for vendor_name in self.vendors:
            profiler = self.profiler_factory(vendor_name)
            command = builder(vendor_name, model_path, config_path, output_root)
            LOGGER.info("Validating %s with command: %s", vendor_name, " ".join(command))
            safe_vendor = vendor_name.replace("/", "_").replace(" ", "_")
            result = profiler.collect_unified_metrics(
                command=command,
                env=env,
                output_dir=output_root / f"{safe_vendor}_profile",
                keep_artifacts=True,
                warmup=warmup,
                iterations=iterations,
                metrics=metrics,
            )
            results[vendor_name] = result

        summary = self._summarise(results, metrics)

        if temp_dir_created:
            LOGGER.info("Cross-vendor validation artifacts stored under %s", output_root)

        return {"runs": results, "summary": summary, "artifact_root": str(output_root)}

    def _summarise(self, runs: Mapping[str, Any], metrics: Optional[Sequence[str]]) -> Dict[str, Any]:
        metrics = list(metrics or ("gpu_utilization", "memory_bandwidth_pct", "cache_hit_rate", "compute_efficiency"))
        if not runs:
            return {metric: {} for metric in metrics}

        baseline_key = next(iter(self.vendors))

        summary: Dict[str, Any] = {}
        for metric in metrics:
            values: Dict[str, Optional[float]] = {}
            for vendor_name, result in runs.items():
                values[vendor_name] = result.get("metrics", {}).get(metric)
            baseline_value = values.get(baseline_key)
            relative: Dict[str, Optional[float]] = {}
            for vendor_name, value in values.items():
                if baseline_value in (None, 0, 0.0) or value is None:
                    relative[vendor_name] = None
                else:
                    relative[vendor_name] = value / baseline_value if baseline_value else None
            summary[metric] = {
                "values": values,
                "baseline": baseline_key,
                "relative_to_baseline": relative,
            }
        return summary

    def generate_report(self, results: Mapping[str, Any], output_path: Path) -> None:
        runs = results.get("runs", {}) if isinstance(results, Mapping) else {}
        summary = results.get("summary", {}) if isinstance(results, Mapping) else {}

        html_lines = [
            "<html>",
            "<head><meta charset='utf-8'><title>Cross-Vendor Profiling Report</title>",
            "<style>body{font-family:Arial, sans-serif;} table{border-collapse:collapse;margin:1em 0;} th,td{border:1px solid #ccc;padding:0.5em;} th{background:#f2f2f2;}</style>",
            "</head>",
            "<body>",
            "<h1>Cross-Vendor Profiling Report</h1>",
        ]

        if runs:
            html_lines.append("<h2>Run Details</h2>")
            for vendor_name, data in runs.items():
                html_lines.append(f"<h3>{vendor_name}</h3>")
                html_lines.append("<pre>" + json.dumps(data.get("metrics", {}), indent=2) + "</pre>")

        if summary:
            html_lines.append("<h2>Metric Summary</h2>")
            html_lines.append("<table>")
            html_lines.append("<tr><th>Metric</th>")
            for vendor_name in runs.keys():
                html_lines.append(f"<th>{vendor_name}</th>")
            html_lines.append("</tr>")
            for metric, info in summary.items():
                html_lines.append(f"<tr><td>{metric}</td>")
                for vendor_name in runs.keys():
                    value = info.get("values", {}).get(vendor_name)
                    if value is None:
                        html_lines.append("<td>n/a</td>")
                    else:
                        html_lines.append(f"<td>{value:.2f}</td>")
                html_lines.append("</tr>")
            html_lines.append("</table>")

        html_lines.append("</body></html>")
        output_path = Path(output_path)
        output_path.write_text("\n".join(html_lines), encoding="utf-8")


__all__ = ["CrossVendorProfiler", "CrossVendorValidator"]

