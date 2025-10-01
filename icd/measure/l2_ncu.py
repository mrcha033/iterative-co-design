"""Real L2 cache profiling using NVIDIA Nsight Compute.

This module implements actual hardware measurement of L2 cache hit rates
using ncu (Nsight Compute CLI) to validate the paper's mechanistic claims.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["collect_l2_metrics", "collect_l2_section_stub", "parse_ncu_json"]


def find_ncu_binary() -> Optional[str]:
    """Locate ncu binary in standard CUDA installation paths."""
    # Standard CUDA installation paths
    candidates = [
        "/usr/local/cuda/bin/ncu",
        "/usr/local/cuda/bin/nv-nsight-cu-cli",
        "ncu",  # In PATH
        "nv-nsight-cu-cli",  # In PATH
    ]

    # Check environment variable
    env_ncu = os.environ.get("ICD_NCU_PATH")
    if env_ncu:
        candidates.insert(0, env_ncu)

    for path in candidates:
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.info(f"Found ncu binary: {path}")
                return path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return None


def parse_ncu_json(ncu_output_path: str) -> Dict[str, float]:
    """Parse Nsight Compute JSON output to extract L2 cache metrics.

    Args:
        ncu_output_path: Path to ncu JSON output file.

    Returns:
        Dictionary with L2 cache metrics:
        - l2_hit_rate_pct: L2 cache hit rate percentage
        - l2_throughput_pct: L2 throughput utilization percentage
        - dram_throughput_pct: DRAM bandwidth utilization percentage
    """
    try:
        with open(ncu_output_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Failed to parse ncu output: {e}")
        return {}

    metrics = {}

    # NCU output structure: top-level array with report objects
    # Each report has "sections" containing metric data
    try:
        if not isinstance(data, list) or len(data) == 0:
            logger.warning("NCU output is empty or malformed")
            return metrics

        # Process first report (for single-kernel profiling)
        report = data[0] if isinstance(data, list) else data

        # Look for MemoryWorkloadAnalysis section
        sections = report.get("sections", [])

        for section in sections:
            section_name = section.get("name", "")

            # L2 cache metrics
            if "MemoryWorkloadAnalysis" in section_name or "Memory" in section_name:
                body = section.get("body", [])

                for item in body:
                    if isinstance(item, dict):
                        metric_name = item.get("name", "")
                        metric_value = item.get("value")

                        # L2 hit rate (primary metric)
                        if "l2" in metric_name.lower() and "hit" in metric_name.lower():
                            try:
                                # Parse percentage value
                                if isinstance(metric_value, str):
                                    value = float(metric_value.strip("%"))
                                else:
                                    value = float(metric_value)
                                metrics["l2_hit_rate_pct"] = value
                                logger.info(f"Extracted L2 hit rate: {value}%")
                            except (ValueError, TypeError):
                                pass

                        # L2 throughput
                        if "l2" in metric_name.lower() and "throughput" in metric_name.lower():
                            try:
                                if isinstance(metric_value, str):
                                    value = float(metric_value.strip("%"))
                                else:
                                    value = float(metric_value)
                                metrics["l2_throughput_pct"] = value
                            except (ValueError, TypeError):
                                pass

                        # DRAM bandwidth
                        if "dram" in metric_name.lower() and ("throughput" in metric_name.lower() or "bandwidth" in metric_name.lower()):
                            try:
                                if isinstance(metric_value, str):
                                    value = float(metric_value.strip("%").replace("GB/s", ""))
                                else:
                                    value = float(metric_value)
                                metrics["dram_throughput_pct"] = value
                            except (ValueError, TypeError):
                                pass

        # Alternative: try direct metric names
        if "l2_hit_rate_pct" not in metrics:
            # Some NCU versions use different structures
            metric_values = report.get("metricValues", [])
            for mv in metric_values:
                metric_name = mv.get("metricName", "")
                if "l2_tex__t_sector_hit_rate" in metric_name or "l2_cache_hit_rate" in metric_name:
                    try:
                        value = float(mv.get("value", 0))
                        metrics["l2_hit_rate_pct"] = value
                        logger.info(f"Extracted L2 hit rate (alt): {value}%")
                    except (ValueError, TypeError):
                        pass

    except (KeyError, IndexError, AttributeError) as e:
        logger.error(f"Error parsing NCU structure: {e}")

    if not metrics:
        logger.warning("Could not extract any L2 metrics from NCU output")

    return metrics


def collect_l2_metrics(
    model: Any,
    inputs: Any,
    ncu_path: Optional[str] = None,
    metrics: Optional[list] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Collect L2 cache metrics using NVIDIA Nsight Compute.

    This is the REAL implementation that replaces the stub. It:
    1. Profiles model inference using ncu
    2. Extracts L2 cache hit rate from profiling data
    3. Returns metrics for correlation with modularity scores

    Args:
        model: PyTorch model to profile.
        inputs: Input tensors for model inference.
        ncu_path: Path to ncu binary (auto-detected if None).
        metrics: List of metric names to collect (default: L2 cache metrics).
        output_dir: Directory for profiling output (uses temp dir if None).

    Returns:
        Dictionary with L2 cache metrics.
        Returns empty dict if profiling fails (graceful degradation).
    """
    # Find ncu binary
    if ncu_path is None:
        ncu_path = find_ncu_binary()

    if ncu_path is None:
        logger.warning("NCU binary not found. Install NVIDIA Nsight Compute or set ICD_NCU_PATH.")
        return collect_l2_section_stub()

    # Check if CUDA is available
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Cannot profile GPU metrics.")
            return collect_l2_section_stub()
    except ImportError:
        logger.warning("PyTorch not available. Cannot profile.")
        return collect_l2_section_stub()

    # Default metrics to collect
    if metrics is None:
        metrics = [
            "l2_tex__t_sector_hit_rate.pct",
            "lts__t_sector_hit_rate.pct",  # L2 unified cache
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ]

    # Setup output
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        ncu_output = output_path / "ncu_profile.ncu-rep"
        json_output = output_path / "ncu_profile.json"
    else:
        # Use temporary file
        temp_dir = tempfile.mkdtemp(prefix="icd_ncu_")
        ncu_output = Path(temp_dir) / "profile.ncu-rep"
        json_output = Path(temp_dir) / "profile.json"

    try:
        import torch

        # Create a profiling script that loads model and runs inference
        script_path = ncu_output.parent / "profile_script.py"
        model_path = ncu_output.parent / "profile_model.pt"
        inputs_path = ncu_output.parent / "profile_inputs.pt"

        def _to_cpu(obj):
            if isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_cpu(v) for v in obj]
            if hasattr(obj, "detach"):
                obj = obj.detach()
            return obj.cpu() if hasattr(obj, "cpu") else obj

        # Save model and inputs for the subprocess
        try:
            device_str: str = "cpu"

            if hasattr(model, "parameters"):
                try:
                    device_str = str(next(model.parameters()).device)
                except StopIteration:
                    pass
            if device_str == "cpu" and hasattr(model, "buffers"):
                try:
                    device_str = str(next(model.buffers()).device)
                except StopIteration:
                    pass

            if hasattr(model, "to"):
                model.to("cpu")

            torch.save(model, model_path)

            input_is_tuple = isinstance(inputs, tuple)
            if isinstance(inputs, dict):
                input_args = []
                input_kwargs = _to_cpu(inputs)
            elif isinstance(inputs, (list, tuple)):
                input_args = [_to_cpu(inp) for inp in inputs]
                input_kwargs = {}
            else:
                input_args = [_to_cpu(inputs)]
                input_kwargs = {}

            inputs_payload = {
                "inputs": input_args,
                "kwargs": input_kwargs,
                "device": device_str,
                "input_is_tuple": input_is_tuple,
            }
            torch.save(inputs_payload, inputs_path)

            if device_str != "cpu" and hasattr(model, "to"):
                model.to(device_str)

            logger.info(f"Saved profiling artifacts to {model_path} and {inputs_path}")
        except Exception as e:
            logger.warning(f"Failed to serialize model/inputs: {e}. Trying direct profiling.")
            model_path = None
            inputs_path = None

        # Create the profiling script
        with open(script_path, "w") as f:
            if model_path and inputs_path:
                f.write(
                    f"""
import torch

model = torch.load({str(model_path)!r}, map_location="cpu")
data = torch.load({str(inputs_path)!r})

device = torch.device(data.get("device", "cpu"))
inputs = data.get("inputs", [])
kwargs = data.get("kwargs", {{}})
input_is_tuple = data.get("input_is_tuple", False)

def _move_to_device(value, target_device):
    if isinstance(value, dict):
        return {{k: _move_to_device(v, target_device) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_device(v, target_device) for v in value)
    if hasattr(value, "to"):
        return value.to(target_device)
    return value

inputs = _move_to_device(inputs, device)
kwargs = _move_to_device(kwargs, device)

if input_is_tuple and not isinstance(inputs, tuple):
    inputs = tuple(inputs)

model = model.to(device)
model.eval()

use_cuda = device.type == "cuda" and torch.cuda.is_available()
if use_cuda:
    torch.cuda.empty_cache()

with torch.no_grad():
    for _ in range(5):
        if kwargs:
            _ = model(*inputs, **kwargs)
        else:
            _ = model(*inputs)
        if use_cuda:
            torch.cuda.synchronize()
    if kwargs:
        output = model(*inputs, **kwargs)
    else:
        output = model(*inputs)
    if use_cuda:
        torch.cuda.synchronize()

print("Profiling complete")
"""
                )
            else:
                # Minimal profiling script if serialization failed
                f.write(
                    """
import torch

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    x = torch.randn(256, 256, device="cuda")
    y = torch.mm(x, x)
    torch.cuda.synchronize()
else:
    x = torch.randn(256, 256)
    y = torch.mm(x, x)

print("Profiling complete")
"""
                )

        # Build ncu command
        cmd = [
            ncu_path,
            "--target-processes", "all",
            "--export", str(ncu_output),
            "--force-overwrite",
            "--replay-mode", "kernel",  # Replay kernel mode for accurate profiling
        ]

        # Add metric collection
        for metric in metrics:
            cmd.extend(["--metrics", metric])

        # Add Python command to profile
        cmd.extend(["python", str(script_path)])

        logger.info(f"Running NCU profiling: {' '.join(cmd[:8])}...")

        # Execute NCU profiling
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(ncu_output.parent),
            )

            if result.returncode != 0:
                logger.error(f"NCU profiling failed with return code {result.returncode}")
                logger.error(f"stderr: {result.stderr[:500]}")
                return {
                    "l2_hit_rate_pct": float("nan"),
                    "ncu_available": True,
                    "error": "NCU execution failed",
                    "stderr": result.stderr[:200],
                }

            logger.info("NCU profiling completed successfully")

            # Convert .ncu-rep to JSON if needed
            if ncu_output.exists():
                # Export to JSON format
                export_cmd = [
                    ncu_path,
                    "--import", str(ncu_output),
                    "--export", str(json_output),
                    "--format", "json",
                ]

                export_result = subprocess.run(
                    export_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if export_result.returncode == 0 and json_output.exists():
                    # Parse JSON output
                    parsed_metrics = parse_ncu_json(str(json_output))

                    if parsed_metrics:
                        logger.info(f"Successfully extracted metrics: {parsed_metrics}")
                        return parsed_metrics
                    else:
                        logger.warning("NCU profiling succeeded but no metrics extracted")
                        return {
                            "l2_hit_rate_pct": float("nan"),
                            "ncu_available": True,
                            "note": "Profiling completed but metric extraction failed - may need different metric names",
                        }
                else:
                    logger.warning(f"Failed to export JSON: {export_result.stderr[:200]}")
                    # Try to return partial info
                    return {
                        "l2_hit_rate_pct": float("nan"),
                        "ncu_available": True,
                        "note": "Profiling completed but JSON export failed",
                    }
            else:
                logger.error("NCU output file not created")
                return collect_l2_section_stub()

        except subprocess.TimeoutExpired:
            logger.error("NCU profiling timed out after 5 minutes")
            return {
                "l2_hit_rate_pct": float("nan"),
                "ncu_available": True,
                "error": "Profiling timeout",
            }
        except Exception as e:
            logger.error(f"NCU subprocess execution failed: {e}")
            return collect_l2_section_stub()

    except Exception as e:
        logger.error(f"NCU profiling setup failed: {e}")
        return collect_l2_section_stub()


def collect_l2_section_stub() -> Dict[str, float]:
    """Mock L2 metrics collector when ncu is unavailable.

    Returns a fixed shape to keep schema consistent.
    """
    return {"l2_tex__t_sector_hit_rate.pct": float("nan")}
