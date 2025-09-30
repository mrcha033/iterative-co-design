"""Real NVIDIA Nsight Compute (NCU) integration for L2 cache profiling.

This module replaces the mock NCU implementation with actual hardware profiling.
It automatically detects NCU installation, generates proper command strings,
parses JSON output, and provides graceful fallbacks.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["NCUProfiler", "measure_l2_hit_rate_ncu", "is_ncu_available"]

logger = logging.getLogger(__name__)


def is_ncu_available() -> bool:
    """Check if NVIDIA Nsight Compute is available."""
    # Check common installation paths
    common_paths = [
        "/usr/local/cuda/bin/ncu",
        "/usr/local/cuda/bin/nsight-cu-cli",
        "/opt/nvidia/nsight-compute/ncu",
        shutil.which("ncu"),
        shutil.which("nsight-cu-cli"),
    ]

    for path in common_paths:
        if path and Path(path).exists():
            return True

    # Check environment variable
    if os.environ.get("NCU_PATH"):
        return Path(os.environ["NCU_PATH"]).exists()

    return False


def find_ncu_binary() -> Optional[str]:
    """Find NCU binary path."""
    # Try environment variable first
    if os.environ.get("NCU_PATH"):
        return os.environ["NCU_PATH"]

    # Check common paths
    common_paths = [
        "/usr/local/cuda/bin/ncu",
        "/usr/local/cuda/bin/nsight-cu-cli",
        "/opt/nvidia/nsight-compute/ncu",
    ]

    for path in common_paths:
        if Path(path).exists():
            return path

    # Try which
    for cmd in ["ncu", "nsight-cu-cli"]:
        found = shutil.which(cmd)
        if found:
            return found

    return None


class NCUProfiler:
    """NVIDIA Nsight Compute profiler for GPU memory metrics."""

    def __init__(self, ncu_path: Optional[str] = None, verbose: bool = False):
        """Initialize NCU profiler.

        Args:
            ncu_path: Path to NCU binary. If None, auto-detects.
            verbose: Enable verbose logging.
        """
        self.ncu_path = ncu_path or find_ncu_binary()
        self.verbose = verbose

        if not self.ncu_path:
            raise RuntimeError(
                "NVIDIA Nsight Compute (NCU) not found. "
                "Please install CUDA toolkit with Nsight Compute or set NCU_PATH environment variable."
            )

        if not Path(self.ncu_path).exists():
            raise RuntimeError(f"NCU binary not found at: {self.ncu_path}")

        logger.info(f"NCU profiler initialized with binary: {self.ncu_path}")

    def profile_command(
        self,
        command: List[str],
        output_dir: Optional[Path] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Profile a command with NCU.

        Args:
            command: Command to profile (as list of strings).
            output_dir: Directory for output files. If None, uses temp directory.
            metrics: List of NCU metrics to collect. If None, uses defaults.

        Returns:
            Dictionary with parsed metrics including:
                - l2_hit_pct: L2 cache hit percentage
                - dram_bandwidth_gbps: DRAM bandwidth in GB/s
                - l2_read_throughput_gbps: L2 read throughput
                - l2_write_throughput_gbps: L2 write throughput
        """
        if metrics is None:
            # Default metrics for memory analysis
            metrics = [
                "l2_tex_hit_rate",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "lts__t_sectors_op_read.sum",
                "lts__t_sectors_op_write.sum",
            ]

        # Create output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="ncu_profile_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "ncu_report"

        # Build NCU command
        ncu_cmd = [
            str(self.ncu_path),
            "--metrics", ",".join(metrics),
            "--target-processes", "all",
            "--export", str(output_file),
            "--force-overwrite",
        ]

        if not self.verbose:
            ncu_cmd.extend(["--log-level", "error"])

        # Add profiled command
        ncu_cmd.extend(command)

        logger.info(f"Running NCU profiling: {' '.join(ncu_cmd)}")

        try:
            result = subprocess.run(
                ncu_cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                check=False,
            )

            if result.returncode != 0:
                logger.error(f"NCU profiling failed: {result.stderr}")
                return self._create_null_result("ncu_failed")

        except subprocess.TimeoutExpired:
            logger.error("NCU profiling timed out")
            return self._create_null_result("timeout")
        except Exception as e:
            logger.error(f"NCU profiling error: {e}")
            return self._create_null_result("error")

        # Parse output
        json_file = Path(f"{output_file}.ncu-rep")
        if not json_file.exists():
            # Try alternative extension
            json_file = Path(f"{output_file}.json")

        if not json_file.exists():
            logger.error(f"NCU output file not found: {json_file}")
            return self._create_null_result("missing_output")

        try:
            with open(json_file) as f:
                ncu_data = json.load(f)
            return self._parse_ncu_json(ncu_data)
        except Exception as e:
            logger.error(f"Failed to parse NCU output: {e}")
            return self._create_null_result("parse_error")

    def _parse_ncu_json(self, ncu_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse NCU JSON output and extract key metrics."""
        result = {
            "l2_hit_pct": None,
            "dram_bandwidth_gbps": None,
            "l2_read_throughput_gbps": None,
            "l2_write_throughput_gbps": None,
            "status": "ok",
        }

        # NCU JSON structure varies by version
        # Try to find metrics in common locations
        metrics_data = {}

        # Try top-level format
        if isinstance(ncu_data, list) and len(ncu_data) > 0:
            first_report = ncu_data[0]
            if "metricValues" in first_report:
                for metric in first_report["metricValues"]:
                    name = metric.get("name", "")
                    value = metric.get("value")
                    metrics_data[name] = value

        # Try nested format
        elif "reports" in ncu_data:
            reports = ncu_data["reports"]
            if reports and len(reports) > 0:
                first_report = reports[0]
                if "metricValues" in first_report:
                    for metric in first_report["metricValues"]:
                        name = metric.get("name", "")
                        value = metric.get("value")
                        metrics_data[name] = value

        # Extract L2 hit rate
        for key in ["l2_tex_hit_rate", "l2_hit_rate", "lts__t_sector_hit_rate"]:
            if key in metrics_data and metrics_data[key] is not None:
                result["l2_hit_pct"] = float(metrics_data[key]) * 100.0
                break

        # Extract DRAM bandwidth
        for key in ["dram__throughput.avg.pct_of_peak_sustained_elapsed", "dram_throughput"]:
            if key in metrics_data and metrics_data[key] is not None:
                result["dram_bandwidth_gbps"] = float(metrics_data[key])
                break

        # Extract L2 throughput
        if "lts__t_sectors_op_read.sum" in metrics_data:
            # Convert sectors to GB (sector = 32 bytes typically)
            sectors = float(metrics_data["lts__t_sectors_op_read.sum"])
            result["l2_read_throughput_gbps"] = (sectors * 32) / 1e9

        if "lts__t_sectors_op_write.sum" in metrics_data:
            sectors = float(metrics_data["lts__t_sectors_op_write.sum"])
            result["l2_write_throughput_gbps"] = (sectors * 32) / 1e9

        return result

    def _create_null_result(self, reason: str) -> Dict[str, Any]:
        """Create null result with error reason."""
        return {
            "l2_hit_pct": None,
            "dram_bandwidth_gbps": None,
            "l2_read_throughput_gbps": None,
            "l2_write_throughput_gbps": None,
            "status": f"failed:{reason}",
        }


def measure_l2_hit_rate_ncu(
    command: List[str],
    output_dir: Optional[Path] = None,
    enabled: bool = True,
) -> Dict[str, Any]:
    """Measure L2 cache hit rate using NCU.

    This is the main entry point that matches the existing API.

    Args:
        command: Command to profile.
        output_dir: Output directory for NCU reports.
        enabled: Whether profiling is enabled. If False, returns nulls.

    Returns:
        Dictionary with metrics or nulls if disabled/unavailable.
    """
    if not enabled:
        return {
            "l2_hit_pct": None,
            "dram_bandwidth_gbps": None,
            "status": "disabled",
        }

    if not is_ncu_available():
        logger.warning("NCU not available, skipping L2 profiling")
        return {
            "l2_hit_pct": None,
            "dram_bandwidth_gbps": None,
            "status": "unavailable",
        }

    try:
        profiler = NCUProfiler()
        return profiler.profile_command(command, output_dir=output_dir)
    except Exception as e:
        logger.error(f"NCU profiling failed: {e}")
        return {
            "l2_hit_pct": None,
            "dram_bandwidth_gbps": None,
            "status": f"error:{str(e)}",
        }