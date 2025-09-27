"""ROCm profiling integration for AMD GPUs.

The implementation executes the ``rocprof`` command line tool and parses the
resulting JSON or CSV reports.  It is designed to behave similarly to the
existing NVIDIA profiler wrappers so that higher level benchmarking code can
switch between vendors with minimal conditional logic.
"""

from __future__ import annotations

import logging
import pathlib
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)


class ROCmProfilerError(RuntimeError):
    """Raised when ROCm profiling fails."""


@dataclass
class ROCmProfilerConfig:
    """Configuration for a profiling session."""

    binary: pathlib.Path
    output_dir: pathlib.Path
    metrics: List[str] = field(default_factory=lambda: ["SQ_WAVES", "VALUUtil"])
    kernel_regex: Optional[str] = None
    additional_args: List[str] = field(default_factory=list)


class ROCmProfiler:
    """Thin wrapper around the ``rocprof`` CLI."""

    def __init__(self, config: ROCmProfilerConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def build_command(self) -> List[str]:
        cmd = [
            "rocprof",
            "--hip-trace",
            "--hsatrace",
            "--stats",
            "--basenames",
            "--output", str(self.config.output_dir / "rocprof"),
        ]
        for metric in self.config.metrics:
            cmd.extend(["--metric", metric])
        if self.config.kernel_regex:
            cmd.extend(["--kernel-regex", self.config.kernel_regex])
        cmd.extend(self.config.additional_args)
        cmd.append(str(self.config.binary))
        LOGGER.debug("Constructed rocprof command: %s", " ".join(cmd))
        return cmd

    def run(self, env: Optional[Dict[str, str]] = None) -> pathlib.Path:
        """Execute rocprof and return the path to the generated stats file."""

        cmd = self.build_command()
        try:
            subprocess.run(cmd, check=True, env=env)
        except FileNotFoundError as exc:
            raise ROCmProfilerError(
                "rocprof executable not found. Ensure ROCm is installed and rocprof is on PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:  # pragma: no cover - requires hardware
            raise ROCmProfilerError(f"rocprof failed with exit code {exc.returncode}") from exc

        stats_file = self.config.output_dir / "rocprof_stats.csv"
        if not stats_file.exists():
            raise ROCmProfilerError(f"rocprof did not produce stats file at {stats_file}")
        return stats_file

    def parse_stats(self, stats_file: pathlib.Path) -> List[Dict[str, str]]:
        """Parse the CSV stats file produced by rocprof."""

        rows: List[Dict[str, str]] = []
        header: Optional[List[str]] = None
        for line in stats_file.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            keys = [key.strip() for key in line.split(",")]
            if header is None:
                header = keys
                continue
            row = dict(zip(header, keys))
            rows.append(row)
        return rows

    def collect(self) -> Dict[str, List[Dict[str, str]]]:
        stats_file = self.run()
        return {"metrics": self.parse_stats(stats_file)}


__all__ = ["ROCmProfiler", "ROCmProfilerConfig", "ROCmProfilerError"]
