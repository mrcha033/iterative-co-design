"""Intel VTune profiler integration."""

from __future__ import annotations

import json
import logging
import pathlib
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)


class VTuneProfilerError(RuntimeError):
    """Raised when VTune profiling fails."""


@dataclass
class VTuneProfilerConfig:
    binary: pathlib.Path
    output_dir: pathlib.Path
    analysis_type: str = "gpu-hotspots"
    result_name: str = "vtune_result"
    env: Dict[str, str] = field(default_factory=dict)


class VTuneProfiler:
    """Wrapper around the ``vtune`` command line interface."""

    def __init__(self, config: VTuneProfilerConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def build_command(self) -> List[str]:
        return [
            "vtune",
            "-collect",
            self.config.analysis_type,
            "-result-dir",
            str(self.config.output_dir / self.config.result_name),
            str(self.config.binary),
        ]

    def run(self) -> pathlib.Path:
        cmd = self.build_command()
        try:
            subprocess.run(cmd, check=True, env=self.config.env)
        except FileNotFoundError as exc:
            raise VTuneProfilerError(
                "VTune CLI not found. Install Intel VTune Profiler and ensure `vtune` is available."
            ) from exc
        except subprocess.CalledProcessError as exc:  # pragma: no cover - requires hardware
            raise VTuneProfilerError(f"VTune profiling failed with exit code {exc.returncode}") from exc

        result_dir = self.config.output_dir / self.config.result_name
        if not result_dir.exists():
            raise VTuneProfilerError(f"VTune result directory was not created at {result_dir}")
        return result_dir

    def export_summary(self, result_dir: pathlib.Path) -> Dict[str, Any]:
        summary_path = result_dir / "summary.json"
        cmd = [
            "vtune",
            "-report",
            "summary",
            "-result-dir",
            str(result_dir),
            "-format",
            "json",
        ]
        try:
            output = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            raise VTuneProfilerError(f"VTune summary export failed: {exc.stderr}") from exc
        summary_path.write_text(output.stdout)
        return json.loads(output.stdout)

    def collect(self) -> Dict[str, Any]:
        result_dir = self.run()
        try:
            summary = self.export_summary(result_dir)
        except VTuneProfilerError as exc:
            LOGGER.warning("Failed to export VTune summary: %s", exc)
            summary = {}
        return {"result_dir": str(result_dir), "summary": summary}


__all__ = ["VTuneProfiler", "VTuneProfilerConfig", "VTuneProfilerError"]
