"""Experimental data collection and aggregation utilities.

Provides systematic collection of metrics from experimental runs, aggregation
into structured datasets, and export to CSV/JSON for analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["DataCollector", "aggregate_architecture_results", "collect_all_experiments"]

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects and aggregates experimental data from run directories."""

    def __init__(self, output_root: Path):
        """Initialize data collector.

        Args:
            output_root: Root directory containing experimental runs.
        """
        self.output_root = Path(output_root)
        self.raw_data_dir = self.output_root / "raw_data"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def collect_run_metrics(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        """Extract all metrics from a single run directory.

        Args:
            run_dir: Path to run directory containing metrics.json

        Returns:
            Dictionary with extracted metrics, or None if unavailable.
        """
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            logger.warning(f"No metrics.json found in {run_dir}")
            return None

        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load metrics from {run_dir}: {e}")
            return None

        # Extract key metrics with defensive access
        latency_data = metrics.get("latency_ms", {})
        if isinstance(latency_data, dict):
            latency_mean = latency_data.get("mean")
            latency_samples = latency_data.get("samples", [])
        else:
            latency_mean = latency_data
            latency_samples = []

        solver_stats = metrics.get("solver_stats", {})
        quality_data = metrics.get("quality", {})

        return {
            "run_dir": str(run_dir),
            "latency_mean": latency_mean,
            "latency_p50": latency_data.get("p50") if isinstance(latency_data, dict) else None,
            "latency_p95": latency_data.get("p95") if isinstance(latency_data, dict) else None,
            "latency_samples": latency_samples,
            "l2_hit_pct": metrics.get("l2_hit_pct"),
            "ept_j_per_tok": metrics.get("ept_j_per_tok"),
            "power_mean_watts": metrics.get("power_stats", {}).get("mean_watts"),
            "energy_joules": metrics.get("power_stats", {}).get("total_energy_joules"),
            "quality_before": quality_data.get("before"),
            "quality_after": quality_data.get("after"),
            "quality_delta": quality_data.get("delta"),
            "modularity": solver_stats.get("Q_final") or solver_stats.get("Q_louvain"),
            "cost_J": solver_stats.get("J"),
            "solver_method": solver_stats.get("method"),
            "clusters": solver_stats.get("clusters"),
            "mode": metrics.get("mode"),
        }

    def aggregate_architecture_results(
        self, architecture: str, baselines: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate all runs for a single architecture.

        Args:
            architecture: Architecture name (e.g., "mamba", "bert").
            baselines: List of baseline methods to collect. If None, auto-detects.

        Returns:
            List of dictionaries, one per run.
        """
        arch_dir = self.output_root / architecture
        if not arch_dir.exists():
            logger.warning(f"Architecture directory not found: {arch_dir}")
            return []

        if baselines is None:
            # Auto-detect baselines from subdirectories
            baselines = [d.name for d in arch_dir.iterdir() if d.is_dir()]

        all_runs = []
        for baseline in baselines:
            baseline_dir = arch_dir / baseline
            if not baseline_dir.exists():
                logger.info(f"Baseline {baseline} not found for {architecture}")
                continue

            # Collect runs
            for run_dir in sorted(baseline_dir.glob("run_*")):
                metrics = self.collect_run_metrics(run_dir)
                if metrics:
                    metrics["architecture"] = architecture
                    metrics["baseline"] = baseline
                    metrics["run_id"] = run_dir.name
                    all_runs.append(metrics)

        logger.info(f"Collected {len(all_runs)} runs for {architecture}")
        return all_runs

    def collect_all_experiments(
        self, architectures: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Collect data from all experiments.

        Args:
            architectures: List of architectures to collect. If None, auto-detects.

        Returns:
            Dictionary mapping architecture names to lists of run data.
        """
        if architectures is None:
            # Auto-detect architectures from subdirectories
            architectures = [
                d.name
                for d in self.output_root.iterdir()
                if d.is_dir() and d.name != "raw_data"
            ]

        all_data = {}
        for arch in architectures:
            arch_data = self.aggregate_architecture_results(arch)
            if arch_data:
                all_data[arch] = arch_data

        return all_data

    def export_to_csv(self, data: List[Dict[str, Any]], output_path: Path) -> None:
        """Export collected data to CSV.

        Args:
            data: List of run dictionaries.
            output_path: Path for output CSV file.
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas required for CSV export. Install with: pip install pandas")
            return

        df = pd.DataFrame(data)

        # Exclude list columns that don't fit CSV format
        for col in df.columns:
            if df[col].dtype == object:
                # Check if column contains lists
                if any(isinstance(v, list) for v in df[col].dropna()):
                    df = df.drop(columns=[col])

        df.to_csv(output_path, index=False)
        logger.info(f"Exported data to {output_path}")

    def export_to_json(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export collected data to JSON.

        Args:
            data: Data dictionary to export.
            output_path: Path for output JSON file.
        """
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported data to {output_path}")


def aggregate_architecture_results(
    output_root: Path, architecture: str
) -> List[Dict[str, Any]]:
    """Convenience function to aggregate results for an architecture.

    Args:
        output_root: Root directory containing experimental runs.
        architecture: Architecture name.

    Returns:
        List of run dictionaries.
    """
    collector = DataCollector(output_root)
    return collector.aggregate_architecture_results(architecture)


def collect_all_experiments(output_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience function to collect all experimental data.

    Args:
        output_root: Root directory containing experimental runs.

    Returns:
        Dictionary mapping architectures to run data lists.
    """
    collector = DataCollector(output_root)
    return collector.collect_all_experiments()