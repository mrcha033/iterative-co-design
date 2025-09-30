#!/usr/bin/env python3
"""Execute systematic experimental matrix for paper validation.

This script runs complete experimental validation across multiple architectures,
baselines, and hardware configurations with proper statistical sampling.
"""

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experimental_matrix.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentSpec:
    """Specification for a single experimental configuration."""

    architecture: str
    config_path: str
    baselines: List[str]
    num_runs: int
    description: str = ""


# Experimental matrix from paper
CORE_EXPERIMENTS = [
    ExperimentSpec(
        architecture="mamba-2.8b",
        config_path="configs/mamba_3b.json",
        baselines=["dense", "algo_only", "linear", "iterative"],
        num_runs=5,
        description="State Space Model (Mamba-2.8B) on WikiText-103",
    ),
    ExperimentSpec(
        architecture="bert-large",
        config_path="configs/bert_large.json",
        baselines=["dense", "algo_only", "linear", "iterative"],
        num_runs=5,
        description="Transformer (BERT-large) on SST-2",
    ),
    ExperimentSpec(
        architecture="resnet-50",
        config_path="configs/resnet50.json",
        baselines=["dense", "algo_only", "linear", "iterative"],
        num_runs=5,
        description="CNN (ResNet-50) on ImageNet",
    ),
]

EXTENDED_EXPERIMENTS = [
    ExperimentSpec(
        architecture="gcn-arxiv",
        config_path="configs/gcn_arxiv.json",
        baselines=["dense", "algo_only", "linear", "iterative"],
        num_runs=5,
        description="GNN (GCN) on OGBN-ArXiv",
    ),
    ExperimentSpec(
        architecture="efficientnet-b0",
        config_path="configs/efficientnet_b0.json",
        baselines=["dense", "algo_only", "linear", "iterative"],
        num_runs=5,
        description="Efficient CNN (EfficientNet-B0) on ImageNet",
    ),
    ExperimentSpec(
        architecture="graphsage-arxiv",
        config_path="configs/graphsage_arxiv.json",
        baselines=["dense", "algo_only", "linear", "iterative"],
        num_runs=5,
        description="GNN (GraphSAGE) on OGBN-ArXiv",
    ),
]


class ExperimentRunner:
    """Orchestrates systematic experimental execution."""

    def __init__(self, output_root: Path, dry_run: bool = False):
        """Initialize experiment runner.

        Args:
            output_root: Root directory for experimental outputs.
            dry_run: If True, print commands without executing.
        """
        self.output_root = Path(output_root)
        self.dry_run = dry_run
        self.failed_runs = []
        self.successful_runs = []

    def run_single_experiment(
        self, spec: ExperimentSpec, baseline: str, run_id: int
    ) -> bool:
        """Execute a single experimental configuration.

        Args:
            spec: Experiment specification.
            baseline: Baseline method name.
            run_id: Run identifier (0-indexed).

        Returns:
            True if successful, False otherwise.
        """
        output_dir = self.output_root / spec.architecture / baseline / f"run_{run_id:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build CLI command
        cmd = [
            sys.executable,
            "-m",
            "icd.cli.main",
            "run",
            "-c",
            spec.config_path,
            "--override",
            f"pipeline.mode={baseline}",
            "--override",
            "measure.ncu_enable=true",
            "--override",
            "measure.power_enable=true",
            "--out",
            str(output_dir),
        ]

        run_name = f"{spec.architecture}/{baseline}/run_{run_id}"
        logger.info(f"Starting: {run_name}")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return True

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout per run
                check=False,
            )

            if result.returncode != 0:
                logger.error(f"Failed: {run_name}")
                logger.error(f"stderr: {result.stderr[-500:]}")  # Last 500 chars
                self.failed_runs.append(run_name)
                return False

            logger.info(f"✓ Completed: {run_name}")
            self.successful_runs.append(run_name)
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout: {run_name}")
            self.failed_runs.append(f"{run_name} (timeout)")
            return False
        except Exception as e:
            logger.error(f"Error in {run_name}: {e}")
            self.failed_runs.append(f"{run_name} (error: {e})")
            return False

    def run_experiment_spec(self, spec: ExperimentSpec) -> dict:
        """Run all configurations for an experiment spec.

        Args:
            spec: Experiment specification.

        Returns:
            Dictionary with execution statistics.
        """
        logger.info("=" * 70)
        logger.info(f"Starting: {spec.description}")
        logger.info(f"Architecture: {spec.architecture}")
        logger.info(f"Baselines: {', '.join(spec.baselines)}")
        logger.info(f"Runs per baseline: {spec.num_runs}")
        logger.info("=" * 70)

        total_runs = len(spec.baselines) * spec.num_runs
        completed = 0

        for baseline in spec.baselines:
            for run_id in range(spec.num_runs):
                success = self.run_single_experiment(spec, baseline, run_id)
                if success:
                    completed += 1

                # Progress update
                progress = (completed / total_runs) * 100
                logger.info(f"Progress: {completed}/{total_runs} ({progress:.1f}%)")

        return {
            "architecture": spec.architecture,
            "total_runs": total_runs,
            "completed": completed,
            "failed": total_runs - completed,
        }

    def run_all_experiments(self, experiments: List[ExperimentSpec]) -> None:
        """Run all experiments in the matrix.

        Args:
            experiments: List of experiment specifications.
        """
        logger.info(f"Starting experimental matrix with {len(experiments)} specs")
        logger.info(f"Output root: {self.output_root}")

        all_stats = []
        for spec in experiments:
            stats = self.run_experiment_spec(spec)
            all_stats.append(stats)

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENTAL MATRIX SUMMARY")
        logger.info("=" * 70)

        total_runs = sum(s["total_runs"] for s in all_stats)
        total_completed = sum(s["completed"] for s in all_stats)
        total_failed = sum(s["failed"] for s in all_stats)

        logger.info(f"Total runs: {total_runs}")
        logger.info(f"Completed: {total_completed}")
        logger.info(f"Failed: {total_failed}")
        logger.info(f"Success rate: {(total_completed/total_runs)*100:.1f}%")

        if self.failed_runs:
            logger.warning(f"\nFailed runs ({len(self.failed_runs)}):")
            for run in self.failed_runs:
                logger.warning(f"  - {run}")


def main():
    parser = argparse.ArgumentParser(
        description="Run systematic experimental matrix for paper validation"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/experimental_matrix"),
        help="Output root directory (default: runs/experimental_matrix)",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run only core experiments (Mamba, BERT, ResNet)",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Include extended experiments (GCN, EfficientNet, GraphSAGE)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        help="Run only specific architecture",
    )

    args = parser.parse_args()

    # Select experiments
    experiments = CORE_EXPERIMENTS.copy()
    if args.extended:
        experiments.extend(EXTENDED_EXPERIMENTS)

    # Filter by architecture if specified
    if args.architecture:
        experiments = [e for e in experiments if args.architecture in e.architecture]
        if not experiments:
            logger.error(f"No experiments found for architecture: {args.architecture}")
            return 1

    # Run experiments
    runner = ExperimentRunner(args.out, dry_run=args.dry_run)
    runner.run_all_experiments(experiments)

    return 0 if not runner.failed_runs else 1


if __name__ == "__main__":
    sys.exit(main())