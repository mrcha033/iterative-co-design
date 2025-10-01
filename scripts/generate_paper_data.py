#!/usr/bin/env python3
"""Generate ALL empirical data for the paper using real hardware measurements.

This is the MASTER script that runs complete experimental validation
to replace mock data with real measurements. Run this on GPU hardware
to generate publication-quality results.

What this script does:
1. Validates mechanistic claim (Modularity → Cache → Latency)
2. Runs experimental matrix (Table 1: Linear vs Iterative)
3. Generates correlation data (Figure 8, Table 2)
4. Produces mediation analysis results (Section 3.5)
5. Creates all figures and tables with REAL data

Usage:
    # Full validation (takes several hours on GPU)
    python scripts/generate_paper_data.py --output results/paper_data

    # Quick test run (small sample size)
    python scripts/generate_paper_data.py --output results/test --quick

    # Specific experiment only
    python scripts/generate_paper_data.py --experiment mamba --output results/mamba_only

Requirements:
    - CUDA-capable GPU (A100/H100 recommended)
    - NVIDIA Nsight Compute installed
    - PyTorch with CUDA support
    - At least 40GB GPU memory for large models
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_prerequisites() -> bool:
    """Verify that all required tools and libraries are available."""
    logger.info("=" * 80)
    logger.info("CHECKING PREREQUISITES")
    logger.info("=" * 80)

    checks_passed = True

    # Check 1: PyTorch with CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✓ PyTorch CUDA available: {device_name} ({memory_gb:.1f} GB)")
        else:
            logger.error("✗ CUDA not available in PyTorch")
            checks_passed = False
    except ImportError:
        logger.error("✗ PyTorch not installed")
        checks_passed = False

    # Check 2: Nsight Compute
    from icd.measure.l2_ncu import find_ncu_binary
    ncu_path = find_ncu_binary()
    if ncu_path:
        logger.info(f"✓ Nsight Compute found: {ncu_path}")
    else:
        logger.warning("⚠ Nsight Compute not found - L2 profiling will be skipped")
        logger.warning("  Install from: https://developer.nvidia.com/nsight-compute")

    # Check 3: Required Python packages
    required_packages = ["scipy", "matplotlib", "networkx", "numpy"]
    for pkg in required_packages:
        try:
            __import__(pkg)
            logger.info(f"✓ {pkg} available")
        except ImportError:
            logger.error(f"✗ {pkg} not installed")
            checks_passed = False

    # Check 4: Disk space
    import shutil
    disk_stats = shutil.disk_usage("/")
    free_gb = disk_stats.free / 1e9
    if free_gb < 10:
        logger.warning(f"⚠ Low disk space: {free_gb:.1f} GB free")
    else:
        logger.info(f"✓ Disk space: {free_gb:.1f} GB free")

    logger.info("=" * 80)
    return checks_passed


def run_mechanistic_validation(output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Run validation of core mechanistic claim: Modularity → Cache → Latency.

    This generates data for:
    - Section 3.5 (Mechanistic Analysis)
    - Figure 7 (Correlation plots)
    - Table 2 (Correlation values)
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: MECHANISTIC VALIDATION")
    logger.info("Validates: Modularity → Cache Hit Rate → Latency")
    logger.info("=" * 80)

    validation_output = output_dir / "mechanistic_validation"
    validation_output.mkdir(parents=True, exist_ok=True)

    # For each model in experimental matrix
    models = args.models if hasattr(args, 'models') and args.models else ["mamba"]

    results = {}
    for model_name in models:
        logger.info(f"\n--- Validating {model_name} ---")

        config_map = {
            "mamba": "configs/mamba.json",
            "bert": "configs/bert_large.json",
            "resnet": "configs/resnet50.json",
            "gcn": "configs/gcn_arxiv.json",
        }

        config_path = config_map.get(model_name, f"configs/{model_name}.json")
        output_file = validation_output / f"{model_name}_validation.json"

        try:
            # Run validation script
            import subprocess
            cmd = [
                sys.executable,
                "scripts/validate_mechanistic_claim.py",
                "--config", config_path,
                "--device", args.device,
                "--num-permutations", str(args.num_permutations),
                "--output", str(output_file),
            ]

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0:
                logger.info(f"✓ {model_name} validation complete")
                with open(output_file) as f:
                    results[model_name] = json.load(f)
            else:
                logger.error(f"✗ {model_name} validation failed")
                logger.error(result.stderr)

        except Exception as e:
            logger.error(f"Error validating {model_name}: {e}")

    return results


def run_experimental_matrix(output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Run complete experimental matrix: Dense, Algo-Only, Linear, Iterative.

    This generates data for:
    - Table 1 (Main Results)
    - Figure 6 (Pareto Frontier)
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: EXPERIMENTAL MATRIX")
    logger.info("Baselines: Dense, Algorithm-Only, Linear, Iterative")
    logger.info("=" * 80)

    matrix_output = output_dir / "experimental_matrix"
    matrix_output.mkdir(parents=True, exist_ok=True)

    # Run experiments using existing infrastructure
    try:
        import subprocess

        cmd = [
            sys.executable,
            "experiments/scripts/run_experimental_matrix.py",
            "--output-root", str(matrix_output),
            "--experiments", "core",  # Core experiments from paper
        ]

        if args.quick:
            cmd.extend(["--num-runs", "2", "--quick"])
        else:
            cmd.extend(["--num-runs", "5"])  # Paper uses n=5

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        if result.returncode == 0:
            logger.info("✓ Experimental matrix complete")
            return {"status": "success", "output_dir": str(matrix_output)}
        else:
            logger.error("✗ Experimental matrix failed")
            logger.error(result.stderr)
            return {"status": "failed", "error": result.stderr}

    except Exception as e:
        logger.error(f"Error running experimental matrix: {e}")
        return {"status": "error", "error": str(e)}


def generate_tables_and_figures(
    mechanistic_results: Dict[str, Any],
    matrix_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate paper tables and figures from experimental results."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: GENERATING TABLES AND FIGURES")
    logger.info("=" * 80)

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: Main Results (Linear vs Iterative)
    logger.info("Generating Table 1 (Main Results)...")
    try:
        # Use existing aggregation script
        import subprocess
        cmd = [
            sys.executable,
            "experiments/scripts/aggregate_table1.py",
            "--input", str(output_dir / "experimental_matrix"),
            "--output", str(tables_dir / "table1_main_results.csv"),
        ]
        subprocess.run(cmd, check=True, timeout=300)
        logger.info("✓ Table 1 generated")
    except Exception as e:
        logger.error(f"✗ Table 1 generation failed: {e}")

    # Table 2: Mechanistic Correlations
    logger.info("Generating Table 2 (Correlations)...")
    try:
        table2_data = []
        for model, results in mechanistic_results.items():
            if "correlations" in results:
                corr = results["correlations"]
                table2_data.append({
                    "Model": model,
                    "Q ↔ L2": corr.get("modularity_vs_l2", "N/A"),
                    "L2 ↔ Latency": corr.get("l2_vs_latency", "N/A"),
                    "Q ↔ Latency": corr.get("modularity_vs_latency", "N/A"),
                })

        with open(tables_dir / "table2_correlations.json", "w") as f:
            json.dump(table2_data, f, indent=2)
        logger.info("✓ Table 2 generated")
    except Exception as e:
        logger.error(f"✗ Table 2 generation failed: {e}")

    # Figures: Already generated by validation scripts
    logger.info("Validation plots available in mechanistic_validation/")


def generate_summary_report(
    mechanistic_results: Dict[str, Any],
    matrix_results: Dict[str, Any],
    output_dir: Path,
    duration_seconds: float,
) -> None:
    """Generate comprehensive summary report."""
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("=" * 80)

    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "duration_seconds": duration_seconds,
            "duration_hours": duration_seconds / 3600,
        },
        "mechanistic_validation": mechanistic_results,
        "experimental_matrix": matrix_results,
        "paper_claims_validated": {},
    }

    # Check if results validate paper claims
    validations = []

    # Claim 1: Modularity correlates with cache (r > 0.7)
    for model, results in mechanistic_results.items():
        if "correlations" in results:
            r_Q_l2 = results["correlations"].get("modularity_vs_l2", 0)
            validates = r_Q_l2 > 0.7
            validations.append({
                "claim": f"Modularity → L2 correlation (r > 0.7) for {model}",
                "measured_value": r_Q_l2,
                "validates": validates,
            })

            # Claim 2: Total effect Q → Latency (r ≈ -0.88)
            r_Q_lat = results["correlations"].get("modularity_vs_latency", 0)
            validates = abs(r_Q_lat + 0.88) < 0.2  # Within ±0.2 of -0.88
            validations.append({
                "claim": f"Modularity → Latency correlation (r ≈ -0.88) for {model}",
                "paper_claim": -0.88,
                "measured_value": r_Q_lat,
                "validates": validates,
            })

    report["paper_claims_validated"] = validations

    # Save report
    report_path = output_dir / "SUMMARY_REPORT.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total duration: {duration_seconds / 3600:.2f} hours")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"\nClaims Validated:")
    for v in validations:
        status = "✓" if v["validates"] else "✗"
        logger.info(f"  {status} {v['claim']}")
        logger.info(f"     Measured: {v['measured_value']:.3f}")

    logger.info(f"\nFull report saved to: {report_path}")
    logger.info(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ALL paper data with real hardware measurements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full validation (several hours)
  python scripts/generate_paper_data.py --output results/paper_data

  # Quick test (reduced samples)
  python scripts/generate_paper_data.py --output results/test --quick

  # Specific models only
  python scripts/generate_paper_data.py --output results/mamba --models mamba bert
        """
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        type=Path,
        help="Output directory for all generated data"
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mamba", "bert", "resnet", "gcn"],
        default=["mamba", "bert"],
        help="Models to validate (default: mamba bert)"
    )

    parser.add_argument(
        "--num-permutations",
        type=int,
        default=20,
        help="Number of permutations for mechanistic validation (default: 20)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (reduced samples, faster)"
    )

    parser.add_argument(
        "--skip-matrix",
        action="store_true",
        help="Skip experimental matrix (only run mechanistic validation)"
    )

    args = parser.parse_args()

    # Setup
    start_time = time.time()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info("PAPER DATA GENERATION - FULL PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Models: {', '.join(args.models)}")
    logger.info(f"Quick mode: {args.quick}")
    logger.info("=" * 80 + "\n")

    # Check prerequisites
    if not check_prerequisites():
        logger.error("\nPrerequisite checks failed. Please install required dependencies.")
        logger.info("\nSee docs/Hardware_Profiling_Integration.md for setup instructions.")
        return 1

    try:
        # Step 1: Mechanistic validation
        mechanistic_results = run_mechanistic_validation(output_dir, args)

        # Step 2: Experimental matrix
        if not args.skip_matrix:
            matrix_results = run_experimental_matrix(output_dir, args)
        else:
            matrix_results = {"status": "skipped"}

        # Step 3: Generate tables and figures
        generate_tables_and_figures(mechanistic_results, matrix_results, output_dir)

        # Step 4: Summary report
        duration = time.time() - start_time
        generate_summary_report(mechanistic_results, matrix_results, output_dir, duration)

        logger.info("\n✓ ALL DATA GENERATION COMPLETE")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("\nNext steps:")
        logger.info("1. Review SUMMARY_REPORT.json")
        logger.info("2. Check tables/ and figures/ directories")
        logger.info("3. Update paper with real data from these results")
        logger.info("4. Verify all claims are validated\n")

        return 0

    except KeyboardInterrupt:
        logger.warning("\n\nInterrupted by user. Partial results may be available.")
        return 130

    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
