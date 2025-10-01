from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ValidationConfig:
    """Configuration for running the full hardware validation pipeline."""

    output_dir: Path | str
    device: str = "cuda"
    models: List[str] = field(default_factory=lambda: ["mamba", "bert"])
    num_permutations: int = 20
    quick: bool = False
    skip_matrix: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        self.models = list(self.models)


@dataclass
class ValidationRunResult:
    """Summary of a validation pipeline invocation."""

    returncode: int
    output_dir: Path
    duration_seconds: float
    mechanistic_results: Dict[str, Any]
    matrix_results: Dict[str, Any]
    summary_report: Optional[Path]


def _log(logger_or_none: Optional[logging.Logger]) -> logging.Logger:
    return logger_or_none or logger


def check_prerequisites(log: Optional[logging.Logger] = None) -> bool:
    """Verify that the environment satisfies hardware validation requirements."""

    log = _log(log)
    log.info("=" * 80)
    log.info("CHECKING PREREQUISITES")
    log.info("=" * 80)

    checks_passed = True

    try:
        import torch  # pylint: disable=import-error

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info("✓ PyTorch CUDA available: %s (%.1f GB)", device_name, memory_gb)
        else:
            log.error("✗ CUDA not available in PyTorch")
            checks_passed = False
    except ImportError:
        log.error("✗ PyTorch not installed")
        checks_passed = False

    from icd.measure.l2_ncu import find_ncu_binary

    ncu_path = find_ncu_binary()
    if ncu_path:
        log.info("✓ Nsight Compute found: %s", ncu_path)
    else:
        log.warning("⚠ Nsight Compute not found - L2 profiling will be skipped")
        log.warning("  Install from: https://developer.nvidia.com/nsight-compute")

    required_packages = ["scipy", "matplotlib", "networkx", "numpy"]
    for pkg in required_packages:
        try:
            __import__(pkg)
            log.info("✓ %s available", pkg)
        except ImportError:
            log.error("✗ %s not installed", pkg)
            checks_passed = False

    import shutil

    disk_stats = shutil.disk_usage("/")
    free_gb = disk_stats.free / 1e9
    if free_gb < 10:
        log.warning("⚠ Low disk space: %.1f GB free", free_gb)
    else:
        log.info("✓ Disk space: %.1f GB free", free_gb)

    log.info("=" * 80)
    return checks_passed


def run_mechanistic_validation(
    config: ValidationConfig, log: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Execute mechanistic validation for the configured models."""

    log = _log(log)
    log.info("\n" + "=" * 80)
    log.info("STEP 1: MECHANISTIC VALIDATION")
    log.info("Validates: Modularity → Cache Hit Rate → Latency")
    log.info("=" * 80)

    validation_output = config.output_dir / "mechanistic_validation"
    validation_output.mkdir(parents=True, exist_ok=True)

    models = config.models if config.models else ["mamba"]

    results: Dict[str, Any] = {}

    config_map = {
        "mamba": "configs/mamba.json",
        "bert": "configs/bert_large.json",
        "resnet": "configs/resnet50.json",
        "gcn": "configs/gcn_arxiv.json",
    }

    for model_name in models:
        log.info("\n--- Validating %s ---", model_name)

        config_path = config_map.get(model_name, f"configs/{model_name}.json")
        output_file = validation_output / f"{model_name}_validation.json"

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validate_mechanistic_claim.py"),
            "--config",
            config_path,
            "--device",
            config.device,
            "--num-permutations",
            str(config.num_permutations),
            "--output",
            str(output_file),
        ]

        log.info("Running: %s", " ".join(cmd))

        try:
            result = subprocess.run(  # noqa: S603, S607
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=PROJECT_ROOT,
                check=False,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error("Error validating %s: %s", model_name, exc)
            continue

        if result.returncode == 0:
            log.info("✓ %s validation complete", model_name)
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    results[model_name] = json.load(f)
            except FileNotFoundError:
                log.error("✗ Expected output missing for %s", model_name)
        else:
            log.error("✗ %s validation failed", model_name)
            if result.stdout:
                log.error(result.stdout)
            if result.stderr:
                log.error(result.stderr)

    return results


def run_experimental_matrix(
    config: ValidationConfig, log: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Run the experimental matrix used in the publication."""

    log = _log(log)
    log.info("\n" + "=" * 80)
    log.info("STEP 2: EXPERIMENTAL MATRIX")
    log.info("Baselines: Dense, Algorithm-Only, Linear, Iterative")
    log.info("=" * 80)

    matrix_output = config.output_dir / "experimental_matrix"
    matrix_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "scripts" / "run_experimental_matrix.py"),
        "--output-root",
        str(matrix_output),
        "--experiments",
        "core",
    ]

    if config.quick:
        cmd.extend(["--num-runs", "2", "--quick"])
    else:
        cmd.extend(["--num-runs", "5"])

    log.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(  # noqa: S603, S607
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
            cwd=PROJECT_ROOT,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.error("Error running experimental matrix: %s", exc)
        return {"status": "error", "error": str(exc)}

    if result.returncode == 0:
        log.info("✓ Experimental matrix complete")
        return {"status": "success", "output_dir": str(matrix_output)}

    log.error("✗ Experimental matrix failed")
    if result.stdout:
        log.error(result.stdout)
    if result.stderr:
        log.error(result.stderr)
    return {"status": "failed", "error": result.stderr}


def generate_tables_and_figures(
    mechanistic_results: Dict[str, Any],
    matrix_results: Dict[str, Any],
    config: ValidationConfig,
    log: Optional[logging.Logger] = None,
) -> None:
    """Create derived tables and figures from experiment outputs."""

    log = _log(log)
    log.info("\n" + "=" * 80)
    log.info("STEP 3: GENERATING TABLES AND FIGURES")
    log.info("=" * 80)

    tables_dir = config.output_dir / "tables"
    figures_dir = config.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating Table 1 (Main Results)...")
    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "scripts" / "aggregate_table1.py"),
            "--input",
            str(config.output_dir / "experimental_matrix"),
            "--output",
            str(tables_dir / "table1_main_results.csv"),
        ]
        subprocess.run(  # noqa: S603, S607
            cmd,
            check=True,
            timeout=300,
            cwd=PROJECT_ROOT,
        )
        log.info("✓ Table 1 generated")
    except Exception as exc:  # pragma: no cover - aggregation is optional
        log.error("✗ Table 1 generation failed: %s", exc)

    log.info("Generating Table 2 (Correlations)...")
    try:
        table2_data = []
        for model, results in mechanistic_results.items():
            if "correlations" in results:
                corr = results["correlations"]
                table2_data.append(
                    {
                        "Model": model,
                        "Q ↔ L2": corr.get("modularity_vs_l2", "N/A"),
                        "L2 ↔ Latency": corr.get("l2_vs_latency", "N/A"),
                        "Q ↔ Latency": corr.get("modularity_vs_latency", "N/A"),
                    }
                )

        with open(tables_dir / "table2_correlations.json", "w", encoding="utf-8") as f:
            json.dump(table2_data, f, indent=2)
        log.info("✓ Table 2 generated")
    except Exception as exc:  # pragma: no cover - defensive
        log.error("✗ Table 2 generation failed: %s", exc)

    log.info("Validation plots available in mechanistic_validation/")


def generate_summary_report(
    mechanistic_results: Dict[str, Any],
    matrix_results: Dict[str, Any],
    config: ValidationConfig,
    duration_seconds: float,
    log: Optional[logging.Logger] = None,
) -> Path:
    """Create a summary report JSON that collates validation artefacts."""

    from datetime import datetime

    log = _log(log)
    log.info("\n" + "=" * 80)
    log.info("GENERATING SUMMARY REPORT")
    log.info("=" * 80)

    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "duration_seconds": duration_seconds,
            "duration_hours": duration_seconds / 3600,
        },
        "mechanistic_validation": mechanistic_results,
        "experimental_matrix": matrix_results,
        "paper_claims_validated": [],
    }

    validations = []
    for model, results in mechanistic_results.items():
        correlations = results.get("correlations")
        if not correlations:
            continue

        r_q_l2 = correlations.get("modularity_vs_l2", 0)
        validations.append(
            {
                "claim": f"Modularity → L2 correlation (r > 0.7) for {model}",
                "measured_value": r_q_l2,
                "validates": r_q_l2 > 0.7,
            }
        )

        r_q_latency = correlations.get("modularity_vs_latency", 0)
        validations.append(
            {
                "claim": f"Modularity → Latency correlation (r ≈ -0.88) for {model}",
                "paper_claim": -0.88,
                "measured_value": r_q_latency,
                "validates": abs(r_q_latency + 0.88) < 0.2,
            }
        )

    report["paper_claims_validated"] = validations

    report_path = config.output_dir / "SUMMARY_REPORT.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info("\n%s", "=" * 80)
    log.info("SUMMARY")
    log.info("%s", "=" * 80)
    log.info("Total duration: %.2f hours", duration_seconds / 3600)
    log.info("Output directory: %s", config.output_dir)
    log.info("\nClaims Validated:")
    for entry in validations:
        status = "✓" if entry["validates"] else "✗"
        log.info("  %s %s", status, entry["claim"])
        log.info("     Measured: %.3f", entry["measured_value"])

    log.info("\nFull report saved to: %s", report_path)
    log.info("%s\n", "=" * 80)

    return report_path


def run_full_validation(
    config: ValidationConfig, log: Optional[logging.Logger] = None
) -> ValidationRunResult:
    """Run the complete real-hardware validation pipeline."""

    log = _log(log)
    start_time = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("\n" + "=" * 80)
    log.info("PAPER DATA GENERATION - FULL PIPELINE")
    log.info("=" * 80)
    log.info("Output directory: %s", output_dir)
    log.info("Device: %s", config.device)
    log.info("Models: %s", ", ".join(config.models))
    log.info("Quick mode: %s", config.quick)
    log.info("=" * 80 + "\n")

    if not check_prerequisites(log):
        log.error("\nPrerequisite checks failed. Please install required dependencies.")
        log.info("\nSee docs/Hardware_Profiling_Integration.md for setup instructions.")
        duration = time.time() - start_time
        return ValidationRunResult(
            returncode=1,
            output_dir=output_dir,
            duration_seconds=duration,
            mechanistic_results={},
            matrix_results={},
            summary_report=None,
        )

    try:
        mechanistic_results = run_mechanistic_validation(config, log)

        if config.skip_matrix:
            matrix_results: Dict[str, Any] = {"status": "skipped"}
        else:
            matrix_results = run_experimental_matrix(config, log)

        generate_tables_and_figures(mechanistic_results, matrix_results, config, log)

        duration = time.time() - start_time
        summary_path = generate_summary_report(
            mechanistic_results, matrix_results, config, duration, log
        )

        log.info("\n✓ ALL DATA GENERATION COMPLETE")
        log.info("Results saved to: %s", output_dir)
        log.info("\nNext steps:")
        log.info("1. Review SUMMARY_REPORT.json")
        log.info("2. Check tables/ and figures/ directories")
        log.info("3. Update paper with real data from these results")
        log.info("4. Verify all claims are validated\n")

        return ValidationRunResult(
            returncode=0,
            output_dir=output_dir,
            duration_seconds=duration,
            mechanistic_results=mechanistic_results,
            matrix_results=matrix_results,
            summary_report=summary_path,
        )
    except KeyboardInterrupt:
        duration = time.time() - start_time
        log.warning("\n\nInterrupted by user. Partial results may be available.")
        return ValidationRunResult(
            returncode=130,
            output_dir=output_dir,
            duration_seconds=duration,
            mechanistic_results={},
            matrix_results={},
            summary_report=None,
        )
    except Exception as exc:  # pragma: no cover - defensive
        duration = time.time() - start_time
        log.error("\n\nFatal error: %s", exc, exc_info=True)
        return ValidationRunResult(
            returncode=1,
            output_dir=output_dir,
            duration_seconds=duration,
            mechanistic_results={},
            matrix_results={},
            summary_report=None,
        )

