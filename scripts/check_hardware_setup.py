#!/usr/bin/env python3
"""Check hardware setup for running real measurements.

This script verifies that all required hardware and software is available
for generating real experimental data.
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def check_cuda():
    """Check CUDA availability."""
    logger.info("Checking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"  ✓ CUDA available: {device_count} device(s)")
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1e9
                compute = f"{props.major}.{props.minor}"
                logger.info(f"    - Device {i}: {name}")
                logger.info(f"      Memory: {memory_gb:.1f} GB, Compute: {compute}")
            return True
        else:
            logger.error("  ✗ CUDA not available")
            logger.error("    Install CUDA-enabled PyTorch:")
            logger.error("    pip install torch --index-url https://download.pytorch.org/whl/cu118")
            return False
    except ImportError:
        logger.error("  ✗ PyTorch not installed")
        logger.error("    Install: pip install torch")
        return False


def check_ncu():
    """Check Nsight Compute availability."""
    logger.info("\nChecking Nsight Compute...")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from icd.measure.l2_ncu import find_ncu_binary

    ncu_path = find_ncu_binary()
    if ncu_path:
        try:
            result = subprocess.run(
                [ncu_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            version_info = result.stdout.strip() if result.returncode == 0 else "Unknown"
            logger.info(f"  ✓ Nsight Compute found: {ncu_path}")
            logger.info(f"    Version: {version_info}")
            return True
        except Exception as e:
            logger.warning(f"  ⚠ NCU found but cannot execute: {e}")
            return False
    else:
        logger.warning("  ⚠ Nsight Compute not found")
        logger.warning("    L2 cache profiling will not be available")
        logger.warning("    Download from: https://developer.nvidia.com/nsight-compute")
        logger.warning("    Or set ICD_NCU_PATH environment variable")
        return False


def check_dependencies():
    """Check Python dependencies."""
    logger.info("\nChecking Python dependencies...")

    required = {
        "scipy": "Statistical analysis",
        "matplotlib": "Figure generation",
        "networkx": "Graph algorithms",
        "numpy": "Numerical computing",
    }

    all_available = True
    for pkg, description in required.items():
        try:
            __import__(pkg)
            logger.info(f"  ✓ {pkg}: {description}")
        except ImportError:
            logger.error(f"  ✗ {pkg} not installed ({description})")
            all_available = False

    if not all_available:
        logger.error("\nInstall missing packages:")
        logger.error("  pip install scipy matplotlib networkx numpy")

    return all_available


def check_disk_space():
    """Check available disk space."""
    logger.info("\nChecking disk space...")
    try:
        import shutil
        usage = shutil.disk_usage("/")
        free_gb = usage.free / 1e9
        total_gb = usage.total / 1e9
        used_pct = (usage.used / usage.total) * 100

        logger.info(f"  Total: {total_gb:.1f} GB")
        logger.info(f"  Free: {free_gb:.1f} GB ({100-used_pct:.1f}%)")

        if free_gb < 10:
            logger.warning("  ⚠ Low disk space (< 10 GB)")
            logger.warning("    Large experiments may fail")
            return False
        else:
            logger.info("  ✓ Sufficient disk space")
            return True
    except Exception as e:
        logger.warning(f"  ⚠ Could not check disk space: {e}")
        return True


def print_recommendations(cuda_ok, ncu_ok, deps_ok, disk_ok):
    """Print recommendations based on check results."""
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)

    if cuda_ok and ncu_ok and deps_ok and disk_ok:
        logger.info("✓ All checks passed! You can run real measurements.")
        logger.info("\nNext steps:")
        logger.info("  1. Run validation:")
        logger.info("     python scripts/validate_mechanistic_claim.py --config configs/mamba.json")
        logger.info("\n  2. Generate all paper data:")
        logger.info("     python scripts/generate_paper_data.py --output results/paper_data")
        logger.info("\n  3. See docs/Hardware_Profiling_Integration.md for details")
        return 0

    else:
        logger.info("⚠ Some checks failed. Address issues above before running measurements.")
        logger.info("\nWhat you CAN do:")
        logger.info("  - Review implementation code")
        logger.info("  - Run unit tests: pytest tests/")
        logger.info("  - Use mock mode for development")

        if not cuda_ok:
            logger.info("\nCUDA Setup:")
            logger.info("  1. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads")
            logger.info("  2. Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118")

        if not ncu_ok:
            logger.info("\nNsight Compute Setup:")
            logger.info("  1. Download: https://developer.nvidia.com/nsight-compute")
            logger.info("  2. Or set: export ICD_NCU_PATH=/path/to/ncu")

        if not deps_ok:
            logger.info("\nDependencies:")
            logger.info("  pip install scipy matplotlib networkx numpy")

        logger.info("\nSee docs/Hardware_Profiling_Integration.md for complete setup guide.")
        return 1


def main():
    logger.info("=" * 80)
    logger.info("HARDWARE SETUP CHECK - Iterative Co-Design")
    logger.info("=" * 80)

    cuda_ok = check_cuda()
    ncu_ok = check_ncu()
    deps_ok = check_dependencies()
    disk_ok = check_disk_space()

    return print_recommendations(cuda_ok, ncu_ok, deps_ok, disk_ok)


if __name__ == "__main__":
    sys.exit(main())
