#!/usr/bin/env python3
"""Generate empirical data for the paper using the integrated validation pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from icd.runtime.validation import ValidationConfig, run_full_validation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ALL paper data with real hardware measurements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Full validation (several hours)
  python scripts/generate_paper_data.py --output results/paper_data

  # Quick test (reduced samples)
  python scripts/generate_paper_data.py --output results/test --quick

  # Specific models only
  python scripts/generate_paper_data.py --output results/mamba --models mamba bert
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Output directory for all generated data",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mamba", "bert", "resnet", "gcn"],
        default=["mamba", "bert"],
        help="Models to validate (default: mamba bert)",
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=20,
        help="Number of permutations for mechanistic validation (default: 20)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (reduced samples, faster)",
    )
    parser.add_argument(
        "--skip-matrix",
        action="store_true",
        help="Skip experimental matrix (only run mechanistic validation)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = ValidationConfig(
        output_dir=args.output,
        device=args.device,
        models=args.models,
        num_permutations=args.num_permutations,
        quick=args.quick,
        skip_matrix=args.skip_matrix,
    )

    result = run_full_validation(config)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

