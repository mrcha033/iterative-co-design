#!/usr/bin/env python3
"""Command line interface for running AutoTVM/Ansor searches.

This script provides a reproducible entrypoint for Phase 1 experiments.  It
loads a PyTorch module, exports it to TVM using :mod:`icd.adapters.tvm_export`,
and optionally saves the resulting runtime artifacts and tuning logs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import pathlib
from typing import Any

from icd.adapters.tvm_export import ExportConfig, compile_pytorch_module

LOGGER = logging.getLogger("run_autotvm")


def resolve_module(path: str) -> Any:
    if ":" not in path:
        raise ValueError("Model path must be in the form 'module:object'")
    module_name, attr_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Python path to the torch.nn.Module factory (module:callable)")
    parser.add_argument("--example-shape", nargs="*", type=int, default=[1, 3, 224, 224], help="Input tensor shape")
    parser.add_argument("--target", default="llvm", help="TVM target triple")
    parser.add_argument("--tuning-trials", type=int, default=0, help="Number of AutoTVM trials")
    parser.add_argument("--use-ansor", action="store_true", help="Use the auto-scheduler instead of AutoTVM")
    parser.add_argument("--tuning-log", type=pathlib.Path, default=None, help="Path to save tuning logs")
    parser.add_argument("--artifacts", type=pathlib.Path, default=None, help="Directory to store compiled artifacts")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    model_factory = resolve_module(args.model)
    model = model_factory()

    import torch

    example_input = torch.randn(*args.example_shape)
    config = ExportConfig(
        example_input=example_input,
        target=args.target,
        tuning_trials=args.tuning_trials,
        tuning_log=args.tuning_log,
        use_ansor=args.use_ansor,
    )

    graph_module = compile_pytorch_module(model, config, artifacts_dir=args.artifacts)
    if graph_module is None:
        LOGGER.error("TVM dependencies missing. Aborting.")
        return

    outputs = graph_module.get_output(0).numpy() if graph_module.get_num_outputs() > 0 else None
    summary = {
        "config": config.as_dict(),
        "outputs_sample": outputs.tolist() if outputs is not None else None,
    }
    if args.artifacts:
        summary_path = args.artifacts / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        LOGGER.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
