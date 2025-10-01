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
from typing import Any, Dict, Iterable, Tuple

import torch

from icd.adapters.tvm_export import ExportConfig, compile_pytorch_module, verify_runtime
from icd.measure.latency import LatencyMeasurer

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
    parser.add_argument("--measure-repeats", type=int, default=100, help="Latency measurement iterations")
    parser.add_argument("--measure-warmup", type=int, default=10, help="Latency warmup iterations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    model_factory = resolve_module(args.model)
    model = model_factory()

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

    def _prepare_inputs(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: _prepare_inputs(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            converted = [_prepare_inputs(v) for v in obj]
            return type(obj)(converted)
        return obj

    def _to_numpy(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        if isinstance(obj, dict):
            return {k: _to_numpy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            converted = [_to_numpy(v) for v in obj]
            return type(obj)(converted)
        return obj

    def _name_inputs(obj: Any) -> Iterable[Tuple[str, Any]]:
        if isinstance(obj, dict):
            return [(str(k), v) for k, v in obj.items()]
        if isinstance(obj, (list, tuple)):
            return [(f"input_{idx}", v) for idx, v in enumerate(obj)]
        return [("input_0", obj)]

    prepared_inputs = _prepare_inputs(example_input)
    named_inputs = list(_name_inputs(_to_numpy(prepared_inputs)))
    for key, value in named_inputs:
        graph_module.set_input(key, value)

    measurer = LatencyMeasurer(warmup_iter=args.measure_warmup, repeats=args.measure_repeats, fixed_clock=True, sync_gpu=False)
    latency = measurer.measure_callable(graph_module.run)

    verification: Dict[str, Any]
    reference_outputs = None
    with torch.no_grad():
        model_cpu = model.eval().cpu()
        if isinstance(prepared_inputs, dict):
            ref = model_cpu(**prepared_inputs)
        else:
            args_tuple = prepared_inputs if isinstance(prepared_inputs, (list, tuple)) else (prepared_inputs,)
            ref = model_cpu(*args_tuple)
        if isinstance(ref, torch.Tensor):
            reference_outputs = [ref.detach().cpu().numpy()]
        elif isinstance(ref, (list, tuple)):
            reference_outputs = [r.detach().cpu().numpy() for r in ref if isinstance(r, torch.Tensor)]

    try:
        verify_runtime(graph_module, named_inputs, reference=reference_outputs)
        verification = {"checked": True}
    except Exception as exc:  # pragma: no cover - depends on TVM runtime
        LOGGER.error("TVM verification failed: %s", exc)
        verification = {"checked": False, "detail": str(exc)}

    outputs = None
    if graph_module.get_num_outputs() > 0:
        outputs = graph_module.get_output(0).numpy().tolist()

    summary = {
        "config": config.as_dict(),
        "latency": latency,
        "verification": verification,
        "outputs_sample": outputs,
        "artifacts": str(args.artifacts) if args.artifacts else None,
    }

    if args.artifacts:
        summary_path = args.artifacts / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        LOGGER.info("Wrote summary to %s", summary_path)

        metadata_path = args.artifacts / "metadata.json"
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except json.JSONDecodeError:
                metadata = {}
        metadata.update({
            "config": config.as_dict(),
            "latency": latency,
            "verification": verification,
        })
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        LOGGER.info("Updated metadata at %s", metadata_path)

    LOGGER.info(
        "TVM latency: %.3f ms (repeats=%d, warmup=%d)",
        latency.get("mean"),
        args.measure_repeats,
        args.measure_warmup,
    )


if __name__ == "__main__":
    main()
