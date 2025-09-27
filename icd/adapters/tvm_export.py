"""Utilities for exporting PyTorch models to TVM and running AutoTVM search.

The goal of this module is to provide an end-to-end bridge between PyTorch
models used throughout the project and the TVM compiler stack.  It supports
three stages:

1. Exporting a PyTorch module to TorchScript or ONNX.
2. Converting the exported module to TVM Relay and building a deployable
   runtime.
3. Running AutoTVM/Ansor style schedule searches and persisting tuning logs.

The code is written so that it can be executed even on systems where TVM is not
installed.  In that scenario we surface detailed error messages that explain the
missing dependencies and provide instructions for installation.  This mirrors
how the rest of the measurement stack handles optional backends (e.g. Nsight
Compute) and allows unit tests to exercise the control flow without a TVM
runtime.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    import tvm
    from tvm import auto_scheduler, autotvm, relay
    from tvm.contrib import graph_executor
except Exception as exc:  # pragma: no cover - optional dependency
    tvm = None  # type: ignore
    relay = None  # type: ignore
    graph_executor = None  # type: ignore
    auto_scheduler = None  # type: ignore
    autotvm = None  # type: ignore
    _TVM_IMPORT_ERROR = exc
else:
    _TVM_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)


class DependencyError(RuntimeError):
    """Raised when an optional dependency is missing."""


def _ensure_torch() -> None:
    if torch is None:
        raise DependencyError(
            "PyTorch is required for TVM export but is not importable.\n"
            f"Original import error: {_TORCH_IMPORT_ERROR}"
        )


def _ensure_tvm() -> None:
    if tvm is None:
        raise DependencyError(
            "TVM is required for compilation but is not importable.\n"
            "Install it via `pip install apache-tvm` or follow the upstream\n"
            "build instructions for GPU support.\n"
            f"Original import error: {_TVM_IMPORT_ERROR}"
        )


@dataclass
class ExportConfig:
    """Configuration for exporting a PyTorch model to TVM."""

    example_input: Any
    target: str = "llvm"
    tuning_trials: int = 0
    tuning_log: Optional[pathlib.Path] = None
    use_ansor: bool = False
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "tuning_trials": self.tuning_trials,
            "tuning_log": str(self.tuning_log) if self.tuning_log else None,
            "use_ansor": self.use_ansor,
            "input_shapes": self.input_shapes,
        }


def export_to_torchscript(module: "torch.nn.Module", example_input: Any) -> "torch.jit.ScriptModule":
    """Trace or script a PyTorch module for export.

    Parameters
    ----------
    module:
        The PyTorch module to export.
    example_input:
        Example inputs used for tracing the module.
    """

    _ensure_torch()

    if hasattr(module, "to"):
        module = module.to("cpu")
        module.eval()

    LOGGER.info("Tracing module %s with example input type %s", module.__class__.__name__, type(example_input))
    with torch.no_grad():
        traced = torch.jit.trace(module, example_input)
    return traced


def convert_to_relay(script_module: "torch.jit.ScriptModule", config: ExportConfig) -> Tuple[Any, Dict[str, Tuple[int, ...]]]:
    """Convert a TorchScript module to a Relay module and parameters."""

    _ensure_tvm()
    _ensure_torch()

    input_shapes = config.input_shapes
    if input_shapes is None:
        if not isinstance(config.example_input, (tuple, list)):
            example_inputs = (config.example_input,)
        else:
            example_inputs = tuple(config.example_input)
        input_shapes = {f"input_{idx}": tuple(inp.shape) for idx, inp in enumerate(example_inputs)}

    LOGGER.info("Converting TorchScript module to Relay with input shapes: %s", input_shapes)
    mod, params = relay.frontend.from_pytorch(script_module, list(input_shapes.items()))
    return mod, params


def build_runtime(
    relay_module: Any,
    relay_params: Dict[str, Any],
    config: ExportConfig,
) -> "graph_executor.GraphModule":
    """Build a Relay module with optional AutoTVM/Ansor tuning."""

    _ensure_tvm()

    target = tvm.target.Target(config.target)
    tuning_log = str(config.tuning_log) if config.tuning_log else None

    if config.use_ansor:
        LOGGER.info("Running Ansor search with %s trials", config.tuning_trials)
        task_extract = auto_scheduler.extract_tasks(relay_module["main"], params=relay_params, target=target)
        if not task_extract:
            LOGGER.warning("No auto-scheduler tasks were extracted from the Relay module.")
        database = auto_scheduler.measure_record.RecordToFile(tuning_log) if tuning_log else None
        search_policy = auto_scheduler.SketchPolicy()
        tuning_options = auto_scheduler.TuningOptions(
            num_measure_trials=config.tuning_trials,
            builder=auto_scheduler.LocalBuilder(),
            runner=auto_scheduler.LocalRunner(),
            measure_callbacks=[auto_scheduler.RecordToFile(tuning_log)] if tuning_log else [],
        )
        for task in task_extract:
            tuner = auto_scheduler.TaskScheduler([task], load_log_file=tuning_log)
            tuner.tune(tuning_options)
        with auto_scheduler.ApplyHistoryBest(tuning_log):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(relay_module, target=target, params=relay_params)
    else:
        if config.tuning_trials > 0:
            LOGGER.info("Running AutoTVM tuning with %s trials", config.tuning_trials)
            tasks = autotvm.task.extract_from_program(relay_module["main"], target=target, params=relay_params)
            if not tasks:
                LOGGER.warning("No AutoTVM tasks were extracted from the Relay module.")
            for task in tasks:
                tuner = autotvm.tuner.XGBTuner(task)
                tuner.tune(
                    n_trial=config.tuning_trials,
                    measure_option=autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=autotvm.LocalRunner()),
                    callbacks=[autotvm.callback.log_to_file(tuning_log)] if tuning_log else [],
                )
        with autotvm.apply_history_best(tuning_log) if tuning_log else contextlib.nullcontext():  # type: ignore[name-defined]
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(relay_module, target=target, params=relay_params)

    dev = tvm.device(str(target.kind.name), 0)
    module = graph_executor.GraphModule(lib["default"].create(dev))
    return module


def save_artifacts(
    graph_module: "graph_executor.GraphModule",
    output_dir: pathlib.Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist the compiled runtime and optional metadata to disk."""

    _ensure_tvm()

    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = output_dir / "deploy_lib.tar"
    graph_path = output_dir / "deploy_graph.json"
    params_path = output_dir / "deploy_params.bin"
    meta_path = output_dir / "metadata.json"

    LOGGER.info("Saving TVM artifacts to %s", output_dir)
    graph_module.module.lib.export_library(str(lib_path))
    graph = graph_module.get_graph_json()
    params = graph_module.get_params()

    graph_path.write_text(graph)
    params_path.write_bytes(tvm.runtime.save_param_dict(params))
    if metadata is not None:
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def compile_pytorch_module(
    module: "torch.nn.Module",
    config: ExportConfig,
    artifacts_dir: Optional[pathlib.Path] = None,
) -> Optional["graph_executor.GraphModule"]:
    """High level convenience wrapper.

    Returns the built runtime or ``None`` when running in dry-run mode with
    missing dependencies.
    """

    try:
        script_module = export_to_torchscript(module, config.example_input)
        relay_module, relay_params = convert_to_relay(script_module, config)
        graph_module = build_runtime(relay_module, relay_params, config)
    except DependencyError as exc:
        LOGGER.error("Missing dependency for TVM compilation: %s", exc)
        return None

    if artifacts_dir is not None:
        save_artifacts(graph_module, artifacts_dir, metadata={"config": config.as_dict()})
    return graph_module


def verify_runtime(
    graph_module: "graph_executor.GraphModule",
    inputs: Iterable[Tuple[str, Any]],
    reference: Optional[Iterable[Any]] = None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> Dict[str, Any]:
    """Run inference with the compiled runtime and optionally compare outputs."""

    _ensure_torch()
    _ensure_tvm()

    input_list = list(inputs)
    LOGGER.info("Running inference on TVM runtime with %d inputs", len(input_list))
    for name, value in input_list:
        graph_module.set_input(name, value)
    graph_module.run()
    outputs = {}
    for idx in range(graph_module.get_num_outputs()):
        outputs[f"output_{idx}"] = graph_module.get_output(idx).numpy()

    if reference is not None:
        for idx, (tvm_out, ref_out) in enumerate(zip(outputs.values(), reference)):
            if not torch.allclose(torch.from_numpy(tvm_out), torch.from_numpy(ref_out), atol=atol, rtol=rtol):
                raise AssertionError(f"TVM output mismatch for output_{idx}")
    return outputs


def run_relay_cli(
    relay_path: pathlib.Path,
    target: str,
    output_dir: pathlib.Path,
    tuning_log: Optional[pathlib.Path] = None,
    python_path: Optional[Iterable[pathlib.Path]] = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke ``tvmc`` on a Relay file for reproducibility."""

    _ensure_tvm()

    cmd = ["tvmc", "compile", str(relay_path), "--target", target, "--output", str(output_dir / "module.tar")]
    if tuning_log:
        cmd.extend(["--tuning-records", str(tuning_log)])
    env = os.environ.copy()
    if python_path:
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in python_path)
    LOGGER.info("Running external TVMC command: %s", " ".join(cmd))
    return subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)


__all__ = [
    "ExportConfig",
    "compile_pytorch_module",
    "convert_to_relay",
    "export_to_torchscript",
    "build_runtime",
    "save_artifacts",
    "verify_runtime",
    "run_relay_cli",
]
