"""Runtime runner resolution utilities.

Provides a small shim to load callables defined in configuration for
measurement runs.  The runner is expected to be a callable with the
signature:

    runner(mode: str, context: dict) -> Optional[dict]

The callable may execute arbitrary inference or benchmarking logic.  The
context dictionary contains run-time metadata (permutation, config, output
directory, etc.) that callers may use.  The return value is optional but can
include auxiliary metrics such as ``tokens`` or ``l2_hit_pct`` that will be
propagated into the final metrics payload when present.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from icd.utils.imports import load_object

RunnerCallable = Callable[[str, Dict[str, Any]], Any]


def _load_from_dotted_path(path: str) -> Any:
    """Backward-compatible shim for legacy imports (delegates to utils)."""

    return load_object(path)


def resolve_runner(pipeline_cfg: Dict[str, Any]) -> RunnerCallable | None:
    """Resolve the runner callable from pipeline configuration.

    Returns ``None`` when no runner is configured.  Raises ``ValueError`` when
    the specification is malformed or does not resolve to a callable.
    """

    runner_spec = (pipeline_cfg or {}).get("runner")
    if runner_spec is None:
        return None
    if callable(runner_spec):
        return runner_spec  # already a callable (useful for tests)
    runner = _load_from_dotted_path(str(runner_spec))
    if not callable(runner):
        raise ValueError(f"configured runner '{runner_spec}' is not callable")
    return runner


def prepare_runner_context(**kwargs: Any) -> Dict[str, Any]:
    """Create a shallow copy of runner context.

    Callers may mutate the returned dictionary without affecting the
    orchestrator's internal state.
    """

    ctx: Dict[str, Any] = {}
    ctx.update(kwargs)
    return ctx


__all__ = ["RunnerCallable", "resolve_runner", "prepare_runner_context"]
