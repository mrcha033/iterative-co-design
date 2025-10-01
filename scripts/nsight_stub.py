"""Generate a deterministic Nsight Compute style payload for CI.

This stub emulates the subset of the Nsight Compute (``ncu``) JSON format that
our parsing utilities rely on.  It allows unit tests and CPU-only environments
to exercise the L2 cache reporting pipeline without requiring the proprietary
profiler.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Mapping

__all__ = ["run_stub"]


def _kernel_section(name: str, hit_rate: float, dram_pct: float) -> Mapping[str, object]:
    """Construct a minimal section mimicking an ncu kernel entry."""

    return {
        "name": name,
        "metrics": {
            "l2_tex__t_sector_hit_rate.pct": hit_rate,
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": dram_pct,
        },
    }


def _summary(sections: Iterable[Mapping[str, object]]) -> Mapping[str, object]:
    sections = list(sections)
    launches = len(sections)
    if launches:
        hit_values = [float(sec["metrics"]["l2_tex__t_sector_hit_rate.pct"]) for sec in sections]
        avg_hit = sum(hit_values) / launches
    else:
        avg_hit = float("nan")
    return {"launches": launches, "avg_l2_hit_pct": avg_hit}


def run_stub(output_path: str | Path, kernels: Iterable[str] | None = None) -> Mapping[str, object]:
    """Write a deterministic Nsight-style JSON payload to ``output_path``.

    Args:
        output_path: Destination file for the JSON payload.
        kernels: Optional iterable of kernel names.  The length of this iterable
            controls the ``launches`` count in the emitted summary.

    Returns:
        The JSON payload that was written to disk.
    """

    kernel_names: List[str] = list(kernels) if kernels is not None else ["stub_kernel"]

    # Deterministic but non-trivial metric values so downstream statistics have
    # stable behaviour across runs.
    sections = [
        _kernel_section(name, 80.0 + idx * 1.5, 25.0 + idx * 2.0)
        for idx, name in enumerate(kernel_names)
    ]

    payload = {
        "summary": _summary(sections),
        "children": [{"metrics": section["metrics"]} for section in sections],
        "kernels": sections,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
