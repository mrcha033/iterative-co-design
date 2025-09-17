"""Built-in runner implementations used by mock configs and tests."""

from __future__ import annotations

import time
from typing import Any, Dict


def mock_inference(mode: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic mock inference runner used for CI.

    The function emulates latency differences between ``linear`` and
    ``iterative`` modes by sleeping for slightly different durations.  It also
    returns synthetic L2 hit deltas so that higher layers can populate
    ``metrics.l2_hit_pct`` without relying on Nsight tooling in CI.
    """

    tokens = int(context.get("tokens", 1024))

    provide_l2 = bool(context.get("provide_l2", False))
    provide_ept = bool(context.get("provide_ept", False))

    if mode == "iterative":
        delay = float(context.get("iter_delay_s", 0.00009))
        l2 = float(context.get("iter_l2_hit", 0.88)) if provide_l2 else None
        ept = float(context.get("iter_ept", 0.92)) if provide_ept else None
    else:
        delay = float(context.get("linear_delay_s", 0.0001))
        l2 = float(context.get("linear_l2_hit", 0.82)) if provide_l2 else None
        ept = float(context.get("linear_ept", 1.0)) if provide_ept else None

    time.sleep(delay)
    result: Dict[str, Any] = {"tokens": tokens}
    if l2 is not None:
        result["l2_hit_pct"] = l2
    if ept is not None:
        result["ept_j_per_tok"] = ept
    return result


__all__ = ["mock_inference"]
