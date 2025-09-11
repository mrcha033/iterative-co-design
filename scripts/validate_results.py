#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def load_metrics(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: validate_results.py <run_dir_linear> <run_dir_iter>")
        return 2
    base_p = Path(argv[0])
    trial_p = Path(argv[1])
    bm = load_metrics(base_p / "metrics.json")
    tm = load_metrics(trial_p / "metrics.json")
    if not bm or not tm:
        print("Missing metrics.json in one or both run directories")
        return 2
    lat0 = bm.get("latency_ms", {}).get("mean")
    lat1 = tm.get("latency_ms", {}).get("mean")
    l20 = bm.get("l2_hit_pct")
    l21 = tm.get("l2_hit_pct")
    e0 = bm.get("ept_j_per_tok")
    e1 = tm.get("ept_j_per_tok")
    def f(x):
        return (x is not None) and isinstance(x, (int, float)) and math.isfinite(x)
    verdict = {
        "latency_rel": (lat1 - lat0) / lat0 if (f(lat0) and f(lat1) and lat0 != 0) else None,
        "l2_pp": (l21 - l20) if (f(l20) and f(l21)) else None,
        "ept_rel": (e1 - e0) / e0 if (f(e0) and f(e1) and e0 != 0) else None,
    }
    print(json.dumps(verdict, indent=2))
    # Smoke acceptance: at least latency improves or L2 improves
    ok = False
    if verdict["latency_rel"] is not None and verdict["latency_rel"] <= -0.05:
        ok = True
    if verdict["l2_pp"] is not None and verdict["l2_pp"] >= 2.0:
        ok = True
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

