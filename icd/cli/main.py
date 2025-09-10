from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from icd.runtime.orchestrator import run as run_pipeline
from icd.runtime.orchestrator import run_pair as run_pipeline_pair
from icd.runtime.orchestrator import run_pair as run_pipeline_pair


def parse_override(override: str) -> Dict[str, Any]:
    # very small parser: key.path=a:b:c → nested dict
    key, val = override.split("=", 1)
    # try parse JSON-ish for simple types
    try:
        parsed = json.loads(val)
    except json.JSONDecodeError:
        parsed = val
    d: Dict[str, Any] = {}
    cur = d
    parts = key.split(".")
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = parsed
    return d


def deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_update(a[k], v)
        else:
            a[k] = v
    return a


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="icd")
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run", help="Run ICD pipeline")
    runp.add_argument("-c", "--config", required=True, help="JSON config path")
    runp.add_argument("--override", action="append", default=[], help="Override key=val (dot.path)")
    runp.add_argument("--out", required=True, help="Output directory")
    pairp = sub.add_parser("pair", help="Run baseline+trial and compare")
    pairp.add_argument("-c", "--config", required=True, help="JSON config path")
    pairp.add_argument("--override", action="append", default=[], help="Override key=val (dot.path)")
    pairp.add_argument("--out", required=True, help="Output directory for pair run root")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        with open(args.config, "r", encoding="utf-8") as f:
            cfg: Dict[str, Any] = json.load(f)
        for ov in args.override:
            deep_update(cfg, parse_override(ov))
        cfg.setdefault("report", {})["out_dir"] = args.out
        run_pipeline(cfg)
        return 0
    elif args.cmd == "pair":
        with open(args.config, "r", encoding="utf-8") as f:
            cfg: Dict[str, Any] = json.load(f)
        for ov in args.override:
            deep_update(cfg, parse_override(ov))
        run_pipeline_pair(cfg, args.out)
        return 0
    elif args.cmd == "calibrate":
        # Placeholder for offline calibration; intentionally no-op per decision
        print("Calibration CLI is stubbed (offline-only, disabled by default).")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
