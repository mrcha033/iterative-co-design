from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from icd.runtime.orchestrator import run as run_pipeline
from icd.runtime.orchestrator import run_pair as run_pipeline_pair
from icd.errors import ConfigError


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
    runp.add_argument("--dry-run", action="store_true", help="Validate config and exit")
    runp.add_argument("--print-schema", action="store_true", help="Print minimal input schema and exit")
    runp.add_argument("--no-measure", action="store_true", help="Skip measurement/report stages (solver only)")
    runp.add_argument("--reuse-perm", help="Reuse an existing permutation from JSON file (perm_before.json format)")
    pairp = sub.add_parser("pair", help="Run baseline+trial and compare")
    pairp.add_argument("-c", "--config", required=True, help="JSON config path")
    pairp.add_argument("--override", action="append", default=[], help="Override key=val (dot.path)")
    pairp.add_argument("--out", required=True, help="Output directory for pair run root")
    pairp.add_argument("--dry-run", action="store_true", help="Validate config and exit")
    pairp.add_argument("--print-schema", action="store_true", help="Print minimal input schema and exit")
    args = ap.parse_args(argv)

    def _print_schema() -> None:
        import json as _json, os as _os
        # project root (../.. from icd/cli/main.py)
        _root = _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__)))
        schema_path = _os.path.join(_root, "docs", "schema", "run_config.schema.json")
        if _os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                print(f.read())
        else:
            # Fallback minimal skeleton
            schema = {
                "pipeline": {"mode": "linear|iterative", "repeats": "int", "warmup_iter": "int", "fixed_clock": "bool"},
                "graph": {"source": "mock|pytorch|trace", "mock": {"d": "int", "blocks": "int", "noise": "float", "seed": "int", "normalize": "sym|row|none"}},
                "solver": {"time_budget_s": "float", "refine_steps": "int", "k_blocks": "int", "rng_seed": "int"},
                "report": {"out_dir": "path"},
            }
            print(_json.dumps(schema, indent=2, ensure_ascii=False))

    def _validate_config(cfg: Dict[str, Any]) -> list[str]:
        errs: list[str] = []
        if not isinstance(cfg.get("pipeline"), dict):
            errs.append("Missing pipeline")
        if not isinstance(cfg.get("graph"), dict):
            errs.append("Missing graph")
        if not isinstance(cfg.get("solver"), dict):
            errs.append("Missing solver")
        if not isinstance(cfg.get("report"), dict):
            errs.append("Missing report")
        else:
            if not cfg.get("report", {}).get("out_dir"):
                errs.append("report.out_dir required")
        mode = cfg.get("pipeline", {}).get("mode")
        if mode not in ("linear", "iterative"):
            errs.append("pipeline.mode must be 'linear' or 'iterative'")
        repeats = cfg.get("pipeline", {}).get("repeats")
        if repeats is not None and (not isinstance(repeats, int) or repeats < 1):
            errs.append("pipeline.repeats must be positive integer if set")
        warmup = cfg.get("pipeline", {}).get("warmup_iter")
        if warmup is not None and (not isinstance(warmup, int) or warmup < 0):
            errs.append("pipeline.warmup_iter must be non-negative integer if set")
        fixed_clock = cfg.get("pipeline", {}).get("fixed_clock")
        if fixed_clock is not None and not isinstance(fixed_clock, bool):
            errs.append("pipeline.fixed_clock must be boolean if set")
        # Graph
        src = cfg.get("graph", {}).get("source")
        if src not in ("mock", "pytorch", "trace"):
            errs.append("graph.source must be one of: mock|pytorch|trace")
        if src == "mock":
            m = cfg.get("graph", {}).get("mock", {})
            for k in ("d", "blocks", "noise", "seed"):
                if k not in m:
                    # defaultable but warn
                    continue
            if "normalize" in cfg.get("graph", {}) and cfg.get("graph", {}).get("normalize") not in ("sym", "row", "none"):
                errs.append("graph.normalize must be one of: sym|row|none")
        if src == "trace":
            tr = cfg.get("graph", {}).get("trace")
            if tr is None or (isinstance(tr, str) and not tr):
                errs.append("graph.trace is required for source=trace (iterable or path)")
            if "normalize" in cfg.get("graph", {}) and cfg.get("graph", {}).get("normalize") not in ("sym", "row", "none"):
                errs.append("graph.normalize must be one of: sym|row|none")
        # Solver
        tb = cfg.get("solver", {}).get("time_budget_s")
        if tb is not None and not isinstance(tb, (int, float)):
            errs.append("solver.time_budget_s must be number if set")
        rs = cfg.get("solver", {}).get("refine_steps")
        if rs is not None and (not isinstance(rs, int) or rs < 0):
            errs.append("solver.refine_steps must be non-negative integer if set")
        # Measure
        meas = cfg.get("measure", {})
        if not isinstance(meas, dict):
            errs.append("measure must be object if set")
        else:
            for k in ("ncu_enable", "power_enable"):
                if k in meas and not isinstance(meas.get(k), bool):
                    errs.append(f"measure.{k} must be boolean if set")
            if "power_sample_hz" in meas and (not isinstance(meas.get("power_sample_hz"), int) or meas.get("power_sample_hz") <= 0):
                errs.append("measure.power_sample_hz must be positive integer if set")
        # Cache
        cache = cfg.get("cache", {})
        if cache:
            if not isinstance(cache, dict):
                errs.append("cache must be object")
            else:
                if cache.get("enable") and not cache.get("cache_dir"):
                    errs.append("cache.cache_dir required when cache.enable=true")
        # Report formats
        fmts = cfg.get("report", {}).get("formats")
        if fmts is not None:
            if not isinstance(fmts, list) or any(x not in ("html", "csv") for x in fmts):
                errs.append("report.formats must be a list containing only 'html' and/or 'csv'")
        return errs

    try:
        if args.cmd == "run":
            with open(args.config, "r", encoding="utf-8") as f:
                cfg: Dict[str, Any] = json.load(f)
            for ov in args.override:
                deep_update(cfg, parse_override(ov))
            cfg.setdefault("report", {})["out_dir"] = args.out
            if args.no_measure:
                cfg.setdefault("pipeline", {})["no_measure"] = True
            if args.reuse_perm:
                cfg.setdefault("pipeline", {})["reuse_perm"] = args.reuse_perm
            if args.print_schema:
                _print_schema()
                return 0
            issues = _validate_config(cfg)
            if args.dry_run:
                if issues:
                    print("Config issues:\n- " + "\n- ".join(issues))
                    return 2
                print("Config OK")
                return 0
            if issues:
                raise ConfigError("; ".join(issues))
            run_pipeline(cfg)
            return 0
        elif args.cmd == "pair":
            with open(args.config, "r", encoding="utf-8") as f:
                cfg: Dict[str, Any] = json.load(f)
            for ov in args.override:
                deep_update(cfg, parse_override(ov))
            if args.print_schema:
                _print_schema()
                return 0
            issues = _validate_config(cfg)
            if args.dry_run:
                if issues:
                    print("Config issues:\n- " + "\n- ".join(issues))
                    return 2
                print("Config OK")
                return 0
            if issues:
                raise ConfigError("; ".join(issues))
            run_pipeline_pair(cfg, args.out)
            return 0
        elif args.cmd == "calibrate":
            # Placeholder for offline calibration; intentionally no-op per decision
            print("Calibration CLI is stubbed (offline-only, disabled by default).")
            return 0
    except ConfigError as e:
        print(f"ConfigError: {e}")
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
