from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from icd.core.graph import build_w, save_w_npz
from icd.core.solver import fit_permutation
from icd.core.cost import CostConfig, eval_cost
from icd.measure.report import write_csv_report, write_html_report
from icd.measure.ncu_wrapper import parse_l2_hit_from_section_json
from icd.runtime.compare import decide as compare_decide


@dataclass
class RunArtifacts:
    out_dir: str
    config_lock_path: str
    run_log_path: str
    metrics_path: str


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(config: Dict[str, Any]) -> RunArtifacts:
    """Minimal orchestrator: builds W, solves, optionally re-permutes, and writes artifacts.

    This is a MOCK runner to validate pipeline wiring and artifacts.
    """
    cfg = dict(config)
    out_dir = cfg.get("report", {}).get("out_dir", cfg.get("out", "runs/mock"))
    _ensure_dir(out_dir)

    # Snapshot config
    lock_path = os.path.join(out_dir, "config.lock.json")
    _write_json(lock_path, cfg)

    # Run log
    log_path = os.path.join(out_dir, "run.log")
    events = []

    def emit(stage: str, ok: bool, meta: Dict[str, Any] | None = None):
        ev = {"stage": stage, "t": time.time(), "ok": ok, "meta": (meta or {})}
        events.append(ev)

    # Build W (mock-only)
    gcfg = cfg.get("graph", {})
    W = build_w(source=gcfg.get("source", "mock"), **gcfg.get("mock", gcfg))
    emit("W_BUILT", True, {"nnz": W.nnz(), "shape": W.shape, "source": W.meta.get("source")})
    w_path = os.path.join(out_dir, "W.csr.npz")
    save_w_npz(w_path, W)
    # Write meta/ops json for PyTorch source
    if W.meta.get("source") == "pytorch":
        import hashlib, json as _json

        meta_path = os.path.join(out_dir, "w.meta.json")
        ops_path = os.path.join(out_dir, "w.ops.json")
        pt_meta = W.meta.get("pytorch", {})
        used_ops = pt_meta.get("used_ops", [])
        trace_hash = hashlib.sha256(("|".join(used_ops) + f"|D={W.shape[0]}|hops={pt_meta.get('hops')}").encode("utf-8")).hexdigest()
        meta_doc = {
            "D": W.shape[0],
            "nnz": W.nnz(),
            "normalize": gcfg.get("normalize", "sym"),
            "band_kernel": {
                "hops": gcfg.get("pytorch", {}).get("hops", 1),
                "reuse_decay": gcfg.get("pytorch", {}).get("reuse_decay", 0.7),
            },
            "op_weights": {
                "linear": 1.0,
                "matmul": 1.0,
                "addmm": 1.0,
                "bmm": 0.8,
                "sdpa": 1.2,
                "add": 0.25,
                "layout": 0.1,
            },
            "seed": scfg.get("rng_seed", 0),
            "trace_source": "pytorch",
            "trace_hash": pt_meta.get("trace_hash") or trace_hash,
        }
        # bubble up attention block if present
        if pt_meta.get("attention"):
            meta_doc["attention"] = pt_meta.get("attention")
        ops_doc = {
            "used_ops": [f"aten::{k}" for k in used_ops],
            "skipped_ops_count": pt_meta.get("skipped_ops_count", 0),
            "hops": pt_meta.get("hops", 1),
            "roles_present": used_ops,
            "notes": "last-dim feature heuristic; attention-aware mapping enabled when sectioning",
        }
        _write_json(meta_path, meta_doc)
        _write_json(ops_path, ops_doc)

    # Baseline permute
    scfg = cfg.get("solver", {})
    ccfg = CostConfig(
        alpha=cfg.get("cost", {}).get("alpha", 1.0),
        beta=cfg.get("cost", {}).get("beta", 0.2),
        gamma_stability=cfg.get("cost", {}).get("gamma_stability", 0.1),
        mu=cfg.get("cost", {}).get("mu", 0.5),
        g=cfg.get("cost", {}).get("g", 64),
        lambda_=cfg.get("cost", {}).get("lambda", 3),
        tau=cfg.get("cost", {}).get("tau", 0.25),
        modularity_gamma=cfg.get("cost", {}).get("modularity_gamma", 1.2),
        blocks_k=cfg.get("solver", {}).get("k_blocks", 4),
        vec_width=cfg.get("cost", {}).get("vec_width", 16),
        hysteresis=cfg.get("cost", {}).get("hysteresis", 2),
    )

    pi0, stats0 = fit_permutation(
        W,
        time_budget_s=float(scfg.get("time_budget_s", 1.0)),
        refine_steps=int(scfg.get("refine_steps", 1000)),
        cfg=ccfg,
        seed=int(scfg.get("rng_seed", 0)),
    )
    emit("PERMUTED", True, {"J": stats0.get("J"), "C": stats0.get("C"), "Q": stats0.get("Q")})

    # Transform (mock: no-op + meta)
    transform_meta = {"delta_layout": False}
    emit("TRANSFORMED", True, transform_meta)

    # Re-permute if iterative
    mode = cfg.get("pipeline", {}).get("mode", "iterative")
    pi1 = pi0
    stats1 = stats0
    if mode == "iterative":
        pi1, stats1 = fit_permutation(
            W,
            time_budget_s=float(scfg.get("time_budget_s", 1.0)),
            refine_steps=int(scfg.get("refine_steps", 500)),
            cfg=ccfg,
            seed=int(scfg.get("rng_seed", 0)),
        )
        improved = stats1.get("J", 0.0) < stats0.get("J", 0.0)
        emit("REPERMUTED", True, {"improved": improved, "J1": stats1.get("J"), "J0": stats0.get("J")})

    # Acceptance/rollback (ΔJ gate; HW gates optional)
    delta_J = stats1.get("J", 0.0) - stats0.get("J", 0.0)
    epsJ = float(cfg.get("rollback", {}).get("epsilon_J", 0.01))
    accepted = (delta_J <= -epsJ)
    rolled_back = not accepted and (mode == "iterative")

    # Mock measurement: infer proxy metrics from ΔJ (negative is improvement)
    # naive linear proxy for smoke
    lat0 = 100.0
    lat1 = lat0 * (1.0 + min(0.0, delta_J))  # if J decreased by 10%, improve ~10%
    l2_0 = 0.80
    l2_1 = l2_0 * (1.0 - min(0.0, delta_J))
    ept0 = 1.0
    ept1 = ept0 * (1.0 + min(0.0, delta_J))

    # SOP wiring: compute latency stats (CI-safe synthetic loop) and optional L2/EpT
    repeats = int(cfg.get("pipeline", {}).get("repeats", 100))
    warmup = int(cfg.get("pipeline", {}).get("warmup_iter", 20))
    import math
    import statistics

    # synthetic latency samples from proxy
    samples = [lat1 if mode == "iterative" else lat0 for _ in range(repeats)]
    mean = float(sum(samples) / max(1, len(samples)))
    p50 = float(sorted(samples)[len(samples) // 2])
    p95 = float(sorted(samples)[int(math.ceil(0.95 * len(samples))) - 1])
    stdev = float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0
    ci95 = float(1.96 * (stdev / math.sqrt(max(1, len(samples)))))

    # Optional L2/EpT
    measure_cfg = cfg.get("measure", {})
    l2_hit = None
    errors: list[dict] = []
    if measure_cfg.get("ncu_enable", False):
        try:
            l2_res = parse_l2_hit_from_section_json(os.path.join(out_dir, "ncu.json"))
            l2_hit = l2_res.get("l2_hit_pct")
        except Exception as e:
            errors.append({"stage": "ncu", "kind": "error", "detail": str(e)})
            l2_hit = None
    power_enable = measure_cfg.get("power_enable", False)
    ept = None
    if power_enable:
        try:
            # Placeholder: require external epilogger to compute energy; here set None
            ept = None
        except Exception as e:
            errors.append({"stage": "power", "kind": "error", "detail": str(e)})

    metrics = {
        "latency_ms": {"mean": mean, "p50": p50, "p95": p95, "ci95": ci95},
        "l2_hit_pct": l2_hit,
        "ept_j_per_tok": ept,
        "mode": mode,
        "C": stats1.get("C"),
        "Q": stats1.get("Q"),
        "J": stats1.get("J"),
        "env": {"seed": scfg.get("rng_seed", 0), "fixed_clock": cfg.get("pipeline", {}).get("fixed_clock", True)},
        "acceptance": {
            "epsilon_J": epsJ,
            "delta_J": delta_J,
            "accepted": accepted,
            "rolled_back": rolled_back,
            "incomplete": True,  # single-run; baseline not present for deltas
            "note": "HW gates evaluated only when both before/after exist",
        },
        "quality": {
            "task": None,
            "metric": None,
            "before": None,
            "after": None,
            "delta": None,
        },
        "errors": errors,
    }
    # Optional quality hook (CI-safe): if eval.enable is true, include quality field (None by default)
    if cfg.get("eval", {}).get("enable", False):
        metrics["quality"] = None
    metrics_path = os.path.join(out_dir, "metrics.json")
    _write_json(metrics_path, metrics)
    emit("MEASURED", True, {})

    # Simple CSV report
    write_csv_report(out_dir, metrics)
    write_html_report(out_dir, metrics)
    emit("REPORTED", True, {})

    # Persist log
    with open(log_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    return RunArtifacts(out_dir=out_dir, config_lock_path=lock_path, run_log_path=log_path, metrics_path=metrics_path)


def run_pair(config: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    cfg_base = dict(config)
    cfg_base.setdefault("report", {})["out_dir"] = os.path.join(out_dir, "linear")
    cfg_base.setdefault("pipeline", {})["mode"] = "linear"
    base_art = run(cfg_base)

    cfg_trial = dict(config)
    cfg_trial.setdefault("report", {})["out_dir"] = os.path.join(out_dir, "iter")
    cfg_trial.setdefault("pipeline", {})["mode"] = "iterative"
    trial_art = run(cfg_trial)

    base_metrics = _read_json(base_art.metrics_path)
    trial_metrics = _read_json(trial_art.metrics_path)
    fixed_clock = bool(trial_metrics.get("env", {}).get("fixed_clock", True))
    epsJ = float(trial_metrics.get("acceptance", {}).get("epsilon_J", 0.01))
    verdict = compare_decide(base_metrics, trial_metrics, fixed_clock=fixed_clock, eps_J=epsJ)
    _write_json(os.path.join(out_dir, "compare.json"), verdict)
    # update acceptance in trial
    trial_metrics.setdefault("acceptance", {}).update(verdict)
    _write_json(trial_art.metrics_path, trial_metrics)
    return verdict


__all__ = ["RunArtifacts", "run", "run_pair"]
