**Observability Spec (V1)**

- **Scope:** Define events, metrics, sampling policies, overhead budgets, and privacy rules. Matches orchestrator outputs and planned measurement integration.

**Events (run.log)**
- **Format:** one JSON per line: `{stage, t, ok, meta}`.
- **Stages:** `W_BUILT`, `PERMUTED`, `TRANSFORMED`, `REPERMUTED`, `MEASURED`, `REPORTED`.
- **Overhead:** run logging ≤ 1% of wall time; avoid chatty per‑iteration logs.

**Metrics (metrics.json)**
- **latency_ms:** `{mean, p50, p95, ci95}` — sampled post‑warmup.
- **l2_hit_pct:** `float|null` — from Nsight Compute; null when disabled or unavailable.
- **ept_j_per_tok:** `float|null` — energy per token; null when power disabled.
- **mode:** `linear|iterative`.
- **C,Q,J:** scalars from cost evaluation.
- **env:** `{seed, fixed_clock}`.
- **acceptance:** `{epsilon_J, delta_J, accepted, rolled_back, incomplete, note}`.
- **quality:** `null|{task, metric, before, after, delta}` when eval enabled.
- **errors:** `[ {stage, kind, detail} ]` for non‑fatal issues.

**Sampling Rules**
- **Warmup:** ≥50 iterations; exclude from metrics.
- **Repeats:** ≥1000 for latency CI; smaller for smoke tests acceptable with documented CI95.
- **NVML:** 10 Hz default when enabled.
- **Nsight:** MemoryWorkloadAnalysis + MemoryChart sections for L2 metrics; export JSON and parse offline.

**Overhead Budgets**
- **Measurement:** < 5% slowdown when profiling enabled.
- **Logging:** < 1%.
- **Report Generation:** negligible (< 0.5s typical), generate HTML/CSV once per run.

**Privacy & Compliance**
- Never record input payloads, model weights, or personally identifiable information.
- Store only derived metrics and coarse configuration.
- Redact file paths and environment variables unless necessary for RCA; do not include secrets.

**Self‑Review**
- Matches fields emitted in `icd/runtime/orchestrator.py` and `icd/measure/report.py`. Nsight/NVML integration left optional with nulls documented.

