# Observability Spec — Events, Metrics, Overhead Guard

One-line: Fix event/metric keyspace, sampling limits, and overhead <1% requirement.

## Events
- Schema: `{stage, t_start|t, t_end?, ok, meta{...}}`
- Stages: READY, W_BUILT, PERMUTED, TRANSFORMED, REPERMUTED, MEASURED, REPORTED, ROLLBACK.

## Metrics
- `metrics.json`: `{latency_ms{mean,p50,p95}, toks_per_s, l2_hit_pct, ept_j_per_tok, C,Q,J, hw, driver, seed, repeats}`
- NCU: section-based JSON; NVML: `power.csv` (t,power_w @10Hz).

## Overhead Guard
- Profiling overhead <5%; observability runtime overhead <1%.
- Split runners if needed.

## Privacy/Security
- No PII/paths/credentials in logs.

