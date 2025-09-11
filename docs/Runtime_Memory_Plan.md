**Runtime/Memory Plan (V1)**

- **Scope:** Orchestrator responsibilities for pipeline scheduling, streams/events, caching/rollback, KV‑cache memory, OOM/fragmentation handling, and CUDA graph capture policy. Aligns with `icd/runtime/orchestrator.py` mock and planned extensions.

**Scheduling & Control**
- **Stages:** BUILD_W → PERMUTE → TRANSFORM → REPERMUTE? → MEASURE → REPORT.
- **Seeding/Determinism:** Fix `rng_seed`, record into `metrics.env`; all randomness seeded from config.
- **Retries:** On transient failures (measure, report), retry once; on deterministic failures (graph invalid), abort with RCA info.

**Caching Policy**
- **Key:** hash of `(graph.meta, cost cfg, solver cfg, adapter metas)`.
- **Artifacts:** cache `perm.json`, `stats.json` keyed by hash; on hit, skip solver (I‑02).
- **Status:** implementation optional/off by default in current repo to keep CI deterministic.
- **Invalidation:** any `delta_layout=true` or cfg change invalidates.

**KV Cache & Memory**
- **Layout:** contiguous last dim; block size influences stride.
- **Pools:** pre‑allocate slab per stream; reuse across runs with same D/block.
- **Fragmentation:** escalate to compaction if slab free space < 10%; otherwise allocate new slab; record in logs.
- **OOM Handling:** catch OOM, reduce batch/sequence or drop optional features (e.g., power sampling) and continue; emit `errors` entry in `metrics`.

**Streams & Events**
- **Streams:** separate streams for compute and measure; avoid contention by scheduling measurement post‑warmup.
- **Events:** timestamps per stage added to `run.log`; overhead target < 1% of runtime.

**CUDA Graph Capture** (policy)
- Capture steady state if model supports static shapes and adapters froze layout (no further re‑perm). Disable capture when `delta_layout=true` to avoid invalidation.

**Rollback Policy**
- Acceptance gate: if `delta_J > -epsilon_J`, mark `rolled_back=true` and persist baseline π for downstream usage.
- Provide `rollback.reason` and pointers to RCA checklist in logs.

**Self‑Review**
- Extends current mock orchestrator by specifying cache keys, memory pools, and capture policy. Compatible with tests expecting artifacts/logs without enforcing GPU features in CI.
