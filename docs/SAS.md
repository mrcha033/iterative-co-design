# SAS — Iterative HW–SW Co-Design: Memory-Aware Layout Re-Optimization

**One-line summary**
Provide the `permute → transform(S/Q/K) → re-permute` loop as a unified library and CLI. The core workflow is to generate the shared-access weight matrix $W$, run the modularity-based re-permutation, instrument with NCU/NVML, and automate the reporting. If any step fails, roll back immediately.

---

## 0) Assumptions & Constraints

* **Environment**: Python 3.10, CUDA 11.x+, with MPS/CPU guards. Nsight Compute (ncu) and NVML power sampling must be available. At least one A100 or H100 GPU.
* **Scope**: Inference-only pathway. Training, distributed scheduling, and cluster orchestration are excluded.
* **Goals (summary)**: Achieve Latency −20%, L2 hit +10%p, and EpT −15% on two representative workloads (SSM-Mamba-3B and Transformer-BERT-base) for the MVP.

---

## 1) Scope

* **In-scope**

  * $W$ construction (from traces or mocks), the re-permutation solver (spectral + local search), S/Q/K transformation adapters, the pipeline runner, instrumentation/reporting, and the TVM/StableHLO bridge **PoC**.
* **Out-of-scope**

  * Model training pipelines, new hardware design, distributed inference, and automation for commercial cluster operations.

---

## 2) Overview

```
┌────────────────────────────────────────────────────────────────┐
│                         icd CLI / Python API                   │
├────────────────────────────────────────────────────────────────┤
│                        runtime/orchestrator                    │
│  (pipeline scheduling, retries, rollback, artifact paths, cache)│
├───────────────┬─────────────────────┬─────────────────┬─────────┤
│ core/graph    │ core/solver         │ adapters/       │ measure │
│ (trace→W)     │ (spectral+local)    │  S | Q | K      │ (ncu,nvml,
│               │                     │ (transform/meta)│ timers, reporter)
├───────────────┴───────────────┬─────┴─────────────────┴─────────┤
│ bridge/ (TVM or StableHLO PoC)│     storage/ (perm/W/log/art)   │
└───────────────────────────────┴─────────────────────────────────┘
```

* **core/graph**: Generate the shared-access weight matrix $W$ from execution traces or mocks.
* **core/solver**: Combine spectral ordering (Fiedler) and local search (2-opt/adj-swap). Compute the cost $C(\pi)$ and modularity $Q$.
* **adapters/**: Three transformation families—S (sparsity: 2:4/HDS and unstructured), Q (PTQ/FP8 stubs), K (KV-cache compression). Update metadata and trigger actions after each transform.
* **runtime/**: Pipeline orchestrator handling control flow, caching, errors/rollback, retries, and seed/clock pinning.
* **measure/**: Wrappers for Nsight Compute, NVML, and wall-clock timers, plus HTML/CSV reporters.
* **bridge/**: Define insertion points and metadata passes on StableHLO/TVM IR as a **PoC**.
* **storage/**: Artifact directory containing `perm.json`, `W.npz`, `ncu.rep`, `power.csv`, and `report.html`.

---

## 3) Data & Artifacts

* **W(n×n)**: float32 symmetric sparse matrix (CSR/COO). Schema: `{"shape": n, "format": "csr", "nnz": m, "source": "trace|mock", "seed": int}`
* **π (Permutation)**: int32 array of length n. Version tag: `hash(model+task+S/Q/K+seed)`.
* **Transform Meta**: `{ sparsity: {type: "2:4|unstructured", rate}, quant: {dtype: "int8|fp8", method}, kv: {block: int, drop: float} }`
* **Metrics Record**: `{lat_ms, toks_per_s, l2_hit_pct, ept_j_per_tok, hw, driver, seed, clock_cap, repeats}`
* **Reports**: `report.html|.csv` (pre/post comparison with tables/figures), `ncu.json`, `power.csv`.

---

## 4) Control Flow (Pipeline)

### Linear vs Iterative Sequence

1. **BuildW**: Generate and normalize $W$ from an execution trace or mock configuration.
2. **Permute**: `π₀ ← solver.fit(W)`; apply the layout.
3. **Transform**: Apply one or more of S/Q/K (metadata updates included).
4. **Re-permute**: `π₁ ← solver.fit(W’)`; reflect state changes.
5. **Run/Measure**: Measure `lat/l2/EpT` on the same input batch.
6. **Report/Cache**: Produce pre/post comparisons and cache `π₁`. Roll back to `π₀` if anything fails.

> The **baseline** omits Step 4 (Re-permute).

### State Machine

* `READY → W_BUILT → PERMUTED → TRANSFORMED → REPERMUTED → MEASURED → REPORTED`
* On failure: `→ ROLLBACK(π_prev) → REPORTED`.

---

## 5) Responsibilities & Contracts

### 5.1 core/graph

* **Input**: Execution trace events or mock configuration (seed, blocks, noise).
* **Output**: $W$ (CSR/COO).
* **Contract**: Deterministic $W$ for the same input. Enforce `nnz/shape` limits and forbid NaNs.

### 5.2 core/solver

* **Input**: $W$, soft time budget, number of local-search steps, optional k (number of blocks).
* **Output**: Permutation `π`, secondary metrics `C(π), Q(π)`, and cluster/modularity summaries `Q_cluster/Q_final`.
* **Contract**: Finish optimization within the time budget. If no improvement is found, return the initial solution with a flag. Accept Louvain/spectral cluster seeds, preserve their ordering, and track modularity progress.

### 5.3 adapters/S|Q|K

* **Input**: Tensors and layout metadata.
* **Output**: Updated tensors/weights plus **transformation metadata**.
* **Contract**: Report the expected loss/precision impact range and return the trigger condition for re-permutation. On failure, fall back to a no-op.

### 5.4 runtime/orchestrator

* **Responsibilities**: Schedule pipeline stages, manage retries/backoff, handle rollback on failure, manage cache hit/miss, enforce seed/clock pinning, and maintain artifact paths. After each transform, collect activation-based correlations → run Louvain/spectral clustering → provide seeds to the solver.
* **Contract**: Emit logs/events for all steps with <1% overhead. Persist `correlation/` artifacts (`.pt/.json`) and cluster metadata.

### 5.5 measure/

* `benchmark_inference`: Built-in GPU benchmark (warmup/loops, CUDA events, NVTX tagging) aggregating Latency/Throughput/EpT.
* `ncu`: Collect L2 hit data and select profiler key sets.
* `nvml`: Sample power (at a fixed cadence) and compute EpT.
* **Contract**: Respect warmup exclusions, repeat count N, and fixed clock/power-cap options. Write Latency/L2/EpT results to `metrics.json` and automatically evaluate the acceptance gate (BRD goals).

### 5.6 bridge/

* Define **insertion points** in the StableHLO or TVM pass pipeline (e.g., layout-tag attach → pass → lower).
* **Contract**: During the PoC, only round-trip metadata and perform minimal transformations.

---

## 6) Interfaces (ICD Summary & Signatures)

```python
# graph
W = build_w(source: Literal["trace","mock"], **cfg) -> csr_matrix

# solver
pi, stats = fit_permutation(W, time_budget_s=300, refine_steps=2_000)  # stats: {C, Q}

# adapters
out, meta = apply_sparsity(tensor, type="2:4", rate=0.5)
out, meta = apply_quant(tensor, dtype="int8", method="ptq-minmax")
out, meta = apply_kvcache(cache, block=128, drop=0.1)

# orchestrator (within the CLI)
run(config: Dict) -> RunArtifacts  # artifact paths, metrics summary
```

(Detailed types/exceptions/error codes are finalized in the ICD document. They are shortened here.)

---

## 7) Performance & Resource Budgets

* **Re-permute overhead**: ≤ 5 minutes for D≈2.5k (offline). For large graphs use sampling, partial eigenvectors, or block parallelism.
* **Memory ceiling**: $W$ CSR `nnz` ≤ 0.05·D² for mocks; trace-based runs are limited by the trace itself.
* **Instrumentation overhead**: < 5% including ncu/NVML (tune profiler sampling ratio as needed).
* **Reporting time**: Generate HTML/CSV in under 30 seconds.

---

## 8) Concurrency Model

* Pipeline stages run **synchronously** (prioritize determinism).
* Internal parallelism: Allow data-parallel execution for spectral (partial eigenvectors) and local search (thread-per-sample).
* Instrumentation runs in a **separate process** (ncu CLI) for isolation. Use file locks to prevent artifact collisions.

---

## 9) Observability

* **Event schema**: `stage, t_start, t_end, ok, meta{…}`
* **Counters**: `lat_ms, toks_s, l2_hit_pct, ept, Q, C, nnz(W)`
* **Sampling**: Distinguish sampled vs. full ncu profiles.
* **Overhead guard**: Measure and report observability overhead itself.

---

## 10) Error Model & Rollback

* **Graph failure**: If the trace is corrupted or empty, substitute a mock or abort.
* **Solver timeout**: Return the initial solution (flagged) and optionally retry.
* **Adapter failure**: Emit a warning, perform a no-op, and continue the pipeline.
* **Instrumentation failure**: Downgrade to alternate metrics (wall-clock timers).
* **Quality regression**: Automatically roll back to the previous `π` when Latency/L2/EpT worsen.

---

## 11) Security, Licensing, Compliance

* Automatically generate an **SBOM** and run license scans in CI.
* Honor dataset/model cards and forbid storing sensitive information in logs.
* Store profiler outputs locally by default; uploading is opt-in.

---

## 12) Deployment

* Distribution: PyPI package plus the `icd` CLI (`pip install icd-co`).
* Provide a **Dockerfile** (driver matching excluded) and include a conda environment export.
* Device guard:

  ```python
  import torch
  device = ("mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu")
  ```

  (Applies only to the PyTorch pathway. TVM/StableHLO pathways have separate guards.)

---

## 13) Testing Architecture

* **Unit**: Solver monotonicity (stronger blocks → lower C, higher Q), graph determinism, adapter no-op safety.
* **Integration**: Simultaneously emit pre/post metrics for linear vs. iterative flows.
* **E2E**: Automatically reproduce the two representative workloads.
* **Regression**: Gate microbenchmarks (latency/tokens per second) with tolerances.
* **Determinism**: Pin seeds/clocks, run ≥30 repeats, and check the 95% CI in CI jobs.

---

## 14) Key Design Decisions

* **Modularity objective**: Reflects cache-line (block) realities better than TSP-style pairwise optimizations, giving a more faithful objective.
* **Spectral + local search**: Interpretable, stable initial solutions with lightweight refinements. Learned schedulers remain a phase-two option.
* **IR bridge PoC**: Initially focus on metadata round-tripping and minimal passes; full integration is part of the phase-two roadmap.

---

## 15) Risks & Mitigations

* **Instrumentation variability**: Driver/thermal/clock drift → enforce the SOP (fixed clocks, warmup, repeats) and log an environment hash.
* **Scale**: O(D³) when D grows → rely on Lanczos/partial eigenvectors and block-wise independent optimization.
* **Generalization**: Two-workload bias → run ablation sweeps (sparsity/precision/sequence length) and optionally include one additional model.
* **Integration cost**: Framework diversity → prioritize the CLI, keep IR integration opt-in.

---

## 16) Acceptance Criteria

* Achieve **Latency −20% / L2 +10%p / EpT −15%** simultaneously on the two representative workloads with quality degradation ≤ 0.1%p.
* On failure, automatically produce rollback artifacts, reports, and logs that support root-cause analysis.
* Provide a reproducibility package that external reviewers can run successfully within **24 hours**.

---

## 17) Execution Checklist (Owner View)

* [ ] Finalize the SOP (fixed clocks, warmup, repeat count)
* [ ] Finalize the ICD/pass insertion points (bridge PoC)
* [ ] Approve the $W$ schema and compression formats
* [ ] Select solver time budgets and scaling strategy
* [ ] Validate observability schema and overhead measurements
* [ ] Apply regression gates (thresholds) in CI

---

## 18) Config Keys (Excerpt)

```yaml
pipeline:
  mode: iterative   # linear|iterative
  repeats: 1000
  fixed_clock: true
  warmup_iter: 50
graph:
  source: trace     # trace|mock
  mock:
    blocks: 4
    noise: 0.02
    seed: 42
solver:
  time_budget_s: 300
  refine_steps: 2000
  k_blocks: 4
transform:
  sparsity: {type: "2:4", rate: 0.5}
  quant:    {dtype: "int8", method: "ptq-minmax"}
  kv:       {block: 128, drop: 0.10}
measure:
  ncu_metrics: ["l2_tex__t_sector_hit_rate.pct"]
  power_sample_hz: 10
report:
  out_dir: "runs/exp001"
```

---

### Appendix — Simplified Dataflow (DFD)

`trace/mock → W(CSR) → π₀ → S/Q/K → W' → π₁ → run → {lat, l2, ept} → report.html`

---
