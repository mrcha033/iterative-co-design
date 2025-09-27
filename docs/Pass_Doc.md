# Pass Design Doc — Iterative HW–SW Co-Design Layout Re-Optimization

**One-line summary**
Define the **analysis and transform passes** plus IR metadata contracts, tests, and performance guards required to insert the `permute → transform(S/Q/K) → re-permute` loop into the StableHLO/TVM pipelines.

---

## 0) Assumptions & Constraints

* **Scope**: Inference pathway. Excludes training/distributed scheduling.
* **IR**: Prioritize StableHLO (MLIR-based); TVM Unity provides an equivalent **bridge**.
* **Guarantees**: Semantic equivalence, precision preservation, and schedule stability (no accuracy loss due to the transform).
* **Inputs**: `icd.transform_meta` (S/Q/K changes), `icd.layout_perm` (π), `icd.coaccess_block` (block hint).

---

## 1) Goals

1. Perform **layout re-optimization immediately after state transforms** in a safe, deterministic way at the IR level.
2. Rearrange dimensions/strides to **improve cache locality** for downstream kernels.
3. Keep IR metadata **round-trippable** through kernels and runtime.

---

## 2) Pass Pipeline (DAG)

```
Frontend
  └─ attach-icd-metadata (A0)
      ├─ build-coaccess-graph (A1, Analysis)
      ├─ layout-cost-model (A2, Analysis)
      ├─ decide-permutation (T1, Transform - pre)
      ├─ apply-transform S|Q|K (EXT, external)
      ├─ rebuild-coaccess-graph (A1′)
      ├─ redecide-permutation (T2, Transform - post)
      ├─ legalize-layout (T3, Transform)
      └─ verify-and-annotate (V1, Verification)
Lowering → Codegen → Runtime
```

* **A0**: Attach metadata (no-op if absent).
* **A1/A1′**: Build the shared-access graph $W$ (trace/heuristic-based).
* **A2**: Prepare cost model and modularity evaluation.
* **T1/T2**: Decide (and re-decide) permutations.
* **T3**: Legalize shapes/strides (adjust reshape/transposed ops).
* **V1**: Verify equivalence, alignment, and aliasing (custom analysis).

---

## 3) Pass Specifications

### A0. `attach-icd-metadata`

* **Input**: Module/function, user config/ICD data.
* **Output**: Annotated tensors with:

  * `icd.layout_perm : i32[D]`
  * `icd.layout_tag : "icd/v1"`
  * `icd.transform_meta : json` (S/Q/K)
  * `icd.coaccess_block : i32` (optional)
* **Legality**: Preserve type/shape, add attributes only.
* **Failure**: Replace duplicate tags with the newest version (warn).

### A1. `build-coaccess-graph` (Analysis)

* **Role**: Compute the weighted shared-access graph $W$ from operator/tensor usage.
* **Method**:

  * Base case: Estimate producer–consumer paths and reuse distance/fusion scope within IR.
  * Optional: Merge external trace files or sampled counters.
* **Output**: Module attribute `icd.W` (CSR pointer) or sidecar artifact.
* **Complexity**: O(#edges + #uses).
* **Fallback**: If empty/overly sparse, fall back to `W_min` (diagonal).

### A2. `layout-cost-model` (Analysis)

* **Role**: Provide functions for the cost $C(\pi)=\sum_{i<j} W_{ij}|pos(i)-pos(j)|$ and modularity $Q$.
* **Output**: Function handles/tables (optionally cached).
* **Validation**: Ensure monotonic behavior where stronger block structure yields $C↓, Q↑$.

### T1. `decide-permutation` (Transform – pre)

* **Trigger**: Initial permutation or missing `icd.layout_perm`.
* **Algorithm**: Fiedler ordering → local refinement (2-opt/adj-swap) under a time budget.
* **Output**: `icd.layout_perm = π₀`.
* **Guarantee**: Shapes unchanged; only stride/memory-order tags change.
* **Failure**: On timeout, retain the initial layout (flag the event).

### EXT. `apply S|Q|K` (external transform)

* **Role**: Apply sparsity/quantization/KV compression in external stages.
* **Contract**: If `icd.transform_meta.delta_layout=true`, invoke **T2**.

### T2. `redecide-permutation` (Transform – post)

* **Trigger**: `icd.transform_meta.delta_layout=true`.
* **Role**: Re-permute using the updated $W′$ to produce `π₁`.
* **Stop condition**: If `C(π₁) < C(π₀) * (1-ε)` fails, roll back to `π₀`.
* **Output**: Emit improvement statistics (ΔC, ΔQ) and events.

### T3. `legalize-layout`

* **Role**: Canonicalize after layout changes.

  * Collapse `transpose/reshape` chains, align broadcast/pad semantics.
  * Apply vectorization-friendly alignment (align=16/32/64) metadata.
  * Forbid stride conflicts/aliasing.
* **Validation**: Keep shape/dtype identical, preserve precision, produce zero alignment violations.
* **Failure**: If legalization is impossible, roll back to the previous layout.

### V1. `verify-and-annotate`

* **Role**: Check invariants and annotate metrics after the pass.

  * MLIR verifier + custom checks: aliasing, layout-tag consistency, alignment/padding, permutation validity.
  * Attach `icd.metrics.{Q,C,π_hash}`.

---

## 4) IR Examples (Summary)

### 4.1 StableHLO (conceptual)

**Before**

```mlir
%k = "stablehlo.dot"(%q, %v) : (tensor<[B,S,H,D]xf16>, tensor<[S,H,D]xf16>) -> tensor<[B,S,H]xf16>
%o = "stablehlo.add"(%k, %bias) : ...
```

**After (metadata updates, transpose inserted only if needed)**

```mlir
%q {icd.layout_perm = dense<[0,2,1,3]> : tensor<4xi32>} // π: (B,H,S,D)
%v {icd.layout_perm = dense<[1,2,0]>  : tensor<3xi32>}  // π: (H,D,S)
%k = "stablehlo.dot"(%q, %v) : ...
"icd.layout_annot"() {Q=0.47, C=9.0e6} : () -> ()
```

### 4.2 TVM Unity (sketch)

**Before**

```python
with Dataflow():
  q = relax.call_tir("matmul", (A,B), out_sinfo=...)
  o = relax.call_tir("add", (q,bias), ...)
```

**After**

```python
q = annotate_layout(q, perm=[0,2,1,3], tag="icd/v1")
v = annotate_layout(v, perm=[1,2,0], tag="icd/v1")
q, v = legalize_layout(q, v)  # insert transpose as needed before fusion
```

---

## 5) Safety & Legality Invariants

* **Shape/op equivalence**: Identical outputs (relative numerical error ≤ 1e-6).
* **Precision**: Preserve dtype and quantization scales (keep quant metadata intact).
* **Alignment/stride**: Guarantee vectorizable alignment (≥16 bytes).
* **No aliasing**: Track view/take/transpose chains to prevent alias conflicts.
* **Determinism**: Same `seed/time_budget` ⇒ same π (±ε).

---

## 6) Configuration (Flags)

```yaml
passes:
  build_coaccess:
    source: "trace|static"
    fuse_scope: "block|function"
  decide_permutation:
    time_budget_s: 300
    refine_steps: 2000
    epsilon: 0.05
  legalize_layout:
    align: 32
    fuse_transpose: true
  verify:
    strict: true
```

---

## 7) Complexity & Performance

* **A1/A1′**: O(#uses), approximately linear.
* **T1/T2**: Spectral (partial eigenvectors) + local search under a **hard time cap**.
* **T3**: Graph rewriting in linear to near-linear time.
* **Overhead guard**: `time_budget_s` defaults to 300 s; if `improved` is false, roll back automatically.

---

## 8) Failure & Fallback Scenarios

* Excessively sparse/unstable `W` → use a uniform permutation (warn) and exit the pass.
* Legalization failure → roll back to the previous π and log the root cause.
* Transform metadata mismatch → skip re-permutation.
* ncu/NVML unavailable (external measurement) → continue passes but disable measurement hooks.

---

## 9) Test Plan (Required)

### 9.1 FileCheck-style (StableHLO)

* **Attach/propagate**: Confirm `icd.layout_perm` annotations flow producer → consumer.
* **Legalize**: Ensure `transpose+reshape` chains collapse.

### 9.2 Golden IR Snapshots

* Compare dense vs. iterative flows; verify identical outputs and valid permutations.

### 9.3 Numerical Equivalence

* 100 random input cases with relative error ≤ 1e-6.

### 9.4 Performance Regression

* Microbenchmarks (matmul/attention) should show ≥10% reduction in the L2 proxy or an increase in `Q`.

### 9.5 Determinism

* Same seed/time budget → identical permutation hash.

---

## 10) Observability (Metric Injection)

* Attach module attributes at pass exit:

  * `icd.metrics = { "Q0":f, "Q1":f, "C0":f, "C1":f, "pi_hash":"…" }`
* Emit event logs: `stage, elapsed_ms, improved, reason`.

---

## 11) Interactions with Other Passes

* **Upstream**: Run shape/dtype inference before `attach-icd`.
* **Downstream**: Finish `legalize-layout` before fusion/vectorization.
* **Conflict avoidance**: Add preservation rules so CSE/canonicalization does not drop layout annotations.

---

## 12) Security & Compliance

* Remove **personal data or raw samples** when generating trace-based $W$.
* Do not encode sensitive values (paths/accounts) in metadata.

---

## 13) Acceptance Criteria

* On both workloads (SSM/Transformer), after T2:

  * `Q1 > Q0`, `C1 < C0*(1-ε)`
  * Pass numerical equivalence, record zero legalization failures, and meet determinism tests.
* With the pass pipeline toggled on/off, **compile time** increases by ≤10% under default settings.

---

## 14) Implementation Notes

* Spectral step: Use Lanczos/LOBPCG for partial eigenvectors; sample for very large problems.
* Local search: Prefer adjacent swaps; cap 2-opt within the time budget.
* TVM bridge: Map layout tags to `transform_layout` / `schedule.bind` hints.

---

## 15) Open Issues (Roadmap)

* **KV-cache-specific $W$** (reflect sequence length/batch changes).
* **PIM/CIM constraints** (future): incorporate column/bandwidth weights into the cost model.
* **Learned cost-model plugin** (BO/RL) as an optional mode.

---

## 16) Reference Pseudocode

```python
def repermute_after_transform(mod):
    W  = build_coaccess(mod)          # A1′
    Q0, C0, pi0 = metrics_from(mod)
    pi1 = spectral_order(W)
    pi1 = local_refine(W, pi1, time_budget_s=cfg.time_budget)
    C1, Q1 = cost(W, pi1), modularity(W, pi1)

    if C1 < C0 * (1 - cfg.epsilon):
        mod = apply_layout(mod, pi1)  # T2
        mod = legalize_layout(mod)    # T3
        annotate(mod, Q0,Q1,C0,C1,pi1)
    else:
        annotate_no_improve(mod, Q0,Q1,C0,C1)
    verify(mod)                        # V1
    return mod
```

---

## 17) Release Checklist

* [ ] 20 FileCheck cases pass
* [ ] 100-case numerical equivalence with zero failures
* [ ] Determinism/time-budget tests pass
* [ ] Performance regression gate (Q↑/C↓) passes
* [ ] IR metadata round-trip test (attach → lower) passes

---

Consult the **SOP (measurement standard)**, **Cost Spec (cost model/tuning rules)**, and **test plan** for immediate follow-up steps.
