**Graph Construction Spec (A0/V1)**

- **Scope:** Co-access weight matrix `W` construction for iterative layout optimization. Covers inputs, normalization, noise/filtering, CSR caps/validation, and determinism rules, aligned to current mock/PyTorch pathways.

**Inputs**
- **Source:** `graph.source in {mock, pytorch, trace}`; implemented: `mock`, `pytorch` (partial), `trace` planned.
- **Config Keys (common):**
  - **D/d:** target dimension (nodes) when synthetic.
  - **blocks:** block count for synthetic blockiness.
  - **noise:** additive jitter amplitude in [0,1].
  - **seed:** RNG seed; must drive all stochasticity.
  - **normalize:** `sym|row|none`.
  - **nnz_cap:** hard cap for sparsity, else default `min(0.05*D^2, 50e6)`.

**Input Schemas**
- `mock`: simple scalar fields above.
- `pytorch`: object with
  - **model:** handle to model object (runtime-provided).
  - **example_inputs:** sample inputs (tuple/list/tensor) for tracing.
  - **pytorch.hops:int, reuse_decay:float, used_ops:list[str], skipped_ops_count:int, attention?:obj, trace_hash?:str**
  - Output meta files (when pytorch): `w.meta.json`, `w.ops.json` include D, nnz, normalize, band_kernel, op_weights, seed, trace_* as emitted in `icd/runtime/orchestrator.py:70`.
- `trace` (spec): newline-delimited JSON records
  - `{"t": <ns>, "op": <str>, "src": <node_id>, "dst": <node_id>, "w": <float>, "meta": {...}}`
  - Must sort-stable by `t`; missing `w` defaults to 1.0; `node_id` in [0, D-1].

**Construction Rules**
- **Mock (implemented)**
  - Build blocky, symmetric, positive W; skip diagonal. Upper-triangular stored; `to_dense()` symmetrizes for analysis.
  - Noise: uniform in [−noise, +noise] added to base weights then clamped at 0.
- **PyTorch (implemented, partial)**
  - Build via op‑level reuse features; sectioning kernel band generation per `hops`/`reuse_decay` (see `core/graph_pytorch.py`).
  - Meta captures op inventory and trace hash; normalize+cap same as mock.
- **Trace (planned)**
  - Aggregate edge weights by (src,dst) summation; reject negatives/NaNs; drop diagonal.

**Normalization**
- `sym`: scale `W <- D^{-1/2} W D^{-1/2}` using degree approximation (see `core/graph._normalize_sym`).
- `row`: optional row‑stochastic variant (TBD; not currently implemented). If selected, fall back to `sym` with flag noted.
- `none`: no normalization.

**Noise & Filters**
- Noise applied only in mock construction stage.
- After normalization, apply cap/prune then per‑row L1‑normalize on kept entries to preserve relative scale (see `_cap_and_prune`).
- Filters must never introduce NaN/Inf or negative weights.

**CSR Cap & Validation**
- **Cap:** `nnz_cap = min(0.05*D^2, 50_000_000)` unless explicitly set.
- **Pruning:** keep top‑k per row proportional to `nnz_cap / current_nnz` (stable sort by weight desc).
- **Validate:**
  - `len(indptr)==D+1`, `0 <= indices[j] < D`.
  - `data[j] >= 0`, `not math.isfinite(data[j])` → error.
  - `meta` must include: `format='csr'`, `source`, `seed`, `blocks/noise` when mock, `normalize` used, `nnz_before/after` when pruned.

**Determinism Rules**
- For identical inputs and config:
  - `mock`: deterministic via `random.Random(seed)` only; never use global RNG.
  - `pytorch`: tie any randomness to provided `rng_seed` and deterministic backend flags; include `trace_hash` in meta.
  - `trace`: determinism by pure functional aggregation; sorted input required.
- CI Gate (U‑05): Matching CSR JSON payload hash for equal `(source,cfg)`.

**Outputs**
- In‑memory: `CSRMatrix{indptr,indices,data,shape=(D,D),meta}`.
- Files:
  - `W.csr.npz` (JSON payload; no NumPy hard dependency) via `save_w_npz`.
  - Optional when pytorch: `w.meta.json`, `w.ops.json`.

**Open Items / TODO**
- Implement `row` normalization path and doc its math.
- Add `trace` loader and schema validator.
- Emit standalone `meta.json` for mock to match artifact list in AGENTS.

**Self‑Review**
- Matches current code paths in `core/graph.py` and `core/graph_pytorch.py`.
- Determinism and cap/prune behavior align with unit tests; row‑norm noted as TBD coherently.

