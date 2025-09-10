**S/Q/K Adapter Spec (V1)**

- **Scope:** Define Sparsity (S), Quantization (Q), and KV‑Cache (K) adapters’ inputs/outputs/exceptions, quality impact bounds, `delta_layout` trigger conditions, and metadata schema. Aligns with `icd/adapters/*` stubs and orchestrator wiring plan.

**Common Contract**
- **Function Signatures:**
  - `apply_sparsity(tensor, *, type:"2:4"|..., rate:float) -> (tensor, meta)`
  - `apply_quant(tensor, *, dtype:"int8"|"fp8"|..., method:"ptq-minmax"|...) -> (tensor, meta)`
  - `apply_kvcache(cache, *, block:int, drop:float) -> (cache, meta)`
- **Inputs:** immutable views or copy-on-write; adapters must not mutate in‑place unless explicitly documented.
- **Outputs:** transformed object of same shape semantics (unless documented), and `meta` with `delta_layout: bool`, quality deltas, and adapter‑specific fields.
- **Exceptions:** Raise `ValueError` for invalid cfg; never raise on safe no‑op; return original + `delta_layout=false`.

**Metadata Format (`transform_meta`)**
- **Common Keys:**
  - `delta_layout: bool` — whether layout change should trigger re‑permutation.
  - `quality_delta_pp: float` — estimated accuracy delta in percentage points (positive = improve).
  - `notes?: str` — optional rationale string for CI logs.
- **Adapter Keys:**
  - Sparsity: `{sparsity: {type: str, rate: float}}`
  - Quant: `{quant: {dtype: str, method: str}}`
  - KV: `{kv: {block: int, drop: float}}`

**Delta‑Layout Trigger Rules**
- Sparsity (S): `delta_layout = (rate >= 0.25)`; rationale: density change crosses tiling/coalescing thresholds, often alters optimal π.
- Quant (Q): `delta_layout = (dtype in {int8, fp8})` due to vector width/align changes and cacheline packing.
- KV (K): `delta_layout = (block >= 64)` implying different attention windowing and memory stride.
- Multiple adapters: combine via `delta_layout = any(child.delta_layout)`; attach a `triggers: ["S","Q","K"]` list in orchestrator meta (planned) for auditability.

**Quality Impact Bounds**
- Provide adapter‑local upper bounds to guard automatic application without offline calibration:
  - S: For structured `2:4` with rate≤0.5, expected accuracy drop ≤ 0.1pp on typical LLM evals (document references when available).
  - Q: For `int8` PTQ with calibration set, bound ≤ 0.2pp; for `fp8` with dynamic scaling, bound ≤ 0.1pp. If calibration missing, set `quality_delta_pp = None` and require manual gate.
  - K: For KV block≥64 and drop≤0.1, bound ≤ 0.1pp on perplexity. Above bounds should be validated per‑model.
- In CI smoke, adapters keep `quality_delta_pp=0.0` unless eval is enabled; orchestrator writes `quality=None` by default.

**Error Handling & No‑Op**
- Invalid config (unknown dtype, negative rate, block<1) → `ValueError` with clear message.
- Inference‑only flows should default to no‑op with `delta_layout=false` when unsupported environment is detected.

**Adapter Details**
- Sparsity
  - Types: `2:4`, `1:1` (unstructured), blocks `(n:m)`; must be validated.
  - Heuristics: prefer keeping last‑dim dense when sectioning indicates attention features mapped last.
  - Meta examples: `{sparsity:{type:"2:4", rate:0.5}, delta_layout:true, quality_delta_pp:0.0}`.
- Quantization
  - DTypes: `int8`, `fp8`, `int4` (planned). Methods: `ptq-minmax`, `ptq-kld`, `aqt` (planned).
  - Alignment: set `delta_layout` true when vector width or tile size differs from baseline.
  - Meta: `{quant:{dtype:"int8", method:"ptq-minmax"}, delta_layout:true, quality_delta_pp:0.0}`.
- KV‑Cache
  - Params: `block` (token block), `drop` (temporal downsampling).
  - Stride/Align: large blocks change page locality and prefetch behavior → possible π change.
  - Meta: `{kv:{block:128, drop:0.1}, delta_layout:true, quality_delta_pp:0.0}`.

**Integration with Runtime**
- Orchestrator applies adapters per config stage, merges metas:
  - `transform_meta = {delta_layout:any, metas:[...], triggers:[...]}` (planned schema) and emits `TRANSFORMED` event.
  - If `delta_layout` true and `pipeline.mode=iterative`, rerun `fit_permutation` under time budget.
- Caching: include adapter meta in cache key to avoid stale π reuse.

**Testing Guidance**
- Unit: verify delta rules and schema; no in‑place mutation; exceptions on invalid cfg.
- Integration: ensure `delta_layout` triggers re‑perm loop; artifacts unchanged on no‑op.
- Determinism: same tensor/cfg yields identical meta.

**Open Items / TODO**
- Add calibration hooks to populate `quality_delta_pp`.
- Define merged meta schema in code and propagate through IR bridge (attach as IR attributes when applicable).

**Self‑Review**
- Mirrors current stubs in `icd/adapters/` and fits orchestrator’s planned `delta_layout` wiring. Leaves quality estimation and merged meta structure as explicit TODOs.

