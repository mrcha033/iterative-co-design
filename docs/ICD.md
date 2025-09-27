# ICD — Interface Control Document

*Iterative HW–SW Co-Design: Memory-Aware Layout Re-Optimization*

---

## 1) One-line summary

This document defines the **official interface contracts** for the `icd` CLI and Python library. It specifies input/output schemas, exceptions and error codes, versioning/compatibility rules, determinism and performance guards, plus logging and artifact formats. (Audience: ML/system/compiler/IR engineers, test/release PMs.)

---

## 2) Assumptions & Constraints

* Language/runtime: **Python 3.10**. Optional PyTorch device guard; TVM/StableHLO appear as PoC plugins.
* Hardware: Single GPU (A100/H100 recommended). Nsight Compute (ncu) and NVML power sampling are optional.
* Scope: **Inference** path only (exclude training and distributed scheduling).
* Versioning: semver. Initial release `v0.1.0` (experimental).

---

## 3) Deliverables (Interface Specifications)

Refer to docs/USAGE.md for run instructions and artifact examples.

### 3.1 Terminology & Base Types

* `D`: State/feature dimension (int).
* `W`: Shared-access weight matrix (symmetric, non-negative, sparse CSR).
* `π` (pi): Permutation, `int32[D]` representing a permutation of 0..D-1.
* `Q`: Modularity scalar (float64).
* `C`: Cache-proxy cost scalar (float64).
* `RunId`: Content-addressed run hash, `sha256[:12]`.
* Time units: ms (latency), s (sampling), Hz (frequency), W (power), J/token (EpT).

---

### 3.2 Python API (Stable Contract)

All APIs are **synchronous**. Exceptions follow §3.7 “Errors & Exceptions”.

```python
# graph
def build_w(
    source: Literal["trace", "mock"],
    *,
    trace: Optional[Iterable[tuple[int, int, float]]] = None,  # (i,j,weight)
    D: Optional[int] = None,
    blocks: int = 4,
    noise: float = 0.02,
    seed: int = 0,
    normalize: Literal["none", "row", "sym"] = "sym",
) -> "csr_matrix":
    """Create the shared-access weight matrix W from a trace or mock parameters.
    Contract: deterministic (same input → same W), zero diagonal, non-negative, returns CSR."""

def fit_permutation(
    W: "csr_matrix",
    *,
    time_budget_s: int = 300,
    refine_steps: int = 2_000,
    k_blocks: Optional[int] = None,      # modularity community hint
    rng_seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Spectral (Fiedler ordering) initialization + local search.
    Returns: (π:int32[D], stats: {"Q":float, "C":float, "iters":int, "elapsed_s":float, "improved":bool})"""

# S/Q/K adapters — return metadata used to decide whether to re-permute
def apply_sparsity(
    tensor_or_path, *,
    type: Literal["2:4", "unstructured"] = "2:4",
    rate: float = 0.5,
) -> tuple[Any, dict]:
    """meta = {"kind":"sparsity","type":"2:4","rate":0.5,"delta_layout":True}"""

def apply_quant(
    tensor_or_path, *,
    dtype: Literal["int8", "fp8"] = "int8",
    method: Literal["ptq-minmax", "ptq-kld"] = "ptq-minmax",
) -> tuple[Any, dict]:
    """meta = {"kind":"quant","dtype":"int8","method":"ptq-minmax","delta_layout":True}"""

def apply_kvcache(
    cache, *,
    block: int = 128,
    drop: float = 0.10,
) -> tuple[Any, dict]:
    """meta = {"kind":"kv","block":128,"drop":0.10,"delta_layout":True}"""

from icd.graph import CorrelationConfig, collect_correlations, correlation_to_csr
from icd.graph import ClusteringConfig, cluster_graph

corr_cfg = CorrelationConfig(mode="activation", samples=8, threshold=0.05)
matrix, corr_meta = collect_correlations(model, example_inputs, cfg=corr_cfg)
W_corr = correlation_to_csr(matrix, cfg=corr_cfg)

cluster_cfg = ClusteringConfig(method="louvain", rng_seed=0)
communities = cluster_graph(W_corr, cluster_cfg)

# orchestrator
@dataclass
class RunArtifacts:
    run_id: str
    out_dir: str
    metrics: dict   # schema in §3.4
    paths: dict     # artifact file paths

def run(config: dict) -> RunArtifacts:
    """Execute the pipeline:
    (1) Build W → (2) π₀ → (3) Transform → (4) π₁ → (5) Measure → (6) Report/cache.
    See §3.3 for the config schema."""

# Measurement utilities (can be used directly)
from icd.measure.runner_gpu import BenchmarkConfig, benchmark_inference

cfg = BenchmarkConfig(repeats=200, warmup=20, tokens_per_batch=1024)
metrics = benchmark_inference(model, example_inputs, cfg)

def measure_latency(fn, *, repeats:int=1000, warmup:int=50, fixed_clock:bool=True) -> dict: ...
def measure_l2_hit(ncu_cmd: str, *, metrics: list[str]) -> dict: ...
def measure_power_nvml(*, sample_hz:int=10, duration_s:int) -> dict: ...
```

#### Determinism contract

* Identical `seed`/`rng_seed`/fixed-clock options must yield **the same π, the same W, and results within ±ε**.
* Must finish within `time_budget_s`; exceeding it raises `TimeoutError` (with `stats.improved=False`, returning the initial solution when allowed).

---

### 3.3 `run(config)` Input Schema (YAML/JSON equivalent)

```yaml
pipeline:
  mode: iterative              # linear | iterative
  repermute_on_delta: false    # if true, re-permute when transform_meta.delta_layout=true (even in linear mode)
  repeats: 1000
  warmup_iter: 50
  fixed_clock: true
  runner: icd.runtime.runners.mock_inference   # optional dotted callable path
  runner_context:
    tokens: 1024
graph:
  source: trace                # trace | mock
  trace: []                    # list of (i,j,weight) or a file path
  mock: {D: 2560, blocks: 4, noise: 0.02, seed: 42}
  normalize: sym               # none|row|sym
  loader: icd.experiments.hf.load_hf_sequence_classifier   # optional PyTorch loader
  loader_kwargs:
    model_name: bert-base-uncased
    sequence_length: 128
    batch_size: 4
solver:
  time_budget_s: 300
  refine_steps: 2000
  k_blocks: 4
  rng_seed: 0
transform:
  sparsity: {enable: true, type: "2:4", rate: 0.5}
  quant:    {enable: false, dtype: "int8", method: "ptq-minmax"}
  kv:       {enable: false, block: 128, drop: 0.10}
measure:
  ncu_enable: true
  ncu_metrics: ["l2_tex__t_sector_hit_rate.pct"]
  power_enable: true
  power_sample_hz: 10
report:
  out_dir: "runs/exp001"
  formats: ["html","csv"]
  # If specified, only generate these formats; otherwise produce both HTML and CSV.
cache: {enable: false, cache_dir: ".icd_cache"}
```

* **Contract**: If `pipeline.mode=linear`, skip Step 4 (re-permute) unless `repermute_on_delta=true` and the transform metadata contains `delta_layout=true`.
* **Defaults**: Fall back to PRD/SAS defaults when unspecified.
* **Iterative guard**: With `pipeline.mode=iterative`, at least one of `transform.*.enable=true` or `graph.correlation.enable=true` must be set.
* **Cache**: Active only when both `cache.enable=true` and `cache.cache_dir` are provided.
* **Validation**: Schema violations raise `ConfigError`.
* **Runner**: If `pipeline.runner` names a `module:function` (or dotted path), that callable executes the actual inference/measurement loop. Its return dictionary must propagate `l2_hit_pct`, `ept_j_per_tok`, and `tokens` to the metrics. Without it, the mock proxy runner is used.
* **Runner context fields**:

  | Key | Description |
  | --- | ----------- |
  | `model_loader` | Dotted-path function returning `(model, example_inputs)` for runner/init dual use. |
  | `model_loader_args` / `model_loader_kwargs` | Positional/keyword arguments passed to `model_loader`. |
  | `graph_model` / `graph_example_inputs` | Objects created during graph construction. The orchestrator injects them into the runner context for reuse. |

* **PyTorch loader**: When `graph.loader` is set, the dotted-path function must return `(model, example_inputs)`. Reuse the same function via `pipeline.runner_context.model_loader` to share the model between graph building and measurement.

---

### 3.4 Output Schema (Metrics & Artifacts)

**Metrics (JSON)**

```json
{
  "run_id": "a1b2c3d4e5f6",
  "hardware": {"gpu":"A100-40GB","driver":"535.xx"},
  "env": {"seed":0,"fixed_clock":true},
  "pipeline": {"mode":"iterative","repeats":1000,"warmup":50},
  "graph": {"D":2560,"nnz":123456,"normalize":"sym","source":"trace"},
  "solver": {"Q0":0.31,"C0":1.23e7,"Q1":0.47,"C1":9.01e6,"elapsed_s":212.4,"improved":true},
  "transform": {"sparsity":{"type":"2:4","rate":0.5},"quant":null,"kv":null},
  "latency_ms": {"mean": 12.8, "p50": 12.6, "p95": 13.5},
  "throughput_toks_s": 1843.2,
  "l2_hit_pct": 87.4,
  "ept_j_per_tok": 0.92,
  "quality_delta_pct": 0.03
}
```

**Artifacts (paths)**

```
{out}/
  W.csr.npz            # Shared-access weight (JSON payload; no NumPy dependency)
  w.meta.json          # Graph metadata (PyTorch path adds ops.json)
  perm_before.json     # π₀
  perm_after.json      # π₁ (iterative)
  ncu.json             # L2/profile results (optional)
  power.csv            # Power samples (optional)
  report.html
  report.csv
  run.log              # Events/errors
  config.lock.json     # Final resolved configuration
```

**File format contracts**

* `perm*.json`: `{"D":2560,"pi":[...int...],"hash":"sha256"}`
* `W.csr.npz`: JSON payload with CSR arrays (`indptr`, `indices`, `data`, `shape`, `meta`), zero diagonal, non-negative. (SciPy npz remains an optional native path later.)
* `ncu.json`: Reduced map of metric name → value from the ncu CLI output.
* `report.*`: Auto-generated pre/post comparison tables and figures.

---

### 3.5 CLI (Stable Contract)

```
python -m icd.cli.main run -c config.json
  --override pipeline.mode=iterative
  --override solver.time_budget_s=180
  --out runs/exp001
```

Options:

* `--dry-run`: Validate schema/contract only.
* `--print-schema`: Dump the current input schema skeleton.
* `--no-measure`: Skip measurement/reporting and run only the solver.
* `--reuse-perm PATH`: Reuse a prior π (perm_before.json format, cache/reuse path).
* (Planned) `--no-measure`: Run only the solver (no benchmarking).

**Exit codes**

* `0` success, `2` ConfigError, `3` ResourceError (GPU/tool unavailable), `4` MeasureError, `5` SolverTimeout.

---

### 3.6 IR Bridge (StableHLO/TVM PoC) Metadata Contract

**Tensor/IR attribute keys**

* `icd.layout_perm`: `i32[D]` (π)
* `icd.layout_tag`: `"icd/v1"` (version tag)
* `icd.coaccess_block`: `i32` (block hint)
* `icd.transform_meta`: JSON string (e.g., `{"sparsity":{"type":"2:4","rate":0.5},...}`)

**Example pass pipeline insertion**

* Frontend → **attach-icd-metadata** → (optional) **icd-layout-pass** → Lowering.
* Contract: Tensors without metadata tags are **ignored (no-op)**. Emit warnings when tags mismatch.

---

### 3.7 Errors & Exceptions (Taxonomy)

| Code         | Exception/State         | Cause/Description                     | Handling                 |
| ------------ | ----------------------- | ------------------------------------- | ------------------------ |
| `E_CFG_001`  | `ConfigError`           | Schema violation/type mismatch        | Abort immediately        |
| `E_RES_101`  | `ResourceError`         | GPU/driver/ncu/NVML unavailable       | Downgrade if possible / abort |
| `E_SLV_201`  | `TimeoutError`          | Exceeded `time_budget_s`              | Return initial solution + flag |
| `E_SLV_202`  | `SolverNotConverged`    | Numerical failure/NaN                 | Roll back to initial solution |
| `E_ADP_301`  | `TransformError`        | S/Q/K application failed              | Warn, perform no-op, continue |
| `E_MSR_401`  | `MeasureError`          | ncu/NVML execution failed             | Substitute wall-clock metrics |
| `E_IO_501`   | `ArtifactError`         | File write/permission issue           | Abort                      |
| `E_INT_900`  | `InternalError`         | Unexpected exception                  | Abort (file a bug)         |

**Contract**: Maintain **recoverable rollback paths** when available and record failure causes/alternatives in the report.

---

### 3.8 Event & Logging Schema (Observability)

```json
{"ts":"2025-09-09T08:11:12Z","stage":"BUILD_W","ok":true,"meta":{"D":2560,"nnz":123456}}
{"ts":"2025-09-09T08:11:13Z","stage":"PERMUTE","ok":true,"meta":{"Q":0.31,"C":1.23e7,"elapsed_s":4.1}}
{"ts":"2025-09-09T08:11:30Z","stage":"TRANSFORM","ok":true,"meta":{"kind":"sparsity","rate":0.5}}
{"ts":"2025-09-09T08:15:02Z","stage":"REPERMUTE","ok":true,"meta":{"Q":0.47,"C":9.01e6}}
{"ts":"2025-09-09T08:15:20Z","stage":"MEASURE","ok":true,"meta":{"lat_ms":12.8,"l2_hit_pct":87.4,"ept":0.92}}
{"ts":"2025-09-09T08:15:21Z","stage":"REPORT","ok":true,"meta":{"out":"runs/exp001"}}
```

* Overhead ceiling: **<1%** (adjust sampling levels accordingly).
* Log levels: `INFO/DEBUG/WARN/ERROR`. Do not log sensitive information.

---

### 3.9 Compatibility & Versioning Policy

* **semver**: `MAJOR.MINOR.PATCH`.

  * MINOR releases may **add new keys** when defaults exist.
  * Only MAJOR releases may **remove keys or change semantics**.
* Freeze actual keys/defaults in `config.lock.json`.
* Anchor the IR metadata version via `icd.layout_tag`.

---

### 3.10 Cache & Reuse Contract

* Cache key: `hash(model_id, task_id, S/Q/K meta, D, seed, solver params)`
* On cache hit, skip `fit_permutation` (optionally revalidate with a flag).
* Automatically ignore mismatches (emit warnings).

---

## 4) Testing (Contract Validation)

* **Schema tests**: Invalid types/missing fields must raise `ConfigError`.
* **Determinism tests**: Same input/seed/clock → π/metrics variation **≤ ε (1%)**.
* **Error-path tests**: ncu unavailable → raise `MeasureError`, then downgrade to wall-clock metrics.
* **Performance contract tests**: For D=2.5k with `time_budget_s=300`, ensure `elapsed_s ≤ 300` and `stats.improved` true frequency ≥ 95%.
* **IR tag round-trip tests**: Attach → pass → lower while preserving `icd.layout_perm`.

---

## 5) Performance Notes, Risks, Alternatives

* **Spectral cost**: O(D³); mitigate with partial eigenvectors, sampling, or block partitioning.
* **Measurement noise**: Enforce fixed clocks, warmup, and repetitions N (≥1000).
* **IR diversity**: Initially limit to **metadata round-tripping**; expand transformations incrementally in the Pass Doc.

---

## 6) Appendix — JSON Schema (Excerpt)

**Input schema (`run.config`)**

```json
{
  "type":"object",
  "properties":{
    "pipeline":{"type":"object","properties":{
      "mode":{"enum":["linear","iterative"]},
      "repeats":{"type":"integer","minimum":1},
      "warmup_iter":{"type":"integer","minimum":0},
      "fixed_clock":{"type":"boolean"}
    },"required":["mode"]},
    "graph":{"type":"object","properties":{
      "source":{"enum":["trace","mock"]},
      "trace":{"anyOf":[
        {"type":"array","items":{"type":"array","items":[{"type":"integer"},{"type":"integer"},{"type":"number"}], "minItems":3,"maxItems":3}},
        {"type":"string"}]},
      "mock":{"type":"object","properties":{
        "D":{"type":"integer","minimum":2},
        "blocks":{"type":"integer","minimum":1},
        "noise":{"type":"number","minimum":0.0},
        "seed":{"type":"integer"}
      }},
      "normalize":{"enum":["none","row","sym"]}
    },"required":["source"]},
    "solver":{"type":"object","properties":{
      "time_budget_s":{"type":"integer","minimum":1},
      "refine_steps":{"type":"integer","minimum":0},
      "k_blocks":{"type":["integer","null"]},
      "rng_seed":{"type":"integer"}
    }},
    "transform":{"type":"object"},
    "measure":{"type":"object"},
    "report":{"type":"object","properties":{
      "out_dir":{"type":"string"},
      "formats":{"type":"array","items":{"enum":["html","csv"]}}
    }}
  },
  "required":["pipeline","graph","solver","report"]
}
```

**See §3.4 for the output metrics schema.**

---

### 7) Related Documents

* **PRD**: Goals, metrics, acceptance criteria
* **SAS**: Components, data, flow, budgets
* **SOP (measurement standard)**: Fixed clock, warmup, repeat rules
* **Pass Design Doc**: IR insertion points and transformation rules
* **Cost Spec**: Definitions and tuning rules for C(π) and Q
