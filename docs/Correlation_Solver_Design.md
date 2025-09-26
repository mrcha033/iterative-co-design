# Correlation Graph & Solver Upgrade — Design Notes

## Goals
- Replace mock/heuristic co-access matrices with data-driven correlation graphs derived from activation or weight statistics.
- Integrate community detection (Louvain / spectral) into the permutation solver to maximize modularity while honoring cost constraints.
- Preserve determinism: identical seeds and inputs must yield identical graphs, clusters, and permutations.
- Maintain observability: persist intermediate artifacts (`runs/<tag>/correlation/*`) for reproducibility and QA gates.

## Activation / Weight Trace Collection
- **Entry point**: new module `icd/graph/correlation.py` exposing
  `collect_activation_stats(model, example_inputs, layers, *, mode="activation", seed, dtype, device_guard, samples)`.
- **Activation mode**:
  - Register forward hooks on selected layers (default: modules matching hidden dimension `D`).
  - For each inference batch (min(N, samples)), accumulate mean-centered outer product `x^T x`.
  - Use `torch.no_grad()`, `torch.inference_mode()`, optionally `torch.cuda.amp.autocast(False)` for determinism.
  - Store running sums in `float64` on CPU; flush to disk as `.npz` (`mean`, `cov`, `count`).
- **Weight mode**:
  - Compute cosine similarity or overlap metrics between rows/columns of specific weight tensors.
  - Useful when activation data unavailable; still deterministic.
- **Seeding & ordering**:
  - Lock RNGs (`torch.manual_seed`, CUDA deterministic algorithms).
  - Snapshot layer order to maintain consistent mapping between tensor indices and graph nodes.
- **Configuration**: extend `graph` config with `correlation` block (layers, sample_count, mode, subsampling).

## Graph Construction
- Convert accumulated covariance matrix `C` (size `D x D`) to sparse adjacency `W`:
  - Threshold small weights, enforce non-negativity, optionally normalize by degree.
  - Support row/symmetric normalization (reuse existing `_normalize_row/_normalize_sym`).
  - Provide deterministic pruning (`top_k_per_row`, `nnz_cap`).
- Persist artifacts:
  - `runs/<tag>/correlation/activations.json` (metadata: layers, samples, seed, dtype).
  - `runs/<tag>/correlation/C.npy` (dense or chunked) or CSR via `save_w_npz`.
  - `runs/<tag>/correlation/feature_map.json` mapping layer names → index ranges.

## Clustering / Modularity Maximization
- Introduce dependency on `networkx` (or lightweight Louvain implementation). Guard import; fall back to heuristic with warning if unavailable.
- Algorithm options:
  1. **Louvain** (default): `community-louvain` package for modularity optimization.
  2. **Spectral clustering**: use SciPy/NumPy eigenvectors when `k` specified.
- Deterministic runs:
  - Expose `solver.rng_seed`; pass to clustering algorithm where supported.
  - Retry loop (fixed iterations) capturing best modularity; break ties with lexicographic order of communities.
- Output: list of clusters (list of node indices). Build permutation by concatenating clusters sorted by modularity contribution (or size).
- Store summary: `runs/<tag>/correlation/clusters.json` containing cluster sizes, modularity score, algorithm, seed, runtime.

## Solver Integration
- Extend `fit_permutation` signature:
  - Accept optional `initial_partition` or `cluster_permutation` produced by clustering stage.
  - Compute modularity using exact Newman formulation on CSR (shared utility `modularity(W, partition)`).
  - Retain local adjacent refinement to fine-tune intra-cluster ordering while ensuring monotonic decrease in cost `J`.
- Update stats dict to include:
  - `Q_before`, `Q_after`, `cluster_count`, `modularity_improved`.
  - `time_cluster_s`, `time_refine_s`.
- Cache interplay:
  - Extend permutation cache key with correlation signature hash (covariance metadata + clustering config).

## CLI / Config Updates
- `docs/ICD.md` / schema: add `graph.correlation` and `solver.clustering` blocks.
- CLI overrides for quick experiments: `--override graph.correlation.layers=[...]`, `--override solver.clustering.method=louvain`.
- Provide helper script `scripts/collect_correlation.py` to precompute traces.

## Testing Strategy
- Unit tests:
  - Deterministic activation collector on toy model: verify covariance matches manual computation.
  - Clustering output on synthetic block matrix with known communities (compare modularity vs heuristic).
  - Solver integration keeps/improves modularity vs identity.
- Integration smoke:
  - Extended `tests/integration/test_orchestrator_correlation.py` running mock model with correlation path and verifying artifacts exist.
- Performance guard:
  - Add profiling hooks to ensure collection/rescaling stays within budget for `D≈2.5k` (`SAS` constraint <5 min re-permute).

## Open Questions / Follow-ups
- Evaluate dependency footprint (`networkx` vs `python-louvain` vs custom). Consider optional extra `[graph]` extras requirement.
- Decide on default layer selection heuristics (e.g., final hidden state vs all hidden states) for models like BERT vs Mamba.
- Investigate GPU-to-CPU transfer overhead; potential to downsample or chunk activations to avoid O(D^2) memory explosion.
- Plan fallback when no correlation data available: revert to current heuristic with warning + gate flag.

