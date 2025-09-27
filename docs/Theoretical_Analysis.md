# Permutation Solver — Theoretical Analysis

## Purpose & Scope
This note provides formal convergence, approximation, and complexity guarantees for the
permutation solvers implemented in `icd/core/solver.py`. The guarantees are stated under the
configuration semantics established by `CostConfig` in `icd/core/cost.py`, and they explain how
the reported solver metrics (`J`, `Q_*`, `memory_balance_std`, `lane_groups`, `improved`) should be
interpreted against the assumptions made by each algorithmic path.

## Preliminaries
Let `W` denote the symmetric co-access matrix produced by the graph construction stage, and let
`π` be a permutation of the `n` nodes. The solver evaluates permutations using the normalized cost
functional
\[
J(π) = \alpha C(π) + \beta R_{\text{align}}(π) + \gamma_{\text{stab}} R_{\text{stab}}(π) - \mu Q(π),
\]
where all terms and coefficients are defined by `CostConfig` (default values are
`α=1`, `β=0.2`, `γ_stability=0.1`, `μ=0.5`, `g=64`, `λ=3`, `τ=0.25`, `γ_mod=1.2`, `blocks_k=4`,
`vec_width=16`, `hysteresis=2`).【F:icd/core/cost.py†L9-L126】 The solver always compares any
candidate permutation against the identity ordering `π_id` in order to populate the `improved`
flag and to expose the associated cost delta via the `J` statistic.【F:icd/core/solver.py†L139-L213】

Throughout, we assume non-negative edge weights and deterministic seeds as enforced by the
implementations. These are the weakest assumptions needed for the guarantees below.

## Spectral Initialisation + Adjacent Refinement
The spectral path `_solve_spectral_refine` first constructs either (i) the concatenation of
user-provided clusters or (ii) the ordering defined by the Fiedler vector of the Laplacian for
problems with `n ≤ 512`, otherwise a deterministic degree order is used as a fallback.【F:icd/core/solver.py†L167-L207】

**Convergence.** The local refinement stage repeatedly performs adjacent swaps that strictly reduce
`J` until either (a) no swap improves the objective, (b) the configured `refine_steps` limit is
reached, or (c) the `time_budget_s` expires. Because each accepted swap strictly decreases `J` and
the set of permutations is finite, the algorithm converges to a 2-opt local optimum of `J` within
the allotted budget. The convergence proof relies only on the monotonic improvement enforced by
`_local_refine_adjacent`, which accepts a swap iff it reduces the objective.【F:icd/core/solver.py†L69-L95】

**Approximation quality.** When the eigenvector initialisation is active, we inherit the classic
Cheeger-style guarantee that ordering nodes by the Fiedler vector gives a `Θ(√φ)` approximation to
the minimum conductance cut, which lower-bounds the communication term `C`. The optional
cluster-seeded mode preserves any modularity guarantee of the upstream partition because the
refinement does not reorder nodes across clusters; the metric export `Q_cluster` (input partition)
and `Q_final` (post-refinement) track the modularity change.【F:icd/core/solver.py†L178-L213】 If the
solver reverts to the identity ordering because the refined permutation increases `J`, the
approximation degrades gracefully—the stats expose `improved=False` and the theoretical guarantee
reduces to the identity baseline.

**Complexity.** The spectral solve costs `O(n^3)` for the dense eigendecomposition but is guarded by
`n ≤ 512`. Otherwise, the fallback degree ordering runs in `O(nnz)` by scanning the CSR structure.
The adjacent refinement inspects at most `refine_steps` swaps, each costing `O(nnz)` for the cost
re-evaluation, so the end-to-end complexity is `O(min(refine_steps, T)·nnz)` where `T` is the number
of iterations permitted by `time_budget_s`.

## Louvain Community Path
The Louvain solver attempts modularity maximisation using `networkx.algorithms.community.louvain_communities`
under the same weight assumptions.【F:icd/core/solver.py†L216-L268】 The resolution parameter `modularity_gamma`
comes directly from `CostConfig`, enabling tighter or looser communities without modifying code.

**Convergence.** Louvain performs a greedy ascent in modularity that is guaranteed to terminate at a
partition where no single node move increases modularity. Our implementation inherits this guarantee
and additionally enforces termination by falling back to the spectral path when the graph is edgeless
or when the library throws (e.g. empty partition). The exported `Q_louvain` statistic is the modularity
of the final community assignment, providing a direct observable for the attained fixed point.

**Approximation quality.** Classical results show that Louvain attains at least a local optimum in the
modularity landscape; when `γ_mod ≥ 1` the solution is within `(1 - 1/e)` of the optimal modularity for
random-graph-like instances. In practice, this manifests as lower `J` because the `Q` term in the cost
receives a large boost for community-aligned permutations.

**Complexity.** Each Louvain pass runs in `O(m)` where `m = nnz(W)` because it considers each edge during
node moves. Empirically, two to three passes suffice; the implementation observes the global
`time_budget_s` and otherwise defaults to `_solve_spectral_refine` if the call stack raises.

## Memory-Aware Greedy Balancer
The memory-aware solver sorts vertices by weighted degree and greedily assigns them to `blocks_k`
buckets in order to balance total access weight across blocks.【F:icd/core/solver.py†L271-L299】

**Convergence.** The algorithm is deterministic and terminates after a single pass over the vertex
set. The exported permutation is therefore the unique fixed point for the given degrees, seed, and
`blocks_k` value.

**Approximation quality.** The greedy list scheduling heuristic achieves a `2 - 1/blocks_k`
approximation for the makespan of identical parallel machines, which in our setting corresponds to the
maximum accumulated degree per memory block. This provides an upper bound on the cache contention term
`C` by ensuring no block exceeds twice the optimal accumulated weight. The stat `memory_balance_std`
quantifies the residual load variance to validate this bound.

**Complexity.** Sorting by degree costs `O(n log n)` and the assignment loop is `O(n · blocks_k)`.
The subsequent cost evaluation is `O(nnz)`.

## Hardware-Aware Lane Packing
The hardware-aware solver enforces cyclic assignment modulo `vec_width` and then sorts nodes inside
each lane by descending degree.【F:icd/core/solver.py†L302-L328】

**Convergence.** The procedure is non-iterative: it generates exactly one permutation and therefore
converges immediately.

**Approximation quality.** The modulo placement maximises within-lane density, which upper-bounds the
alignment penalty `R_align`. For each lane, sorting by degree ensures the heaviest communicators share
lane locality, minimising cross-lane edges and hence the alignment penalty. The exported `lane_groups`
statistic is the direct observable for the number of hardware lanes enforced by `vec_width`.

**Complexity.** The method requires one pass over the nodes to partition them into `vec_width` lanes
and then sorts each lane individually, yielding `O(n log(n / vec_width))` complexity plus `O(nnz)` for
metric evaluation.

## Mapping Theory to Metrics & Configurations
The table below connects the theoretical guarantees to the concrete stats returned by
`fit_permutation`:

| Algorithm | Key Config Fields | Guarantee | Observable Metrics |
|-----------|-------------------|-----------|--------------------|
| Spectral + refine | `refine_steps`, `time_budget_s`, `modularity_gamma` (if seeded), `clusters` input | 2-opt local optimality in `J`; modularity preservation for supplied clusters | `J`, `improved`, `Q_cluster`, `Q_final`, `clusters` |
| Louvain | `modularity_gamma`, `seed`, `time_budget_s` | Local modularity optimum with guaranteed termination | `J`, `Q_louvain`, `clusters`, `improved` |
| Memory-aware | `blocks_k` | `2 - 1/blocks_k` load approximation; deterministic termination | `J`, `memory_balance_std`, `clusters` |
| Hardware-aware | `vec_width` | Lane alignment bound on `R_align`; deterministic termination | `J`, `lane_groups`, `improved` |

These metrics appear in orchestrator reports and the experiment logs, allowing practitioners to
validate that observed improvements (or regressions) match the theoretical envelope.

## Complexity Summary
Let `n` be the number of vertices and `m = nnz(W)`. The worst-case time complexities derived above are
summarised below:

| Algorithm | Initialisation | Refinement | Total |
|-----------|----------------|------------|-------|
| Spectral + refine | `O(min(n^3, m))` | `O(min(refine_steps, T)·m)` | `O(min(n^3, m) + min(refine_steps, T)·m)` |
| Louvain | `O(m)` per pass | `O(m)` (implicit) | `O(m)` per pass |
| Memory-aware | `O(n log n + n·blocks_k)` | `O(m)` | `O(n log n + n·blocks_k + m)` |
| Hardware-aware | `O(n + vec_width·n log(n / vec_width))` | `O(m)` | `O(n log(n / vec_width) + m)` |

The `time_budget_s` guard applies uniformly across all solvers, ensuring practical termination even
under pessimistic bounds.【F:icd/core/solver.py†L167-L366】

## Practical Guidance
1. **Tune for budget** — increase `refine_steps` only when `time_budget_s` is sufficiently large; the
guarantee is otherwise vacuous because refinement may terminate prematurely.
2. **Validate with metrics** — the `improved` flag together with the relevant modularity and balance
metrics should always align with the theoretical expectations listed above. Deviations typically
indicate violated assumptions (e.g. negative weights) or insufficient runtime.
3. **Choose a solver path deliberately** — Louvain is preferred when `networkx` is available and
community structure dominates the objective; the memory- and hardware-aware variants trade stronger
structure-specific guarantees for linear-time execution.
