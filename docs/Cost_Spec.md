# Cost Spec — Memory-Aware Layout Re-Optimization

**One-line summary**
Define the objective $J(\pi)$ that combines the shared-access weight matrix $W$ and permutation $\pi$ via the **cache-distance cost $C(\pi)$**, **modularity $Q(\pi)$**, and **alignment/stability regularizers $R(\pi)$**, then calibrate the objective to real hardware so that improvements map directly to better L2 hit, latency, and EpT.

---

## 1) Assumptions & Constraints

* **Scope**: Inference pipeline. Focus on 1D permutations within a tensor feature dimension (or block) and collapse multi-axis cases into block tags.
* **Inputs**: $W\in\mathbb{R}_{\ge 0}^{D\times D}$ (symmetric, sparse) and the initial/previous permutation $\pi_0$.
* **Measurements**: Use standard metrics such as the Nsight Compute L2 hit rate (sector-based).【F:docs/Cost_Spec.md†L66-L70】
* **Generality**: Adopt the weighted-network modularity definition (**Newman 2006** classic). Introduce the resolution parameter $\gamma$ to avoid resolution limits (multi-resolution).【F:docs/Cost_Spec.md†L71-L80】

---

## 2) Objective (Top Level)

$$
\min_{\pi\in \mathfrak{S}_D}~ J(\pi)\;=\;\alpha~\underbrace{\widetilde{C}(\pi)}_{\text{cache distance}}\;+\;\beta~\underbrace{R_{\text{align}}(\pi)}_{\text{alignment / vectorization}}\;+\;\gamma~\underbrace{R_{\text{stability}}(\pi;\pi_0)}_{\text{stability}}\;-\;\mu~\underbrace{\widetilde{Q}(\pi)}_{\text{modularity}}
$$

* Set $\alpha, \beta, \gamma, \mu$ through hardware calibration (§6).
* Intuition: **Shorter distances + better alignment + greater stability + stronger communities ⇒ better memory locality ⇒ higher L2 hit ⇒ lower latency/EpT** (I/O-aware principle).【F:docs/Cost_Spec.md†L82-L87】

---

## 3) Component Definitions

### 3.1 Cache-distance cost $C(\pi)$

* Position function: $pos_\pi(i)\in\{0,\dots,D-1\}$.
* **Grouping unit** $g$: Number of elements per cache line/warp-coalesced tile (e.g., line bytes ÷ element size).
* **Distance kernel** $\phi(d)$ (Huber/saturating):

  $$
  d_{ij}=\Big\lfloor\frac{|pos_\pi(i)-pos_\pi(j)|}{g}\Big\rfloor,\quad
  \phi(d)=\begin{cases}
  d, & d\le \lambda\\
  \lambda + \tau(d-\lambda), & d>\lambda,\quad 0<\tau<1
  \end{cases}
  $$

  $$
  C(\pi)=\sum_{i<j} W_{ij}\,\phi(d_{ij})
  $$

  — Interpretation: Keeping entries within the same line/tile reduces cost; long-distance penalties saturate once everything is equally bad.
* **Normalization**: $\widetilde{C}(\pi)=C(\pi)/C_{\max}$ with $C_{\max}=\phi(d_{\max})\sum_{i<j}W_{ij}$ and $d_{\max}=\lfloor (D-1)/g\rfloor$.

> L2 aggregates by 32B sector/line, so **higher hit rates reduce downstream memory accesses**. $\phi$ approximates the sector/line reality.【F:docs/Cost_Spec.md†L89-L105】

### 3.2 Modularity $Q(\pi)$ (weighted network, resolution $\gamma$)

* Degree $k_i=\sum_j W_{ij}$ and $2m=\sum_{i}k_i$.
* **Blocking rule**: Slice $\pi$ into $k$ contiguous blocks $B_1,\dots,B_k$ (fixed length or heuristic).
* **Definition**:

  $$
  Q_\gamma(\pi)=\frac{1}{2m}\sum_{i,j}\left(W_{ij}-\gamma\frac{k_i k_j}{2m}\right)\mathbf{1}\{i,j\in B_c\ \text{for some }c\}
  $$

  — $\gamma=1$ is standard; $\gamma>1$ increases sensitivity to small communities, reducing the resolution limit.
* **Normalization**: Approximate $\widetilde{Q}=(Q_\gamma-\mathbb{E}_{\text{null}}[Q_\gamma])/(Q_{\max}-\mathbb{E}_{\text{null}}[Q_\gamma])$.【F:docs/Cost_Spec.md†L107-L116】

### 3.3 Alignment / Vectorization regularizer $R_{\text{align}}$

* **Vector width** $v\in\{8,16,32\}$ (elements), **alignment** $a$ bytes.
* Simple approximation:

  $$
  R_{\text{align}}(\pi)=\frac{1}{\sum_{i<j} W_{ij}}\sum_{i<j}W_{ij}\cdot \mathbf{1}\{pos_\pi(i)\bmod v \ne pos_\pi(j)\bmod v\}
  $$

  — Within a tile, having the same modulo class improves coalescing.【F:docs/Cost_Spec.md†L118-L124】

### 3.4 Stability regularizer $R_{\text{stability}}$

* **Suppress churn** (layout rebuild cost / cache regeneration):

  $$
  R_{\text{stability}}(\pi;\pi_0)=\frac{1}{D}\sum_{i} \mathbf{1}\{|pos_\pi(i)-pos_{\pi_0}(i)|>h\}
  $$

  — Provide a hysteresis threshold $h$.【F:docs/Cost_Spec.md†L126-L133】

---

## 4) Constraints, Convergence, Numerical Guards

* **Constraints**: Enforce legal shape/dtype/stride (guaranteed by Pass T3).
* **Stopping**: Terminate when $\Delta J < \varepsilon$ or the time budget expires (default 300 s). If no improvement, roll back to $\pi_0$.
* **Determinism**: Same seed/clock ⇒ same $\pi$ (±ε).
* **Complexity**: $C$ runs in $O(\mathrm{nnz}(W))$; $Q$ uses block boundary accumulation $O(\mathrm{nnz}+kD)$.【F:docs/Cost_Spec.md†L135-L142】

---

## 5) Tuning Parameters (Defaults & Guidance)

* $g$ (elements per line/tile): **Derive from hardware** (e.g., 128B line with fp16 ⇒ $g=64$).
* $\lambda, \tau$: Place $\lambda$ near the **tile boundary** (2–4), choose $\tau\in[0.1,0.3]$.
* $\gamma$: 1 (default) → 1.5 (highlight smaller communities).
* Weights: fix $\alpha=1$, start with $\mu≈0.5$, $\beta\in[0.1,0.3]$, $\gamma\in[0.0,0.2]$ (refine after calibration).【F:docs/Cost_Spec.md†L144-L151】

---

## 6) Calibration: Linking Metrics and Objective

**Goal**: Estimate how changes in $\Delta \widetilde{C}$, $\Delta \widetilde{Q}$, and the regularizers explain **L2 hit / latency / EpT** improvements.

1. Run a **pilot set** of about $N≈20$ configurations (random, spectral, local-search combinations).
2. Fit **ridge regressions**:

   $$
   \Delta \text{L2} = a_0 + a_1(-\Delta \widetilde{C}) + a_2(\Delta \widetilde{Q}) + a_3(-\Delta R_{\text{align}}) + \varepsilon
   $$

   $$
   \Delta \text{Latency} = b_0 + b_1(-\Delta \widetilde{C}) + b_2(\Delta \text{L2}) + \varepsilon
   $$

   $$
   \Delta \text{EpT} = c_0 + c_1(\Delta \text{Latency}) + c_2(\Delta \text{L2}) + \varepsilon
   $$
3. **Back-solve weights**: Rescale $(\alpha,\beta,\mu)$ proportional to regression coefficients (linear/Bayesian update).
4. **Validate**: Check cross-validated $R^2$ and residuals.
5. **Operational use**: Monitor whether $\Delta J$ matches the expected $\Delta \text{L2} / \Delta \text{Latency}$ trends during the pipeline.

> Definitions of L2 hit/sector/throughput follow the Nsight Compute guide.【F:docs/Cost_Spec.md†L153-L172】  
> The I/O-aware perspective follows FlashAttention (minimizing memory hierarchy traffic).【F:docs/Cost_Spec.md†L153-L172】  
> General scheduling/cost-model formulations follow Halide/TVM Ansor literature.【F:docs/Cost_Spec.md†L153-L172】

---

## 7) Implementation Spec (Contracts)

### 7.1 API Mapping (per ICD)

* `fit_permutation(W, ...) -> (pi, stats)` must compute **$C$ and $Q$**, returning `stats={"C":…, "Q":…, "J":…, "improved":…}`.
* Persist $\alpha, \beta, \gamma, \mu, g, \lambda, \tau, \gamma$ (modularity) under `solver.cfg`.

### 7.2 Numerical Stability

* $W$ normalization options: `"none" | "row" | "sym"`; default `sym`: $W\leftarrow D^{-1/2}WD^{-1/2}$.
* Avoid zero/NaN frequencies by adding $\epsilon=10^{-9}$.
* For sparse $W$, store only the upper triangle in CSR and accumulate accordingly.【F:docs/Cost_Spec.md†L174-L188】

---

## 8) Unit Tests (Required)

1. **Monotonicity (block structure)**

   * Synthetic 2-block $W$: increasing intra-block cohesion should yield $\widetilde{C}\downarrow$ and $\widetilde{Q}\uparrow$.
2. **Scale invariance**

   * Scaling $W\leftarrow aW$ ($a>0$) must preserve $\arg\min J$ and keep $\widetilde{C},\widetilde{Q}$ within the same range.
3. **Line/tile sensitivity**

   * Increasing $g$ should reduce $C$ for the same $\pi$ (larger tile → relaxed penalties).
4. **Determinism**

   * Identical seed/time_budget → identical permutation hash.
5. **Calibration consistency**

   * Pilot regression should achieve $R^2_{\text{L2}}>0.6$; otherwise re-run parameter grid search.
6. **IR legality linkage**

   * After layout changes, Pass T3 must report zero stride/alignment violations (snapshot check).【F:docs/Cost_Spec.md†L190-L204】

---

## 9) Performance Notes / Risks / Alternatives

* **Complexity**: Evaluating $C,Q$ is approximately $O(\mathrm{nnz}(W))$; solver cost (spectral + local search) dominates.
* **Risk**: Modularity resolution limits → sweep $\gamma$ / use multiple block rules.【F:docs/Cost_Spec.md†L206-L212】
* **Risk**: Excessive alignment penalties reduce permutation diversity → cap $\beta$, use warm restarts.
* **Alternative**: Detect blocks via Louvain/Leiden first (contiguous placement), then fine-tune the 1D permutation (hybrid).【F:docs/Cost_Spec.md†L206-L212】

---

## 10) Recommended Initial Parameters

```yaml
cost:
  alpha: 1.0
  beta: 0.2
  gamma_stability: 0.1
  mu: 0.5
  g: 64          # elements per line (e.g., 128B line, fp16)
  lambda: 3
  tau: 0.25
  modularity_gamma: 1.2
  blocks_k: 4    # heuristic number of contiguous blocks for Q
```

---

## 11) Pseudocode (Evaluation Routine)

```python
def eval_cost(W, pi, pi_prev, cfg):
    pos = invperm(pi)                         # O(D)
    g = cfg.g
    C = 0.0
    for (i, j, wij) in upper_triangle_nnz(W): # CSR iterate
        dij = abs(pos[i] - pos[j]) // g
        phi = dij if dij <= cfg.lambda else cfg.lambda + cfg.tau * (dij - cfg.lambda)
        C += wij * phi
    C_norm = C / (cfg.C_max or (phi_max(cfg, D, g) * sum_nnz(W)))

    Q = modularity_contiguous_blocks(W, pi, k=cfg.blocks_k, gamma=cfg.modularity_gamma)
    Q_norm = normalize_modularity(Q)          # null-model baseline approx.

    R_align = align_penalty(W, pos, v=cfg.vec_width)
    R_stab  = stability_penalty(pi, pi_prev, h=cfg.hysteresis)

    J = cfg.alpha * C_norm + cfg.beta * R_align + cfg.gamma_stability * R_stab - cfg.mu * Q_norm
    return J, {"C":C_norm, "Q":Q_norm, "R_align":R_align, "R_stab":R_stab, "J":J}
```

---

## 12) Acceptance Criteria (Cost Spec Perspective)

* Pass all unit/integration tests (§8).
* After pilot calibration, observe that **$\Delta \widetilde{C}\downarrow$ and $\Delta \widetilde{Q}\uparrow$** correlate positively with **L2 hit improvements and latency/EpT reductions** (p<0.05).
* When applied in the pass pipeline, report **zero legalization failures**, respect determinism, and stay within the time budget.【F:docs/Cost_Spec.md†L214-L224】

---

## 13) References (Key Sources)

* **Modularity definition / spectral approach**: Newman, 2006, *PNAS*.【F:docs/Cost_Spec.md†L226-L231】
* **Resolution limit & $\gamma$**: Fortunato & Barthélemy, 2007, *PNAS*; RB/Leiden literature.【F:docs/Cost_Spec.md†L226-L231】
* **I/O-aware principle**: FlashAttention, 2022.【F:docs/Cost_Spec.md†L226-L231】
* **Scheduling/cost-model tradition**: Halide (PLDI 2013), TVM Ansor (OSDI 2020).【F:docs/Cost_Spec.md†L226-L231】
* **L2 metric definitions**: NVIDIA Nsight Compute guide.【F:docs/Cost_Spec.md†L226-L231】

---

[1]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/?utm_source=chatgpt.com "2. Profiling Guide — NsightCompute 13.0 documentation"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC1482622/?utm_source=chatgpt.com "Modularity and community structure in networks - PMC"
[3]: https://pubmed.ncbi.nlm.nih.gov/16723398/?utm_source=chatgpt.com "Modularity and community structure in networks - PubMed"
[4]: https://www.pnas.org/doi/full/10.1073/pnas.0605965104?utm_source=chatgpt.com "Resolution limit in community detection | PNAS"
[5]: https://en.wikipedia.org/wiki/Leiden_algorithm?utm_source=chatgpt.com "Leiden algorithm"
[6]: https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com "[2205.14135] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[7]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html?utm_source=chatgpt.com "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[8]: https://dl.acm.org/doi/abs/10.1145/2499370.2462176?utm_source=chatgpt.com "Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines: ACM SIGPLAN Notices: Vol 48, No 6"
[9]: https://dspace.mit.edu/handle/1721.1/85943?utm_source=chatgpt.com "Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines"
[10]: https://www.usenix.org/conference/osdi20/presentation/zheng?utm_source=chatgpt.com "Ansor: Generating High-Performance Tensor Programs for Deep Learning | USENIX"
[11]: https://ar5iv.labs.arxiv.org/html/2006.06762v4?utm_source=chatgpt.com "[2006.06762] Ansor: Generating High-Performance Tensor Programs for Deep Learning"
[12]: https://www.researchgate.net/publication/1913681_Fast_Unfolding_of_Communities_in_Large_Networks?utm_source=chatgpt.com "(PDF) Fast Unfolding of Communities in Large Networks"
[13]: https://en.wikipedia.org/wiki/Louvain_method?utm_source=chatgpt.com "Louvain method"
