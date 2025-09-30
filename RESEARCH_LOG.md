# Research Log: Iterative Co-Design Experimental Validation

## Overview
This log tracks the first real experimental validation of the iterative co-design hypothesis. Paper currently contains mock data - these experiments will provide actual empirical evidence.

---

## 2025-09-30: Critical Bug Fix & Restart

### Morning: Discovery
- Found critical bug: Mamba permutations were being rejected
- Error: `len(perm)=4096, expect hs=2560 inter=5120`
- Root cause: `_maybe_override_feature_dim_from_config` had flawed logic
- All Mamba experiments from previous runs were invalid (permutations never applied)

### Afternoon: Deep Analysis
- Conducted comprehensive codebase analysis
- Identified conceptual gaps between paper claims and implementation:
  - Graph W doesn't actually measure memory access patterns (heuristic proxy)
  - "Iterative" is more 2-stage pipeline than true iteration
  - No hardware profiling feedback into graph construction
  - Unclear if approach generalizes across architectures

### Fix Implementation
- **Commit:** `9b57362` - "fix: Force HuggingFace config.hidden_size for graph dimensions"
- **Change:** Always use `config.hidden_size` instead of FX-inferred dimensions
- **Impact:** Mamba graphs now built with D=2560 (correct) instead of D=4096 (wrong)
- **Test:** Created `test_mamba_fix.py` - all tests passing ✓

### Evening: Experiment Restart
- Stopped running experiments (were using broken code)
- Restarting with fixed code
- Created minimal smoke test config: `configs/mamba_smoke_test.json`

### Smoke Test Parameters
```json
{
  "repeats": 100,        // Fast iteration (vs 500-600 in full)
  "warmup_iter": 20,     // Minimal warmup
  "sequence_length": 512, // Shorter than full (2048)
  "correlation": {
    "enable": true,      // Use real data
    "samples": 8         // Fewer samples for speed
  },
  "solver": {
    "time_budget_s": 60  // Quick solve
  },
  "pair": {
    "baseline_methods": ["dense", "iterative"]  // Just 2 baselines
  }
}
```

### What to Watch For
✅ **Success indicators:**
- Log shows: `[IASP] applying perm len=2560 to mamba expect(hs)=2560`
- NO errors: `permutation rejected`
- Metrics differ between dense and iterative
- Any improvement in latency (even 1%)

❌ **Failure indicators:**
- Still seeing `permutation rejected` errors
- Identical metrics for dense vs iterative
- Dimension mismatch errors
- Iterative worse than dense

### Hypotheses to Test
- **H1:** Dimension fix enables successful permutation application
- **H2:** Graph with correlation captures meaningful co-access patterns
- **H3:** Modularity optimization improves cache locality
- **H4:** Approach provides measurable latency improvements

---

## [Next Entry - Date TBD]

### Smoke Test Results
**Status:** [PENDING]

**Observations:**
- [ ] Permutation application:
- [ ] Dimension used:
- [ ] Modularity scores:
- [ ] Latency comparison:

**Key Metrics:**
```
Dense Baseline:
  - Latency: TBD ms
  - Modularity: TBD
  - Cache hit rate: TBD%

Iterative:
  - Latency: TBD ms (X% improvement)
  - Modularity: TBD
  - Cache hit rate: TBD%
```

**Decision:**
- [ ] Scale up to full experiments
- [ ] Debug and iterate
- [ ] Pivot approach

**Notes:**
[TBD]

---

## Experimental Tracking

### Smoke Tests Completed
- [ ] Mamba-2.8B (minimal)
- [ ] BERT-large (if Mamba succeeds)
- [ ] ResNet-50 (if BERT succeeds)

### Full Experiments Completed
- [ ] Mamba-2.8B (5 runs × 4 baselines)
- [ ] BERT-large (5 runs × 4 baselines)
- [ ] ResNet-50 (5 runs × 4 baselines)
- [ ] GCN (if others succeed)
- [ ] EfficientNet (if others succeed)
- [ ] GraphSAGE (if others succeed)

### Analysis Completed
- [ ] Statistical significance tests
- [ ] Effect size calculations (Cohen's d)
- [ ] Correlation analysis (modularity ↔ cache hits)
- [ ] Ablation studies (correlation on/off)

---

## Key Insights (Running List)

### What We Know
1. Infrastructure is solid (solvers, measurement, profiling all work)
2. Permutation math is correct (PWP, Pinv_W_P verified)
3. HDS/sparsity integration functional (Gumbel-TopK working)
4. Bug prevented any real Mamba validation until now

### What We're Testing
1. Does dimension fix enable permutation application?
2. Does graph capture meaningful patterns?
3. Does modularity correlate with cache performance?
4. Do improvements generalize across architectures?

### What We Don't Know
1. If approach produces claimed 15-25% improvements
2. If correlation is necessary or heuristic graphs work
3. If mechanism is modularity→cache or something else
4. If CNN/GNN architectures benefit from this approach

---

## Future Directions (If Current Approach Fails)

### Pivot Option A: Profiling-Based Graphs
- Use NCU to measure actual memory access patterns
- Build graph W from profiled co-access (not heuristics)
- Direct hardware feedback

### Pivot Option B: Architecture-Specific Methods
- Transformers: Current approach (permute hidden dims)
- SSMs: Permute state space dimensions
- CNNs: Different layout optimization (channel ordering?)
- GNNs: Graph-specific permutations

### Pivot Option C: Learnable Layouts
- Use hardware metrics as training signal
- Differentiable permutations (Gumbel-Sinkhorn)
- Learn optimal layouts from data

---

## Questions for Investigation

### Technical Questions
1. Does correlation.enable make a significant difference?
2. Do different solvers (Louvain vs spectral) produce similar results?
3. Is there a correlation between graph modularity and solver modularity?
4. Does permutation actually change weight matrix memory layout?

### Conceptual Questions
1. What does the graph W actually capture?
2. Is spatial adjacency assumption valid for neural networks?
3. Can permutation preserve model semantics for all architectures?
4. Is modularity the right objective or should we optimize cache hits directly?

### Methodological Questions
1. How many runs needed for statistical significance?
2. What's the minimal detectable improvement?
3. How do we attribute improvements (causality)?
4. What controls are needed to rule out confounds?

---

## Contact & Collaboration Notes
[Space for collaboration discussions, feedback from reviewers, etc.]

---

*Last Updated: 2025-09-30*