# Experiment Quick Start Guide

## 🚀 Running the Smoke Test

### In Docker Container

```bash
# 1. Ensure you have latest code
cd /workspace/iterative-co-design
git pull
git log --oneline -3  # Should show commit 9b57362 (dimension fix)

# 2. Activate environment (if using venv)
source venv/bin/activate

# 3. Run smoke test
python -m icd.cli.main run configs/mamba_smoke_test.json

# 4. Monitor in real-time (in another terminal)
tail -f runs/mamba_smoke_test/run.log
```

---

## 👀 What to Look For in Logs

### ✅ SUCCESS Indicators

**Permutation Application:**
```
[IASP] applying perm len=2560 to mamba expect(hs)=2560 expect(inter)=5120
```
- `len=2560` matches `expect(hs)=2560` ← **This is the key!**
- No "skip" or "rejected" messages

**Graph Construction:**
```
[INFO] Graph built: D=2560, source=pytorch, nnz=...
[INFO] feature_dim_source: hf_config.hidden_size
```
- `D=2560` (not 4096!)
- Source confirms using HF config

**Solver Output:**
```
[INFO] Solver: louvain, time=X.Xs, modularity=0.XX
[INFO] Permutation computed: length=2560
```
- Modularity score > 0.5 is good
- Modularity > 0.7 is excellent

**Baseline Comparison:**
```
[dense] latency_mean: 24.3 ms
[iterative] latency_mean: 21.8 ms  ← 10% improvement!
```
- **Any** improvement validates the fix worked
- 5-10% is encouraging
- 15%+ is excellent

---

### ❌ FAILURE Indicators

**Permutation Rejection (OLD BUG):**
```
[IASP] skip backbone.layers.X.mixer (len(perm)=4096, expect hs=2560 inter=5120)
Mamba permutation rejected for all 64 collected modules
```
- If you see this → still using old code or new bug
- **ACTION:** Stop immediately, debug

**Wrong Dimension:**
```
[INFO] Graph built: D=4096
```
- Should be 2560 for Mamba
- **ACTION:** Config override failed, check logs

**Identical Results:**
```
[dense] latency_mean: 24.3 ms
[iterative] latency_mean: 24.3 ms  ← Exactly the same!
```
- Permutation might not be applied
- Or identity permutation (no change)
- **ACTION:** Check permutation stats in output

**Degradation:**
```
[dense] latency_mean: 24.3 ms
[iterative] latency_mean: 26.7 ms  ← 10% WORSE!
```
- Graph or solver produced bad permutation
- **ACTION:** Investigate graph quality

---

## 📊 Analyzing Results

### Quick Analysis Script

```bash
# After experiment completes
cd runs/mamba_smoke_test

# Check if permutation was applied
grep "applying perm" run.log

# Check dimensions used
grep "feature_dim" run.log

# Compare latencies
grep "latency_mean" metrics.json

# Check modularity
grep "modularity" metrics.json
```

### Manual Inspection

```python
import json

# Load metrics
with open('runs/mamba_smoke_test/metrics.json') as f:
    metrics = json.load(f)

# Compare baselines
dense = metrics['dense']
iterative = metrics['iterative']

improvement = (dense['latency_mean'] - iterative['latency_mean']) / dense['latency_mean']
print(f"Improvement: {improvement*100:.1f}%")

# Check modularity
print(f"Modularity: {iterative.get('modularity', 'N/A')}")

# Power consumption
if 'energy_per_token_j' in iterative:
    energy_improvement = (dense['energy_per_token_j'] - iterative['energy_per_token_j']) / dense['energy_per_token_j']
    print(f"Energy improvement: {energy_improvement*100:.1f}%")
```

---

## 🎯 Decision Tree

### After Smoke Test Completes

```
┌─ Permutation Applied Successfully?
│
├─ YES ──┬─ Improvement > 0%?
│        │
│        ├─ YES ──┬─ Improvement > 5%?
│        │        │
│        │        ├─ YES → ✅ EXCELLENT! Scale up to full experiments
│        │        │
│        │        └─ NO → ⚠️  Marginal. Run longer test (300 iterations)
│        │
│        └─ NO → ❌ Investigate:
│                  - Check if identity permutation
│                  - Analyze graph quality
│                  - Test correlation on/off
│
└─ NO → 🔴 CRITICAL:
         - Check code version (should be 9b57362+)
         - Inspect permutation application code
         - Look for new dimension mismatch errors
```

---

## 🔬 Debugging Commands

### Check Permutation Stats
```bash
# From run directory
python3 << 'EOF'
import json
with open('perm_before.json') as f:
    perm = json.load(f)
    pi = perm['pi']

print(f"Permutation length: {len(pi)}")
print(f"Identity? {pi == list(range(len(pi)))}")
print(f"First 10 elements: {pi[:10]}")
print(f"Modularity: {perm.get('stats', {}).get('modularity', 'N/A')}")
EOF
```

### Visualize Graph (for small models)
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import mmread

# Load graph
W = mmread('w.npz.mtx').toarray()

# Visualize
plt.figure(figsize=(10, 10))
plt.imshow(W, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Graph W (Co-access Matrix)')
plt.savefig('graph_viz.png')
```

### Check Model Weights Changed
```python
import torch

# Load model
model_before = torch.load('model_before.pt')
model_after = torch.load('model_after.pt')

# Compare first layer
layer_before = model_before.backbone.layers[0].mixer.in_proj.weight
layer_after = model_after.backbone.layers[0].mixer.in_proj.weight

# Should be different if permutation applied
print(f"Weights identical: {torch.equal(layer_before, layer_after)}")
print(f"Max difference: {(layer_before - layer_after).abs().max()}")
```

---

## 📈 Scaling Up to Full Experiments

### If Smoke Test Succeeds

```bash
# Run full Mamba experiments
for run in {1..5}; do
    for baseline in dense algo_only linear iterative; do
        python -m icd.cli.main run \
            configs/mamba_3b.json \
            --override "report.out_dir=runs/mamba_full/run_${run}/${baseline}"
    done
done
```

### Monitoring Long-Running Experiments

```bash
# In separate terminal
watch -n 10 'tail -20 runs/mamba_full/run_1/dense/run.log'

# Check progress
find runs/mamba_full -name "metrics.json" | wc -l  # Count completed runs
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: CUDA Out of Memory
**Symptom:** RuntimeError: CUDA out of memory
**Solution:**
- Reduce batch_size in config
- Reduce sequence_length
- Use smaller model variant

### Issue 2: NCU Profiling Fails
**Symptom:** NCU binary not found
**Solution:**
- Set `ncu_enable: false` in smoke test
- NCU requires specific CUDA toolkit version
- Profiling optional for initial validation

### Issue 3: Experiments Stuck
**Symptom:** No progress for > 10 minutes
**Solution:**
- Check GPU utilization: `nvidia-smi`
- Kill and restart if hung
- Increase timeout in config

### Issue 4: Results Not Reproducible
**Symptom:** Different runs give different results
**Solution:**
- Check if `rng_seed` is set in config
- Verify `fixed_clock: true` for deterministic timing
- Run more iterations for stable averages

---

## 🎓 Interpreting Results for Paper

### Minimal Success (Publishable)
- ✅ Permutation applies successfully
- ✅ Any measurable improvement (even 2-5%)
- ✅ Understand mechanism (even if different than hypothesized)
- Paper focus: "Proof of concept, architectural considerations"

### Moderate Success (Strong Paper)
- ✅ 5-15% improvement across 2+ architectures
- ✅ Modularity correlates with performance
- ✅ Correlation matters (vs heuristic graphs)
- Paper focus: "Iterative co-design for specific architectures"

### Full Success (Top Venue)
- ✅ 15-25% improvement across 4+ architectures
- ✅ Strong mechanistic validation (modularity → cache → latency)
- ✅ Statistical significance (p < 0.01, d > 1.0)
- Paper focus: "Orthogonality fallacy, general co-design principle"

---

## 📝 Data Collection Checklist

For each experiment run:
- [ ] Config file snapshot saved
- [ ] Full logs captured (run.log)
- [ ] Metrics JSON generated
- [ ] Permutation files saved (perm_before.json, perm_after.json)
- [ ] Graph files saved (w.npz)
- [ ] Hardware metrics (if NCU/NVML enabled)
- [ ] Error/warning analysis
- [ ] Research log updated

---

*Quick Reference - Keep this handy while running experiments!*