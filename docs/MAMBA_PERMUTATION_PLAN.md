# Mamba Permutation Support Plan

## Problem Statement

The codebase needs to support permutation application for both:
1. **Original mamba-ssm** (has A, B, C modules) ✅ Works
2. **HuggingFace Transformers Mamba** (has different structure) ❌ Broken

## Current Situation

### What Works (Original mamba-ssm)
```python
# Original structure (from mamba_ssm.models.mixer_seq_simple)
class Mamba(nn.Module):
    def __init__(...):
        self.A = nn.Parameter(...)  # (d_inner, d_state)
        self.B = nn.Linear(...)      # Projects to B matrix
        self.C = nn.Linear(...)      # Projects to C matrix
```

**Current permutation logic (apply_pi.py:422-471):**
- A.weight: `PWP_inv(A, pi, pinv)` - hidden→hidden transformation
- B.weight: `reindex_rows(B, pi)` - row reindexing
- C.weight: `reindex_cols(C, pi)` - column reindexing

### What's Broken (HuggingFace Mamba)

**HuggingFace structure (from transformers.models.mamba.MambaMixer):**
```python
class MambaMixer(nn.Module):
    def __init__(self, config, layer_idx):
        # Core SSM parameters
        self.A_log = nn.Parameter(...)  # (intermediate_size, state_size)
        self.D = nn.Parameter(...)      # (intermediate_size,)

        # Projections
        self.in_proj = nn.Linear(hidden_size, intermediate_size * 2)
        self.conv1d = nn.Conv1d(intermediate_size, intermediate_size, ...)
        self.x_proj = nn.Linear(intermediate_size, dt_rank + state_size * 2)
        self.dt_proj = nn.Linear(dt_rank, intermediate_size)
        self.out_proj = nn.Linear(intermediate_size, hidden_size)
```

**Key differences:**
1. No separate A, B, C modules
2. A_log is a parameter, not a module
3. B and C are computed dynamically from x_proj output
4. Different dimension names: hidden_size (d_model) vs intermediate_size (d_inner)

## Architecture Analysis

### Dimension Flow in HuggingFace Mamba

```
Input: (batch, seq_len, hidden_size)
  ↓
in_proj: hidden_size → intermediate_size * 2
  ↓ (split into xz and xz_residual)
conv1d: intermediate_size → intermediate_size
  ↓
x_proj: intermediate_size → (dt_rank + state_size * 2)
  ↓ (split into dt, B, C)
dt_proj: dt_rank → intermediate_size
  ↓
SSM computation with A_log, B, C, D, dt
  ↓
out_proj: intermediate_size → hidden_size
  ↓
Output: (batch, seq_len, hidden_size)
```

### What Dimension Should Be Permuted?

**Option 1: Permute `hidden_size` (d_model)**
- This is the external interface dimension
- Consistent across the entire model
- **Problem:** intermediate_size = 2 * hidden_size (expansion), how to map?

**Option 2: Permute `intermediate_size` (d_inner)**
- This is where SSM computation happens
- A_log, D operate on this dimension
- **Problem:** Need to create pi_inner from pi_hidden

**Option 3: Permute both separately**
- Most flexible but complex
- Need two permutations: pi_hidden and pi_inner
- **Problem:** How to relate them?

## Critical Questions

### Q1: What does the original permutation optimize?

Looking at the paper and code:
- Optimizes **memory access patterns** of state dimensions
- Modularity metric groups co-accessed dimensions
- Goal: reduce cache misses during SSM computation

**Answer:** The permutation targets the SSM state dimension, which is `intermediate_size` in HF Mamba.

### Q2: How does permutation propagate through projections?

If we permute intermediate_size (d_inner), how does it affect layers?

```
in_proj: d_model → d_inner
  - Weight shape: (d_inner, d_model)
  - After pi_inner permutation: reindex_rows(weight, pi_inner)

x_proj: d_inner → (dt_rank + 2*d_state)
  - Weight shape: (dt_rank + 2*d_state, d_inner)
  - After pi_inner permutation: reindex_cols(weight, pi_inner)

out_proj: d_inner → d_model
  - Weight shape: (d_model, d_inner)
  - After pi_inner permutation: reindex_cols(weight, pi_inner)
```

### Q3: How to create pi_inner from pi_hidden?

**Assumption:** intermediate_size = expansion_factor * hidden_size (typically 2x)

**Option A: Repeat and scale**
```python
pi_inner = torch.cat([pi_hidden * expansion_factor + i
                       for i in range(expansion_factor)])
# E.g., pi_hidden = [2,0,1] → pi_inner = [4,0,2, 5,1,3] for expansion=2
```

**Option B: Block expansion**
```python
pi_inner = torch.repeat_interleave(pi_hidden, expansion_factor) * expansion_factor
pi_inner += torch.arange(expansion_factor).repeat(len(pi_hidden))
# E.g., pi_hidden = [2,0,1] → pi_inner = [4,5, 0,1, 2,3] for expansion=2
```

**Option C: Independent optimization**
```python
# Run solver on intermediate dimension directly
# Problem: Correlation matrix would be on d_inner, not d_model
```

## Proposed Solution

### Phase 1: Minimal Support (Quick Fix)

**Assumption:** The graph is built on `hidden_size`, so pi operates on d_model.

**Implementation:**
1. Detect HuggingFace Mamba (check for `in_proj`, `out_proj`, `A_log`)
2. Expand pi_hidden → pi_inner using block expansion (Option B)
3. Apply transformations:
   - `in_proj.weight`: `reindex_rows(weight, pi_inner)`
   - `x_proj.weight`: `reindex_cols(weight, pi_inner)`
   - `dt_proj.weight`: No change (operates on dt_rank, not d_inner)
   - `out_proj.weight`: `reindex_cols(weight, pi_inner)` + `reindex_rows(weight, pi_hidden)`
   - `A_log`: `reindex_rows(A_log, pi_inner)`
   - `D`: `reindex_vec(D, pi_inner)`
   - `conv1d.weight`: `reindex_rows(weight, pi_inner)` (channels dimension)

### Phase 2: Proper Support (Future Work)

1. Build correlation matrix on `intermediate_size` directly
2. Optimize permutation for d_inner (where SSM happens)
3. Handle in_proj/out_proj as interface layers
4. More accurate modeling of Mamba's actual data flow

## Implementation Checklist

### Step 1: Update Module Detection (runners_hf.py) ✅
- [x] Already detects HF Mamba (in_proj, x_proj, out_proj)
- [x] Store additional metadata: `_hf_mamba=True`, `_module_ref`

### Step 2: Create apply_pi_to_mamba_hf() (apply_pi.py) ✅
- [x] New function: `apply_pi_to_mamba_hf(module_dict, pi)`
- [x] Helper: `_expand_permutation_for_intermediate(pi, expansion_factor)`
- [x] Expand pi_hidden → pi_inner with block expansion
- [x] Apply transformations to each layer (in_proj, x_proj, out_proj)
- [x] Handle A_log and D parameters
- [x] Handle conv1d layer

### Step 3: Update Dispatch Logic (runners_hf.py) ✅
- [x] Check for `_hf_mamba` flag in module_dict
- [x] Call `apply_pi_to_mamba_hf` instead of `apply_pi_to_mamba`
- [x] Import and export new function

### Step 4: Testing ⏳
- [ ] Test on mamba-130m-hf (small, fast)
- [ ] Verify no "no applicable Mamba modules" error
- [ ] Check output shapes remain correct
- [ ] Validate GPU memory usage is reasonable
- [ ] Run short experiment to completion

### Step 5: Validation
- [ ] Compare results: original mamba-ssm vs HF Mamba
- [ ] Ensure both give similar relative improvements
- [ ] Document any differences in behavior

## Risk Assessment

### High Risk
- **Wrong transformations:** If dimension mapping is incorrect, model will produce garbage
- **Shape mismatches:** Could cause crashes during forward pass

### Medium Risk
- **Performance differences:** HF Mamba vs original might have different cache behavior
- **Expansion factor assumptions:** Code assumes 2x, but might be configurable

### Low Risk
- **Backward compatibility:** Changes are additive, won't break original mamba-ssm

## Testing Strategy

### Unit Test
```python
def test_hf_mamba_permutation():
    from transformers import MambaForCausalLM
    model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

    # Apply permutation
    pi = torch.randperm(model.config.hidden_size)
    apply_pi_to_mamba_hf(module_dict, pi)

    # Check shapes
    assert model(inputs).shape == original_shape

    # Check no NaN/Inf
    assert torch.isfinite(model(inputs).logits).all()
```

### Integration Test
```bash
# Run small experiment with HF Mamba
python -m icd.cli.main run -c configs/mamba.json --out /tmp/test_hf
```

## Decision Point

**Recommendation:** Implement Phase 1 (Minimal Support) now.

**Rationale:**
1. Gets experiments running quickly
2. Can validate approach with small model first
3. Phase 2 requires more research and testing
4. Phase 1 is good enough for paper experiments

**Time Estimate:**
- Phase 1: 2-3 hours (implementation + testing)
- Phase 2: 1-2 days (research + proper implementation)

## Next Steps

1. ✅ Create this plan document
2. ✅ Review with user, get approval
3. ✅ Implement Phase 1
4. ⏳ Test with mamba-130m-hf (ready for RunPod testing)
5. ⏳ Test with mamba-2.8b-hf (ready for RunPod testing)
6. ✅ Update configs to use HF models (mamba_3b.json already uses HF)
7. ⏳ Document usage in experiments/README.md

---

**Status:** PHASE 1 IMPLEMENTATION COMPLETE - READY FOR TESTING
**Last Updated:** 2025-09-30
**Author:** Claude Code

## Implementation Summary

Phase 1 has been successfully implemented with the following changes:

1. **icd/runtime/apply_pi.py**:
   - Added `_expand_permutation_for_intermediate()` helper function
   - Added `apply_pi_to_mamba_hf()` function for HF Mamba permutation
   - Handles dimension expansion (hidden_size → intermediate_size)
   - Applies permutations to in_proj, x_proj, out_proj, A_log, D, conv1d

2. **icd/runtime/runners_hf.py**:
   - Updated `_collect_mamba_modules_from_model()` to store `_module_ref`
   - Updated dispatch logic in `_apply_pi_sequence()` to check `_hf_mamba` flag
   - Imports and calls `apply_pi_to_mamba_hf()` for HF Mamba models

3. **Configs**:
   - `mamba_3b.json` already uses HF model: `state-spaces/mamba-2.8b-hf`
   - `mamba_ssm_2.8b.json` uses original mamba-ssm: `state-spaces/mamba-2.8b`
   - Both configs are ready for experiments

The implementation is ready for testing on RunPod A100 GPU.