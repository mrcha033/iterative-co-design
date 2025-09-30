# Fix: Correlation Graph Dimension Mismatch (4096 vs 2560)

## Problem

When running Mamba experiments with correlation-based graph refinement enabled, permutation application was failing with:

```
[IASP] skip backbone.layers.X.mixer (len(perm)=4096, expect hs=2560 inter=5120):
permutation length 4096 does not match hidden_size 2560 or intermediate_size 5120
```

**Root Cause**: The correlation graph collection was inferring dimensions from layer activations by flattening tensors, which produced wrong dimensions for models like Mamba where:
- Hidden layers output `(batch, seq, 2560)`
- Intermediate/conv layers output different shapes that flatten to 4096 or 5120
- The flattening logic `tensor.reshape(tensor.shape[0], -1)` would turn `(B, 512, 2560)` into `(B, 512*2560)` - completely wrong!

## The Fix

### 1. Proper 3D Tensor Handling (icd/graph/correlation.py)

**Before (BUGGY)**:
```python
tensor = tensor.reshape(tensor.shape[0], -1)  # Flattens everything!
stats = self._get_stats(name, tensor.shape[1], tensor.device)
```

**After (FIXED)**:
```python
# Handle 3D tensors properly
if tensor.ndim == 3:
    feature_dim = tensor.shape[2]  # Use last dimension
    tensor = tensor.mean(dim=1)    # Average over sequence: (B,S,F) -> (B,F)
elif tensor.ndim == 2:
    feature_dim = tensor.shape[1]
else:
    tensor = tensor.reshape(tensor.shape[0], -1)
    feature_dim = tensor.shape[1]

stats = self._get_stats(name, feature_dim, tensor.device)
```

**Impact**: For Mamba with shape `(batch, 512, 2560)`, this correctly extracts `feature_dim=2560` instead of flattening to some huge number.

### 2. Covariance Aggregation with Mixed Dimensions (icd/graph/correlation.py)

Added logic to handle when different layers have different output dimensions:

```python
def covariance(self) -> Tuple[TorchTensor, List[Dict[str, object]]]:
    covariances = [stats.covariance() for stats in self._stats.values()]

    # Check if all covariances have the same shape
    shapes = [cov.shape for cov in covariances]
    if len(set(shapes)) > 1:
        # Filter to most common dimension
        from collections import Counter
        shape_counts = Counter(shapes)
        most_common_shape, _ = shape_counts.most_common(1)[0]
        filtered_covs = [cov for cov in covariances if cov.shape == most_common_shape]
        covariances = filtered_covs

    # Aggregate
    base = covariances[0].clone()
    for cov in covariances[1:]:
        base += cov
    return base / len(covariances)
```

**Impact**: If some layers output 5120-dim and most output 2560-dim, we'll use the majority (2560) and ignore outliers.

### 3. Fallback Validation (icd/runtime/orchestrator.py)

Added a safety check to validate correlation graph dimensions match the base graph:

```python
W_iter = correlation_to_csr(cov, cfg=corr_cfg_obj)

# Validate correlation graph dimensions match base graph
if W_iter.shape[0] != W.shape[0]:
    expected_dim = W.shape[0]
    actual_dim = W_iter.shape[0]
    logger.warning(
        "[IASP] Correlation graph dimension mismatch: expected D=%d (from base graph), "
        "got D=%d (from activations). Falling back to base graph W.",
        expected_dim,
        actual_dim,
    )
    W_iter = W
    correlation_meta["dimension_mismatch"] = {
        "expected": expected_dim,
        "actual": actual_dim,
        "fallback": "base_graph",
    }
```

**Impact**: Even if correlation collection produces wrong dimensions, we'll catch it and fall back to the base graph instead of crashing during permutation application.

## Why This Happened

1. **graph_pytorch.py fix (commit 9b57362)** correctly fixed dimension inference for the **base graph** `W` (from FX tracing)
2. But **correlation.py** uses a completely different code path - it hooks layer activations during forward passes
3. The correlation code was naively flattening all tensors, which broke for sequence models (transformers, Mamba, etc.)
4. Mamba outputs 3D tensors `(batch, sequence, features)` in most layers
5. Flattening these produced `batch × sequence × features` = some huge wrong number

## Files Changed

1. **icd/graph/correlation.py** (lines 172-205, 215-235)
   - Handle 3D tensors properly by taking last dimension and averaging over sequence
   - Aggregate only covariances with matching (most common) dimensions

2. **icd/runtime/orchestrator.py** (lines 897-912)
   - Add validation to detect dimension mismatches between correlation and base graphs
   - Fall back to base graph if mismatch detected

## Testing

After applying this fix:

```bash
# In Docker container
cd /workspace/iterative-co-design
find icd -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
pip install -e . --no-deps --force-reinstall

# Run smoke test
python -m icd.cli.main run -c configs/mamba_smoke_test.json --out /tmp/test_fixed
```

Expected output:
- ✅ No more `[IASP] skip backbone.layers.X.mixer` messages
- ✅ Permutations applied successfully to all 64 Mamba layers
- ✅ Correlation graph has D=2560 (matching hidden_size)
- ✅ Iterative co-design actually runs (not falling back to dense)

## Related Issues

- Original fix in commit `9b57362` only addressed FX-based graph construction
- This fix addresses the **orthogonal issue** in correlation-based graph construction
- Both issues had the same symptom (4096 vs 2560) but different root causes

## Additional Notes

The `expected_dim` parameter was added to `CorrelationConfig` for future use - we could pass `hidden_size` explicitly to force correlation collection to match, but the automatic detection via 3D tensor handling should work for most cases.