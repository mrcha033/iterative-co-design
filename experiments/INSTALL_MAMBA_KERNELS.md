# Install Mamba Optimized CUDA Kernels

**Issue:** HuggingFace Mamba falls back to slow sequential implementation.

**Warning Message:**
```
The fast path is not available because one of `(selective_state_update,
selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`
is None. Falling back to the sequential implementation...
```

**Impact:**
- ❌ ~10-50x slower than optimized implementation
- ❌ May timeout or take excessively long for experiments
- ❌ Won't give realistic performance numbers

**Solution:** Install optimized CUDA kernels

---

## Quick Fix (Recommended)

```bash
cd /workspace/iterative-co-design
source venv/bin/activate

# Install mamba-ssm (provides optimized kernels for HF transformers)
pip install mamba-ssm

# Install causal-conv1d (required dependency)
pip install causal-conv1d

# Verify installation
python -c "
import mamba_ssm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from causal_conv1d import causal_conv1d_fn
print('✓ Optimized kernels installed')
"
```

**After installation, HuggingFace Mamba will automatically use the fast kernels!**

---

## Why This Works

HuggingFace's `transformers` library checks for `mamba_ssm` package at runtime:
1. If `mamba_ssm` is available → uses optimized CUDA kernels
2. If `mamba_ssm` is missing → falls back to slow Python implementation

By installing `mamba-ssm`, you get:
- ✅ Fast selective scan operations
- ✅ Fast causal convolutions
- ✅ ~10-50x speedup
- ✅ Still uses HuggingFace model interface (no config changes needed)

---

## Verification

After installation, test that fast path is used:

```bash
python -c "
from transformers import MambaForCausalLM
import torch

# Load model
model = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')

# Check if fast ops are available
mixer = model.backbone.layers[0].mixer
print(f'selective_scan_fn: {mixer.selective_scan_fn is not None}')
print(f'causal_conv1d_fn: {mixer.causal_conv1d_fn is not None}')

if mixer.selective_scan_fn is not None:
    print('✓ Fast path will be used!')
else:
    print('✗ Still using slow sequential implementation')
"
```

Expected output:
```
selective_scan_fn: True
causal_conv1d_fn: True
✓ Fast path will be used!
```

---

## If Installation Fails

If you get compilation errors:

### Option 1: Try pre-built wheels
```bash
pip install mamba-ssm --no-build-isolation
```

### Option 2: Install from conda (if available)
```bash
conda install -c conda-forge mamba-ssm
```

### Option 3: Use original mamba-ssm config
If kernels won't install, switch back to original mamba-ssm implementation:

```bash
# Your original error showed mamba-ssm IS installed globally
# Use it without venv:
cd /workspace/iterative-co-design

# Edit scripts to use global python (no venv)
sed -i 's|source venv/bin/activate|# source venv/bin/activate|g' experiments/scripts/*.sh

# Switch configs back to original
sed -i 's/mamba_3b\.json/mamba_ssm_2.8b.json/g' experiments/scripts/*.sh

# Run experiments with global python
bash experiments/scripts/runpod_quickstart.sh
```

---

## Best Solution

The cleanest approach:

```bash
cd /workspace/iterative-co-design
source venv/bin/activate

# Install both packages in venv
pip install mamba-ssm causal-conv1d

# Verify both work
python -c "
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from transformers import MambaForCausalLM
print('✓ Both original and HF Mamba available')
print('✓ Fast kernels available')
"

# Now run experiments (already using HF config)
bash experiments/scripts/runpod_quickstart.sh
```

This gives you:
- ✅ HuggingFace Mamba interface (better compatibility)
- ✅ Optimized CUDA kernels (fast performance)
- ✅ Our new permutation support (tested implementation)
- ✅ Original mamba-ssm also available (for comparison)

---

## Summary

**Current Issue:** HF Mamba using slow sequential fallback

**Fix:** `pip install mamba-ssm causal-conv1d`

**Why:** HF transformers checks for mamba-ssm and uses its optimized kernels if available

**Time to fix:** 2-3 minutes

**Command:**
```bash
cd /workspace/iterative-co-design
source venv/bin/activate
pip install mamba-ssm causal-conv1d
bash experiments/scripts/runpod_quickstart.sh
```

After this, experiments should run at full speed with optimized CUDA kernels!