# Fix mamba-ssm Import Issue on RunPod

**Issue:** `ModuleNotFoundError: No module named 'mamba_ssm'` even though packages are installed globally.

**Root Cause:** Packages installed globally but not in virtual environment, or virtual environment not activated.

---

## Option 1: Use HuggingFace Mamba (RECOMMENDED) ✅

**Already done!** All experiment scripts now use `configs/mamba_3b.json` which uses HuggingFace Transformers instead of mamba-ssm.

```bash
cd /workspace/iterative-co-design
git pull origin main
source venv/bin/activate
bash experiments/scripts/runpod_quickstart.sh
```

This should work without any mamba-ssm installation issues.

---

## Option 2: Fix mamba-ssm Import (If you prefer original)

### Step 1: Diagnose the Issue

```bash
cd /workspace/iterative-co-design
source venv/bin/activate
python check_mamba_import.py
```

This will show where each package is installed.

### Step 2: Install mamba-ssm in Virtual Environment

If diagnostic shows mamba-ssm is NOT in venv:

```bash
source venv/bin/activate

# Install mamba-ssm and causal-conv1d in venv
pip install mamba-ssm causal-conv1d

# Verify
python -c "from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel; print('✓ OK')"
```

### Step 3: Revert Config Changes

If you want to use original mamba-ssm after installing it:

```bash
# Edit experiment scripts to use mamba_ssm_2.8b.json instead of mamba_3b.json
sed -i 's/mamba_3b\.json/mamba_ssm_2.8b.json/g' experiments/scripts/*.sh
```

---

## Option 3: Use Global Python (Not Recommended)

If packages are installed globally but not in venv:

```bash
cd /workspace/iterative-co-design

# Don't activate venv, use global python
which python  # Should show /usr/bin/python3 or similar
python -c "from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel; print('✓ OK')"

# Run experiments without activating venv
bash experiments/scripts/runpod_quickstart.sh
```

**Caution:** This may cause dependency conflicts.

---

## Quick Decision Matrix

### Use HuggingFace Mamba if:
- ✅ You want it to work immediately (no extra setup)
- ✅ You're testing our new HF Mamba permutation support
- ✅ You don't care about comparing original vs HF implementations

**Command:**
```bash
cd /workspace/iterative-co-design
source venv/bin/activate
bash experiments/scripts/runpod_quickstart.sh  # Already uses HF
```

### Use Original mamba-ssm if:
- ✅ You want exact reproduction of original mamba-ssm behavior
- ✅ You want to compare original vs HF performance
- ✅ You're willing to spend 5 minutes fixing imports

**Commands:**
```bash
cd /workspace/iterative-co-design
source venv/bin/activate
pip install mamba-ssm causal-conv1d
# Revert scripts to use mamba_ssm_2.8b.json
bash experiments/scripts/runpod_quickstart.sh
```

---

## Recommended Path Forward

**I recommend Option 1 (HuggingFace Mamba)** because:

1. ✅ **It already works** - no extra setup needed
2. ✅ **Tests our implementation** - validates the HF Mamba permutation support we just built
3. ✅ **Same model** - `state-spaces/mamba-2.8b-hf` is the same model as `state-spaces/mamba-2.8b`, just different implementation
4. ✅ **Better compatibility** - HuggingFace Transformers has better package management

The original mamba-ssm package is a research codebase that can have dependency conflicts. The HuggingFace version is production-ready.

---

## Verification Commands

After choosing either option, verify it works:

```bash
# Test import
python -c "from transformers import MambaForCausalLM; print('✓ HF Mamba OK')"
# OR
python -c "from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel; print('✓ Original Mamba OK')"

# Test full loader
python -c "
from icd.experiments.hf import load_hf_causal_lm
model, inputs = load_hf_causal_lm('state-spaces/mamba-2.8b-hf', device='cpu', sequence_length=128)
print(f'✓ Loaded model: {model.config.model_type}')
"
```

---

## What Configs Do

### mamba_3b.json (HuggingFace)
```json
{
  "model_loader": "icd.experiments.hf.load_hf_causal_lm",
  "model_loader_kwargs": {
    "model_name": "state-spaces/mamba-2.8b-hf"
  }
}
```
- Uses `transformers.MambaForCausalLM`
- Requires: `pip install transformers` ✅ Already installed
- Structure: `in_proj`, `x_proj`, `out_proj`, `A_log`, `D`

### mamba_ssm_2.8b.json (Original)
```json
{
  "model_loader": "icd.experiments.hf.load_mamba_ssm_causal_lm",
  "model_loader_kwargs": {
    "model_name": "state-spaces/mamba-2.8b"
  }
}
```
- Uses `mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel`
- Requires: `pip install mamba-ssm causal-conv1d` ❌ Import failing
- Structure: `A`, `B`, `C` modules

---

## Summary

**Current Status:**
- Experiment scripts use `mamba_3b.json` (HuggingFace)
- This should work immediately on RunPod
- No mamba-ssm installation needed

**To run now:**
```bash
cd /workspace/iterative-co-design
git pull
source venv/bin/activate
bash experiments/scripts/runpod_quickstart.sh
```

If you want to use original mamba-ssm instead, follow Option 2 above.