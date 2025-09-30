# RunPod Quick Fix - Ready to Run

**Date:** 2025-09-30
**Status:** ✅ FIXED - Ready for experiments

## Issue Resolved

**Error:** `ModuleNotFoundError: No module named 'mamba_ssm'`

**Root Cause:** Experiment scripts were using `configs/mamba_ssm_2.8b.json` which requires the `mamba-ssm` package (original implementation), but this package is not installed on RunPod.

**Solution:** Switched all scripts to use `configs/mamba_3b.json` which uses HuggingFace Transformers (`state-spaces/mamba-2.8b-hf`). This leverages our newly implemented HF Mamba permutation support.

---

## What Changed

### Updated Scripts:
1. ✅ `experiments/scripts/run_table1_batch.sh` → Uses `mamba_3b.json`
2. ✅ `experiments/scripts/run_quantization_batch.sh` → Uses `mamba_3b.json`
3. ✅ `experiments/scripts/runpod_quickstart.sh` → Uses `mamba_3b.json`

### Commits:
- `ad0d052` - feat: Add HuggingFace Mamba permutation support (Phase 1)
- `928f0ba` - fix: Switch experiment scripts to use HuggingFace Mamba config

---

## Ready to Run on RunPod

### Step 1: Pull Latest Changes

```bash
cd /workspace/iterative-co-design
git pull origin main
```

**You should see these commits:**
```
ad0d052 feat: Add HuggingFace Mamba permutation support (Phase 1)
928f0ba fix: Switch experiment scripts to use HuggingFace Mamba config
```

### Step 2: Verify Environment

```bash
source venv/bin/activate

# Verify transformers is installed (should be already)
python -c "from transformers import MambaForCausalLM; print('✓ transformers OK')"

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

### Step 3: Run Experiments

**Option A: Quick Start (14 hours, minimal viable paper)**
```bash
bash experiments/scripts/runpod_quickstart.sh
```

**Option B: Full Table 1 (20 hours, all architectures)**
```bash
bash experiments/scripts/run_table1_batch.sh
```

**Option C: Quantization only (4 hours)**
```bash
bash experiments/scripts/run_quantization_batch.sh
```

### Step 4: Monitor Progress

```bash
# Watch logs
tail -f experiments/table1_minimal/progress.log  # For quickstart
tail -f experiments/table1/progress.log          # For full table1

# Watch GPU
watch -n 1 nvidia-smi
```

---

## What This Tests

Running these experiments will:
1. ✅ Test HuggingFace Mamba permutation support (first real-world use!)
2. ✅ Validate dimension expansion logic (hidden_size → intermediate_size)
3. ✅ Verify all transformations work correctly (in_proj, x_proj, out_proj, A_log, D, conv1d)
4. ✅ Collect real performance metrics on A100 GPU

---

## Expected Behavior

### During Graph Construction (CPU phase):
- Low GPU utilization (2-5%) - **This is normal!**
- High CPU usage
- Building correlation matrices

### During Measurement (GPU phase):
- High GPU utilization (80-95%)
- Running forward passes
- Profiling with Nsight Compute

### Output Files:
```
experiments/table1_minimal/
├── dense/run_1/
│   ├── metrics.json          # Latency, L2 cache, modularity
│   ├── config.lock.json      # Exact config used
│   └── experiment.log        # Full log
├── linear/run_1/
├── iterative/run_1/
└── progress.log              # Overall progress
```

---

## Troubleshooting

### If you see "no applicable Mamba modules" error:
This should NOT happen anymore with the fix. If it does:
1. Make sure you pulled latest changes (`git log` should show `928f0ba`)
2. Check config is correct: `grep model_name configs/mamba_3b.json` should show `mamba-2.8b-hf`

### If experiments are slow:
- Graph construction is CPU-bound (normal, takes 10-30 min)
- Once measurements start, GPU should be 80%+
- Check: `watch -n 1 'nvidia-smi; echo "---"; top -bn1 | head -10'`

### If CUDA OOM:
- Shouldn't happen with Mamba-2.8B on A100 40GB
- If it does, reduce batch size in config

---

## What Happens Next

After experiments complete:
1. Download results: `rsync -avz root@[runpod-ip]:/workspace/iterative-co-design/experiments/ ./experiments_runpod/`
2. Run analysis: `python experiments/scripts/aggregate_table1_minimal.py experiments/table1_minimal/`
3. Generate figures: (plotting scripts to be created)

---

## Success Criteria

Experiments are successful if:
- ✅ No "no applicable Mamba modules" error
- ✅ Forward passes complete without NaN/Inf
- ✅ GPU utilization reaches 80%+ during measurement
- ✅ Output metrics.json files are created
- ✅ Latency improvements of 15-25% vs dense baseline (as claimed in paper)

---

## Summary

**Status:** ✅ Ready to run
**Command:** `bash experiments/scripts/runpod_quickstart.sh`
**Time:** 14 hours
**Output:** All core paper data

The HF Mamba permutation support is implemented and integrated. This is now a real-world test of the implementation!