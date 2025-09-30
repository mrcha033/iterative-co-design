# Quick Fix for Running Experiments

If you're already running experiments and hitting the "No such file or directory" error, here's how to fix it:

## Option 1: Apply the Fix (Recommended)

```bash
# On RunPod, pull the latest changes
cd /workspace/iterative-co-design
git pull origin main

# Pre-create all directories
bash experiments/scripts/fix_directories.sh

# Resume your experiments - the script will continue from where it failed
```

## Option 2: Manual Directory Creation

If you can't pull, run this manually:

```bash
cd /workspace/iterative-co-design

# Create all experiment directories
for arch in mamba bert resnet50 gcn_arxiv; do
    for baseline in dense algo_only linear iterative; do
        for run in {1..5}; do
            mkdir -p "experiments/table1/$arch/$baseline/run_$run"
        done
    done
done

for strategy in quant_perm perm_quant iterative; do
    for run in {1..6}; do
        mkdir -p "experiments/quantization/$strategy/run_$run"
    done
done
```

## Mamba Fast Path Warning

The warning about "fast path not available" is expected. The fallback implementation works fine for experiments, just slightly slower. To install the fast kernels (optional):

```bash
pip install causal-conv1d>=1.0.0
pip install mamba-ssm --no-build-isolation
```

If installation fails, the fallback will work - latency measurements will still be valid for comparing linear vs iterative approaches.

## Continue Experiments

Once directories are created, just re-run your batch script:

```bash
bash experiments/scripts/run_table1_batch.sh
```

The script will skip completed runs (checks for `metrics.json`) and continue from where it stopped.