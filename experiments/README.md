# NVIDIA A100 Experimental Guide

This guide provides a streamlined workflow for running all paper experiments on a single A100 GPU via RunPod.

## Initial Setup (Run Once)

### 1. Launch RunPod Instance
- Select: **1x A100 (40GB or 80GB)** with CUDA 11.8+
- Template: PyTorch or Base CUDA image
- Storage: At least 100GB

### 2. Clone and Install
```bash
# SSH into your pod
cd /workspace

# Clone repository
git clone https://github.com/mrcha033/iterative-co-design.git
cd iterative-co-design

# Create environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e .[experiments]
pip install transformers datasets mamba-ssm torch-geometric ogb

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

### 3. Set Up Nsight Compute
```bash
# Verify ncu is available
ncu --version

# Set environment variable
export ICD_NCU_CMD='ncu --metrics l2_cache_hit_rate,dram__throughput.avg.pct_of_peak_sustained_elapsed --target-processes all --export json --export-file {out}'

# Add to ~/.bashrc for persistence
echo 'export ICD_NCU_CMD="ncu --metrics l2_cache_hit_rate,dram__throughput.avg.pct_of_peak_sustained_elapsed --target-processes all --export json --export-file {out}"' >> ~/.bashrc
```

### 4. Create Experiment Directory Structure
```bash
mkdir -p experiments/{table1,quantization,mechanism,ablations,generalization,figures,results}
mkdir -p experiments/table1/{mamba,bert,resnet,gcn}/{dense,algo_only,linear,iterative}
```

---

## Experiment Execution Order

**Estimated Total Time: 40-60 GPU hours**

### Phase 1: Main Results (Table 1) - **Priority 1**
**Time: ~20 hours | GPU Memory: ~30GB peak**

This is your core contribution - run this first!

```bash
cd /workspace/iterative-co-design
source venv/bin/activate

# Run the batch script (handles all 4 architectures × 4 baselines × 5 runs)
bash experiments/scripts/run_table1_batch.sh
```

**What this does:**
- Mamba-2.8B: Dense → Algo-Only → Linear → Iterative (5 runs each)
- BERT-large: Same pattern
- ResNet-50: Same pattern
- GCN: Same pattern
- **Total: 80 runs**

**Monitor progress:**
```bash
# Check logs
tail -f experiments/table1/progress.log

# Check GPU utilization
watch -n 1 nvidia-smi
```

**Outputs:**
- `experiments/table1/*/run_*/metrics.json` - Latency, L2 cache, modularity
- `experiments/table1/results_summary.csv` - Aggregated metrics

---

### Phase 2: Quantization Experiment (Figure 2) - **Priority 2**
**Time: ~4 hours | GPU Memory: ~30GB**

```bash
bash experiments/scripts/run_quantization_batch.sh
```

**What this does:**
- Quant→Permute: 6 runs
- Permute→Quant: 6 runs
- Iterative (Permute→Quant→RePermute): 6 runs
- **Total: 18 runs**

**Outputs:**
- `experiments/quantization/*/run_*/metrics.json`
- `experiments/quantization/summary.csv`

---

### Phase 3: Mechanistic Analysis - **Priority 3**
**Time: ~8 hours | GPU Memory: ~35GB**

```bash
# 3.1 Deep Dive: Modularity-Cache-Latency (Table 3)
bash experiments/scripts/run_mechanism_deepdive.sh

# 3.2 Synthetic Validation (Figure 3)
python scripts/generate_synthetic_validation.py --output experiments/mechanism/synthetic/
python scripts/validate_synthetic.py experiments/mechanism/synthetic/

# 3.3 Memory Hierarchy Analysis (Table 5)
bash experiments/scripts/run_memory_hierarchy.sh
```

**Outputs:**
- `experiments/mechanism/mamba_deepdive/results.csv` - Modularity, L2 hit rate, latency
- `experiments/mechanism/synthetic/validation_results.json`
- `experiments/mechanism/memory_hierarchy/hierarchy_metrics.json`
- `experiments/results/table3_mamba_deepdive.csv` (summary) and `experiments/results/table3_mamba_deepdive_raw.csv`

---

### Phase 4: Ablation Studies - **Priority 4**
**Time: ~12 hours | GPU Memory: ~30GB**

```bash
# Run comprehensive ablations
bash experiments/scripts/run_ablations_batch.sh
```

**What this does:**
- Component ablations (clustering variants, objectives, iterations)
- Sensitivity analysis (hyperparameters, layer selection)
- Cross-architecture comparisons (modularity vs TSP)

**Outputs:**
- `experiments/ablations/component/results.csv`
- `experiments/ablations/sensitivity/results.csv`
- `experiments/ablations/cross_arch/results.csv`

---

### Phase 5: Generalization (Hardware/Batch Size) - **Priority 5**
**Time: ~6 hours | GPU Memory: ~30GB**

```bash
# Since you only have A100, document it as such
bash experiments/scripts/run_generalization_a100.sh

# Batch size sensitivity
bash experiments/scripts/run_batch_sensitivity.sh
```

**Outputs:**
- `experiments/generalization/a100/results.csv`
- `experiments/generalization/batch_sensitivity/scaling_data.csv`

---

### Phase 6: TVM Baseline Comparison - **Priority 6**
**Time: ~8 hours (3000 trials per model)**

```bash
# Run AutoTVM/Ansor tuning (long-running)
python scripts/run_autotvm.py --model mamba-2.8b --trials 3000 --out experiments/tvm/mamba/
python scripts/run_autotvm.py --model bert-large --trials 3000 --out experiments/tvm/bert/

# Compare results
python scripts/compare_tvm_baseline.py experiments/tvm/ experiments/table1/ --output experiments/tvm/comparison.csv
```

---

## Data Analysis & Figure Generation

After experiments complete, run analysis scripts:

```bash
cd /workspace/iterative-co-design

# Table 1: Statistical analysis with paired t-tests, Cohen's d
python scripts/analyze_table1.py experiments/table1/ --output experiments/results/table1_stats.csv

# Figure 2: Quantization bar chart
python scripts/plot_quantization_barchart.py experiments/quantization/ --output experiments/figures/quantization_results_barchart.png

# Table 3: Mechanistic analysis
python scripts/extract_mechanism_metrics.py experiments/mechanism/mamba_deepdive/ --output experiments/results/table3_mamba_deepdive.csv

# Figure 3: Synthetic validation
python scripts/plot_synthetic_validation.py experiments/mechanism/synthetic/ --output experiments/figures/synthetic_validation.png

# Figure 5: Hardware heatmap (single GPU, but document as A100 baseline)
python scripts/plot_hardware_heatmap.py experiments/generalization/ --output experiments/figures/hardware_generalization_heatmap.png

# Figure 6: Pareto frontier
python scripts/plot_pareto_frontier.py experiments/pareto/ --output experiments/figures/pareto_frontier_mamba.pdf

# Figure 7: Statistical distributions
python scripts/plot_latency_distributions.py experiments/table1/ --output experiments/figures/latency_distributions.png
python scripts/plot_scaling_width.py experiments/scaling/ --output experiments/figures/scaling_with_width.png

# Mediation analysis (Bootstrap)
python scripts/mediation_analysis.py experiments/mechanism/mamba_deepdive/ --bootstrap-samples 5000 --output experiments/results/mediation_analysis.json

# Generate all figures at once
python scripts/generate_all_figures.py experiments/ --output experiments/figures/
```

---

## Download Results to Local Machine

```bash
# On your local machine
rsync -avz --progress \
  root@[runpod-ip]:/workspace/iterative-co-design/experiments/ \
  ./experiments_runpod/

# Or use RunPod's file browser to download:
# - experiments/results/*.csv
# - experiments/figures/*.png
# - experiments/*/metrics.json
```

---

## Cost Optimization Tips

### 1. Run in Batches
If budget is tight, pause between phases:
- **Day 1:** Table 1 (20h)
- **Day 2:** Quantization + Mechanism (12h)
- **Day 3:** Ablations + TVM (20h)

### 2. Use Spot Instances
RunPod spot instances are ~70% cheaper. Enable auto-resume if interrupted.

### 3. Monitor GPU Utilization
```bash
# If GPU utilization < 80%, you might be CPU-bound
# Check with:
watch -n 1 'nvidia-smi; echo "---"; top -bn1 | head -20'
```

### 4. Cache Intermediate Results
```bash
# Enable caching to avoid recomputation
python -m icd.cli.main run -c configs/mamba.json \
  --override cache.enable=true \
  --override cache.cache_dir=/workspace/.icd_cache \
  --out experiments/cached_run/
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in configs
# Edit configs/mamba.json:
{
  "graph": {
    "loader_kwargs": {
      "batch_size": 1  // Reduce if OOM
    }
  }
}
```

### Nsight Compute Fails
```bash
# Check if running as root
sudo ncu --version

# Or fall back to mock profiling
unset ICD_NCU_CMD
```

### Slow Experiment
```bash
# Check if using CPU by accident
python -c "import torch; print(torch.cuda.current_device())"

# Verify config has GPU device
grep -r "device" configs/*.json
```

---

## Quick Start (Minimal Viable Paper)

If you need results FAST (1 day = 24h):

```bash
# Run only critical experiments for paper acceptance:

# 1. Table 1 - Mamba only (4h)
bash experiments/scripts/run_table1_mamba_only.sh

# 2. Quantization - Mamba (2h)
bash experiments/scripts/run_quantization_mamba_only.sh

# 3. Mechanism - Mamba (4h)
bash experiments/scripts/run_mechanism_mamba_only.sh

# 4. Key ablations - Mamba (4h)
bash experiments/scripts/run_ablations_key_only.sh

# Total: 14h, leaves 10h buffer
```

This gives you enough data to defend the core claims. Add other architectures later if requested by reviewers.

---

## Paper Reproducibility Package

After experiments complete, create a reproducibility archive:

```bash
cd /workspace/iterative-co-design

# Create archive
tar -czf experiments_archive_$(date +%Y%m%d).tar.gz \
  experiments/results/ \
  experiments/figures/ \
  experiments/*/run_*/metrics.json \
  experiments/*/run_*/config.lock.json

# Upload to Zenodo/Figshare/Google Drive for paper supplement
```

---

## Estimated Costs

| RunPod Tier | $/hour | 60h Total | Notes |
|-------------|--------|-----------|-------|
| A100 80GB On-Demand | $1.99 | **$119** | Guaranteed availability |
| A100 80GB Spot | $0.79 | **$47** | May be interrupted |
| A100 40GB Spot | $0.59 | **$35** | Sufficient for most experiments |

**Recommendation:** Use A100 40GB Spot for $35-50 total cost.

---

## Next Steps

1. **Start with Phase 1 (Table 1)** - this validates your core claim
2. **Run overnight** - these are long-running experiments
3. **Check logs regularly** - catch errors early
4. **Download results incrementally** - don't lose data if pod terminates

Good luck with your experiments! 🚀