# Repro Plan — Artifact Evaluation & Reproduction Procedure

**One-line summary**
Automate installation → environment locking → data/model setup → experiments → validation → artifact packaging so that the `linear` vs `iterative` performance comparison can be reproduced within **24 hours** using the same environment, inputs, and seeds.

---

## 1) Assumptions & Constraints

Experiments are provided at two levels:

1. **Mock smoke test** — validates structure/pipeline without heavy dependencies (`configs/mock.json`).
2. **HuggingFace BERT/Mamba** — loads real models and runs inference (`configs/bert.json`, `configs/mamba.json`, `configs/bert_large.json`, `configs/mamba_3b.json`).

To run the HuggingFace experiments (example CPU setup):

```bash
pip install -e .[experiments]
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install mamba-ssm  # required for the Mamba experiment
```

For GPU runs, install the PyTorch wheel that matches your CUDA/cuDNN version; see `docs/USAGE.md` for options.

* **Goal**: Reproduce the PRD targets (Latency −20%, L2 +10%p, EpT −15%) on the two representative workloads (SSM/Mamba-2.8B and Transformer/BERT-base).
* **Environment**: Python 3.10, CUDA 11.x+, one A100/H100 recommended (ncu/NVML available). Include a mock-only path for CPU-only reproduction.
* **Data/models**: Use public checkpoints (Hugging Face, etc.). No proprietary datasets.
* **Determinism**: Fix seeds/clocks and follow the SOP warmup/repeat rules.

---

## 2) Deliverables (What Reproducers Receive)

* **Repro package**:

  * `Makefile`, `scripts/repro_{smoke,full,ae}.sh`, `scripts/collect_artifacts.py`
  * `env/environment.yml` (conda), `Dockerfile` (optional)
  * `configs/{mock.json,trace.json,bert.json,mamba.json,bert_large.json,mamba_3b.json}`
  * `docs/{SOP.md, TESTPLAN.md, PRD.md, SAS.md, ICD.md, PASS_DOC.md, COST_SPEC.md}`
* **Expected outputs**: `runs/*/metrics.json`, `report.{html,csv}`, `ncu.json` (if available), `power.csv`, `config.lock.json`.

---

## 3) Preparation (Reproducer Actions)

### 3.1 Software Installation

```bash
git clone <repo> && cd <repo>
conda env create -f env/environment.yml && conda activate icd
pip install -e .[experiments]
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install mamba-ssm  # required for the Mamba experiment
# Optional Docker path:
# docker build -t icd-repro -f Dockerfile .
```

> Control the HuggingFace cache location via `HF_HOME` or `TRANSFORMERS_CACHE` if needed.

### 3.2 Hardware Check

```bash
python - <<'PY'
import torch, subprocess
print("CUDA:", torch.cuda.is_available(), "Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("NVML:", subprocess.call(["nvidia-smi"]) == 0)
PY
```

* Enable `nvidia-smi -pm 1` for persistence mode when possible; record “unlocked” if power/clock pinning is not permitted.

### 3.3 Model/Data Preparation (Auto-download)

```bash
make fetch_models   # caches public checkpoints/datasets
```

* For offline use, point `ICD_DATA_DIR` to a pre-populated cache directory.

---

## 4) Reproduction Scenarios (Quick → Full)

### 4.1 Repro 0 — Mock Smoke (<10 minutes, CPU-only)

```bash
bash scripts/repro_smoke.sh
```

**Pass criteria**: `ΔC ≤ −10%` and `ΔQ ≥ 0` (Cost Spec baseline). Generates correlation plots in `report.html`.

### 4.2 Repro 1 — BERT-base (Transformer, HuggingFace)

```bash
python -m icd.cli.main run -c configs/bert.json --override pipeline.mode=linear    --out runs/bert_linear
python -m icd.cli.main run -c configs/bert.json --override pipeline.mode=iterative --out runs/bert_iter
python scripts/validate_results.py runs/bert_*
```

**Smoke gate**: Latency ≤ −10%, L2 ≥ +5%p, EpT ≤ −8%  
**Strict gate**: PRD targets (−20% / +10%p / −15%)

### 4.3 Repro 2 — Mamba-130M (SSM, HuggingFace)

```bash
python -m icd.cli.main run -c configs/mamba.json --override pipeline.mode=linear    --out runs/mamba_linear
python -m icd.cli.main run -c configs/mamba.json --override pipeline.mode=iterative --out runs/mamba_iter
python scripts/validate_results.py runs/mamba_*
```

**Smoke/strict gates** same as above.

### 4.4 Repro 3 — Ablation (optional, ~2 hours)

Sweep sparsity `{0.3, 0.5, 0.7}`, precision `{fp8, int8}`, and sequence length `{256, 1024}`:

```bash
make repro_ablation
```

### 4.5 Repro 4 — Large Models (BERT-large, Mamba-2.8B)

> **Objective**: Reproduce performance and statistical results for large-scale checkpoints as required by NeurIPS artifact evaluation.

```bash
bash scripts/repro_large_models.sh runs/large_models
```

**Execution Features**:

- The script sequentially executes `configs/bert_large.json` and `configs/mamba_3b.json` to automatically generate baseline/iterative pairs.
- For each pair, calls `scripts/validate_results.py` to immediately summarize Latency/L2/EpT improvements (95% CI available in `metrics.json`).
- Default sample counts: BERT-large 600 iterations, Mamba-2.8B 500 iterations (excluding warmup). GPU clock locking via `nvidia-smi -lgc` settings recommended.
- Hardware assumption for the Mamba-2.8B configuration: at least 120 GB of aggregate GPU memory (e.g., dual H100 80GB with NVLink or an H100 120GB SXM) to hold activations for 2048-token contexts without gradient checkpointing.
- Results are saved under `runs/large_models/` as HTML/CSV reports and metrics logs.

### 4.6 Repro 5 — IR Pass PoC (optional)

Enable the StableHLO/TVM bridge for a single run:

```bash
icd run -c configs/mamba.yaml --override bridge.enable=true --out runs/mamba_bridge
```

**Pass criteria**: Numerical equivalence (relative error ≤ 1e-6) with performance drift within ±ε.

---

## 5) Validation & Packaging

### 5.1 Automated Validation

```bash
python scripts/validate_results.py runs/*_{linear,iter}
# Checks: ΔLatency, ΔL2, ΔEpT with confidence intervals; π hash and key metrics vary ≤1%
```

### 5.2 Artifact Packaging

```bash
python scripts/collect_artifacts.py runs/mamba_* runs/bert_* -o artifacts/icd_artifacts.zip
# Include: metrics.json, ncu.json, power.csv, report.{html,csv}, config.lock.json, run.log
```

---

## 6) Example Configuration (Excerpt)

```json
{
  "pipeline": {
    "mode": "iterative",
    "repeats": 40,
    "warmup_iter": 5,
    "fixed_clock": true,
    "runner": "icd.runtime.runners_hf.hf_causal_lm_runner",
    "runner_context": {
      "model_loader": "icd.experiments.hf.load_hf_causal_lm",
      "model_loader_kwargs": {
        "model_name": "state-spaces/mamba-130m-hf",
        "sequence_length": 256,
        "batch_size": 2,
        "prompt": [
          "The quick brown fox jumps over the lazy dog",
          "In a distant galaxy, explorers uncovered"
        ],
        "device": "cpu"
      }
    }
  },
  "graph": {
    "source": "pytorch",
    "normalize": "sym",
    "loader": "icd.experiments.hf.load_hf_causal_lm",
    "loader_kwargs": {
      "model_name": "state-spaces/mamba-130m-hf",
      "sequence_length": 256,
      "batch_size": 2,
      "prompt": [
        "The quick brown fox jumps over the lazy dog",
        "In a distant galaxy, explorers uncovered"
      ],
      "device": "cpu"
    },
    "pytorch": {"hops": 2, "reuse_decay": 0.7, "max_len": 256, "byte_weight": true},
    "nnz_cap": 1500000
  }
}
```

---

## 7) Makefile Targets (Excerpt)

```makefile
.PHONY: env repro_smoke repro_bert repro_mamba pack

env:
        conda env create -f env/environment.yml || true

repro_smoke:
        python -m icd.cli.main run -c configs/mock.json --override pipeline.mode=linear   --out runs/mock_linear
        python -m icd.cli.main run -c configs/mock.json --override pipeline.mode=iterative --out runs/mock_iter
        python scripts/validate_results.py runs/mock_*

repro_bert:
        python -m icd.cli.main run -c configs/bert.json --override pipeline.mode=linear    --out runs/bert_linear
        python -m icd.cli.main run -c configs/bert.json --override pipeline.mode=iterative  --out runs/bert_iter
        python scripts/validate_results.py runs/bert_*

repro_mamba:
        python -m icd.cli.main run -c configs/mamba.json --override pipeline.mode=linear    --out runs/mamba_linear
        python -m icd.cli.main run -c configs/mamba.json --override pipeline.mode=iterative  --out runs/mamba_iter
        python scripts/validate_results.py runs/mamba_*

pack:
        python scripts/collect_artifacts.py runs/* -o artifacts/icd_artifacts.zip
```

---

## 8) Expected Duration (Typical)

* **Mock smoke**: ~10 minutes
* **Mamba-2.8B / BERT**: Solver ≤ 5 minutes per case (offline) plus several minutes for the 1000-iteration measurement loop
* **Ablation**: Total time scales with combinations × (solver + measurement)

---

## 9) Fallback Paths

* **ncu unavailable**: `--override measure.ncu_enable=false` (omit L2 reporting).
* **NVML unavailable**: `--override measure.power_enable=false` (omit EpT).
* **Cannot fix clocks**: Record `fixed_clock=false`, tag comparison tables as “unlocked”, and relax determinism gates.
* **Solver timeout**: Return the initial layout with `stats.improved=false`, retry once, then halt.

---

## 10) Automated Pass/Fail Criteria

* **Smoke pass**:
  * Mock: `ΔC ≤ −10% & ΔQ ≥ 0`
  * Mamba/BERT: Latency ≤ −10%, L2 ≥ +5%p, EpT ≤ −8%
* **Final pass (strict)**: Meet PRD targets (−20% / +10%p / −15%) simultaneously with quality loss ≤ 0.1%p.

---

## 11) Logging & Audit Trail

* Preserve `run.log` (event schema) and `config.lock.json` (effective configuration) for every run.
* Include `git rev-parse HEAD` and `conda list --explicit` in `artifacts/manifest.txt`.

---

## 12) Security & Ethics

* Do not include personal data, file paths, or credentials in artifacts.
* Document model/data licenses; record hashes for assets that cannot be redistributed.

---

## 13) Troubleshooting Notes

* Latency unchanged while L2 improves → inspect DRAM bandwidth/NoC congestion metrics.
* High variance → verify warmup, background load, and temperature; increase repetitions.
* Large π drift → re-check seed, time budget, and alignment parameters.
* Performance drop with the bridge enabled → inspect `legalize-layout` failures or fusion-path changes.

---

## 14) Repro Checklist (For Reviewers)

* [ ] Activated environment (`conda activate icd` or Docker run)
* [ ] `make fetch_models` succeeded (configure cache for offline use if needed)
* [ ] `repro_smoke` passed (report generated)
* [ ] `repro_codesign` executed; Latency/L2/EpT goals confirmed
* [ ] `repro_full` passed the smoke gate
* [ ] PRD gate satisfied (final)
* [ ] `pack` executed; `artifacts/icd_artifacts.zip` ready for submission

---
