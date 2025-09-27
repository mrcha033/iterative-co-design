# SOP — Measurement Standard (Latency / L2 Hit / Energy-per-Token)

**One-line summary**
Reproduce pre/post measurements for `permute → transform(S/Q/K) → re-permute` under a fixed environment, and capture Latency, L2 Hit, and EpT (J/token) with confidence intervals. If any stage fails, roll back and rerun immediately.

---

## 0) Assumptions & Constraints

* **Scope**: Inference-only measurements. Excludes training and distributed setups.
* **HW/tools**: ≥1 A100/H100, Python 3.10, Nsight Compute (ncu CLI), NVML power sampling (or an equivalent board power meter).
* **Related documents**: PRD (targets), SAS (components/architecture), ICD (APIs/artifacts), Pass Doc (IR flow), Cost Spec (objective).

---

## 1) Outputs

* **Metrics summary** `metrics.json`: Latency (ms){mean,p50,p95}, tokens/s, L2_hit (%), EpT (J/token), quality deltas (Δacc, etc.), environment hash.
* **Raw logs**: `ncu.json` (or section report), `power.csv` (t, W), `run.log` (events), `config.lock.json`.
* **Reports**: `report.html|csv` with pre/post tables and figures.

---

## 2) Environment Lock (Required Checklist)

* [ ] **Lock GPU state**: persistence mode on; fix application clocks/power cap when possible.
* [ ] **Record driver/library versions**: CUDA, cuDNN, framework versions.
* [ ] **Stabilize temperature & warm up**: Run N_warmup (default 50) iterations.
* [ ] **Fix RNG/seed**: Align data sample, model init, and sampler seeds.
* [ ] **Eliminate background load**: Prefer dedicated node / fixed CPU governor.
* [ ] **Fix inputs**: Keep batch size, sequence length, padding policy, and KV-cache settings identical.

> If this fails, the **measurement is invalid**. Re-lock the environment before retrying.

---

## 3) File & Path Layout

```
{out}/
  config.lock.json   # Snapshot of the effective configuration
  run.log            # Stage-by-stage event log
  ncu.json           # L2/profile results (when available)
  power.csv          # t(s), power_w
  metrics.json       # Summary metrics
  report.html|csv    # Pre/post comparison
```

---

## 4) Step-by-step Procedure

### 4.1 Capture the Environment Fingerprint

1. Query GPU, driver, clocks, power cap, ECC, temperature, and UUID → write to `run.log`.
2. Record Git commit/branch, Docker/conda environment, package versions, model/dataset hashes.
3. Mark any parameter that cannot be fixed (clock, power, etc.) explicitly as “unlocked”.

### 4.2 Warmup & Stabilization

1. Run **50 warmup iterations** with the same input (PRD default).
2. Check convergence of temperature/power fluctuations (e.g., rolling std).

### 4.3 Measure Latency & Throughput

1. Repeat the same input with `repeats=1000`.
2. Use **wall-clock time** (high-resolution monotonic clock) to collect per-iteration latency → compute p50/p95.
3. Compute **tokens/s** from total tokens processed ÷ total time.
4. Write `{lat_ms, toks_per_s}` into `metrics.json`.

### 4.4 Measure L2 Hit

* **Preferred**: Run ncu with **section-based** collection (e.g., Memory Workload/Cache) to extract L2 hit rate.
* **Alternative**: Provide explicit metric names (architecture-specific).
* Run at least three repetitions and record the mean ± standard deviation.
* If ncu overhead significantly alters latency, split into **L2-only** and **latency-only** runs.

### 4.5 Measure Power & EpT

1. Sample NVML (or a board power meter) at a **fixed cadence (default 10 Hz)**.
2. Align the measurement window with the latency loop using start/end timestamps.
3. Apply trapezoidal integration to compute **energy [J]**, then **EpT = energy / generated tokens**.
4. Repeat at least three times and report mean + confidence interval.

### 4.6 Compare Results & Decide Acceptance

1. Run **Baseline (Linear)** and **Iterative** flows under **identical conditions**, ≥3 times each.
2. Record Δ for each metric, the 95% CI, and the effect size (g/Hedges).
3. Check whether PRD targets hold (e.g., Latency −20% / L2 +10%p / EpT −15%).
4. If they fail, perform root-cause analysis (§6) and retry once. If the retry fails, **roll back**.

---

## 5) Command Examples (Reference Snippets)

### 5.1 Lock and Record the Environment (when permitted)

```bash
# Locking depends on permissions/models. Log as "unlocked" if it fails.
nvidia-smi -pm 1
nvidia-smi --query-gpu=name,uuid,driver_version,pstate,clocks.gr,clocks.sm,clocks.mem,power.limit,temperature.gpu --format=csv -i 0
# (Optional) Power cap / clock settings: requires privileges
# nvidia-smi -pl 250
# nvidia-smi -ac <memMHz>,<smMHz>
```

### 5.2 Run the Pipeline (Before / After)

```bash
# Baseline (linear)
icd run -c config.yaml --override pipeline.mode=linear --out runs/linear

# Iterative
icd run -c config.yaml --override pipeline.mode=iterative --out runs/iter
```

### 5.3 Collect L2 Metrics (ncu)

```bash
# Preferred: collect Cache/Memory sections together
nv-nsight-cu-cli --section "MemoryWorkloadAnalysis" --section "MemoryChart" \
  --nvtx --kernel-name <your_kernel_regex> --page raw ./runner --config config.lock.json \
  --export json --export-file runs/iter/ncu.json
```

### 5.4 Log Power Samples (Python example)

```python
# measure/power_logger.py
import time, csv
from measure.power_stub import read_power_w  # NVML or NaN
with open("runs/iter/power.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["t_s","power_w"])
    t0 = time.perf_counter()
    for _ in range(int(10*60)):   # 10Hz * 60s example
        t = time.perf_counter()-t0
        w.writerow([f"{t:.6f}", f"{read_power_w():.3f}"])
        time.sleep(0.1)
```

---

## 6) Verification & Quality Assurance

### 6.1 Consistency & Determinism

* Aim for coefficient of variation ≤ 5% for latency under identical input/seed/clock.
* Ensure π/metric hashes match between sessions within ±ε.

### 6.2 Confidence Intervals & Statistics

* Use at least n≥3 repetitions (n≥5 recommended) to compute mean and 95% CI.
* Present performance claims as **Δ with CI** (e.g., “−22% [−19, −25]”).

### 6.3 Outlier Handling

* Detect outliers via IQR/MAD, log root causes (scheduling/jitter/thermal), and **document whether they were excluded**.

### 6.4 Overhead Guard

* If latency with ncu enabled worsens by **>10%** vs. baseline, switch to **separate runners**.

### 6.5 Failure & Rollback Policy

* When **two or more** of Latency/L2/EpT regress, roll back to the previous `π` automatically.
* If regression persists after one re-measurement, produce an RCA report (§7).

---

## 7) RCA (Root-Cause Analysis) Procedure

1. **Environment**: Did driver/clock/temperature/power caps drift?
2. **Inputs**: Any changes in batch/sequence/KV-cache settings?
3. **Kernel**: Switched between fused/unfused paths or different kernels?
4. **Memory**: L2 hit improved but latency stagnates? → Re-check DRAM bandwidth/NoC congestion metrics.
5. **Solver**: Hit the time budget without improvement? Assess initial quality (see Cost Spec ΔJ).
6. **Statistics**: Too few repetitions or outlier influence? → Rerun with larger n.

---

## 8) Security, Ethics, Compliance

* Never include sensitive data, credentials, or file paths in profiles/logs.
* Document dataset/model/license compliance.
* When reporting energy measurements, state **measurement limitations** (sensor sampling, instrument accuracy).

---

## 9) Acceptance Criteria

* Both representative workloads satisfy the **PRD targets**:

  * Latency ≥ 20% reduction, L2 +10%p or higher, EpT −15% or better (quality drop ≤ 0.1%p).
* Artifacts complete: `metrics.json`, `ncu.json` (or equivalent), `power.csv`, `config.lock.json`, `report.*`
* External reviewers can reproduce the results within **24 hours** using documentation and scripts only.

---

## 10) Failure & Fallback Paths

* **NVML unavailable**: Use an external power meter (board/socket) or report EpT as **N/A**.
* **ncu unavailable**: Omit L2 hit, report only Latency/Throughput (state the limitation).
* **Cannot fix clocks**: Record `fixed_clock=false` and tag results as **not comparable** in CI.

---

## 11) Automation Hooks (ICD Integration)

* When `icd run` completes, automatically:

  1. Save the environment fingerprint →
  2. Execute warmup/measurement loops →
  3. Invoke ncu/NVML →
  4. Integrate EpT →
  5. Generate the before/after report.

---

## 12) Appendix — Standard Output Formats (Excerpt)

**metrics.json (example)**

```json
{
  "run_id":"a1b2c3d4e5f6",
  "env":{"gpu":"A100-40GB","driver":"535.xx","fixed_clock":true,"seed":0},
  "pipeline":{"mode":"iterative","repeats":1000,"warmup":50},
  "latency_ms":{"mean":12.8,"p50":12.6,"p95":13.5},
  "throughput_toks_s":1843.2,
  "l2_hit_pct":87.4,
  "ept_j_per_tok":0.92,
  "quality_delta_pct":0.03
}
```

**power.csv (example)**

```
t_s,power_w
0.000000,210.5
0.100112,212.6
...
```

---

## 13) Final Checklist

* [ ] Environment locked / fingerprint captured
* [ ] Warmup and repetition counts satisfied
* [ ] Latency / Throughput recorded
* [ ] L2 hit captured (when available)
* [ ] EpT integral computed (when available)
* [ ] Pre/post comparison with CI and effect size produced
* [ ] Artifacts and reports stored and versioned

---

## 14) Risks & Mitigations

* **Measurement noise**: Increase repetitions, lock clocks, separate profiling runs.
* **Metric name drift**: Use section-based collection for compatibility and maintain per-device mapping tables.
* **Overhead**: Reduce ncu sampling/range or split into profiler-only executions.

---
