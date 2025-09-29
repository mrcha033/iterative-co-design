# PRD — Iterative HW–SW Co-Design for Memory-Aware Layout Re-Optimization

**One-line summary**
This product specification locks in the goals, scope, metrics, and risks for an **ML systems** deliverable (library + CLI) that consistently achieves **higher L2 hit, lower latency, and lower EpT** by re-permuting layouts after each state transformation (sparsity/quantization/KV-cache compression).

---

## 1) Background / Problem Statement

* Modern models (SSM/Transformers) are heavily **I/O-bound**, making **data movement** the bottleneck.
* Existing pipelines stop at a single optimization pass like `permute → transform(S/Q/K)`, so they fail to reflect the **cost landscape after the state changes**.
* Goal: Automatically **re-permute immediately after each transform** to improve **cache locality (L2 hit)** and reduce **latency/energy (EpT)**.

## 2) Outcomes / Business Impact

* **Inference efficiency**: At equal quality, deliver **Latency −20%**, **L2 hit +10%p**, and **EpT −15%** or better on the representative workloads.
* **Developer productivity**: Integrate non-invasively via a one-line API/CLI in existing pipelines.
* **Reproducibility & adoption**: Provide public scripts and logs to meet **Artifact Evaluation** expectations.

## 3) In-scope vs. Out-of-scope

**In-scope**

* Layout optimization core (spectral initialization + local search), S/Q/K transformation adapters, cost model, measurement pipeline (L2/Latency/EpT), Python library/CLI, and the TVM or StableHLO integration **PoC**.

**Out-of-scope**

* Training acceleration/distributed pipelines, custom hardware design, full large-model retraining, and automation for commercial cluster orchestration.

## 4) User Scenario (Representative Flow)

1. A user starts from an existing inference script (current repository example):

   ```bash
   python -m icd.cli.main run -c configs/mock.json --override pipeline.mode=iterative --out runs/iter
   ```
   See docs/USAGE.md for more examples.
2. The framework runs `(permute → transform → re-permute)`, saving a **before/after report** with L2 hit, latency, and EpT.
3. The user **caches and reuses** the optimal permutation without additional intervention.

## 5) Success Metrics (Definitions & Measurement)

* **Latency (ms)**: Average wall-clock time over 1000 inferences (warmup excluded, fixed clock/power-cap options).
* **Throughput (tokens/s)**: Average throughput under the same conditions.
* **L2 Hit (%)**: L2 hit rate from Nsight Compute or an equivalent tool.
* **Energy-per-Token (J/token, EpT)**: NVML or power-meter integral of samples ÷ generated token count.
* **Adoptability**: One-line API integration completes within ≤ 30 minutes (based on the guide) with failure rate < 5%.

> **Acceptance criteria**
>
> * Both representative workloads (SSM-Mamba-2.8B, Transformer-BERT-base) satisfy Latency −20% / L2 +10%p / EpT −15% with quality loss ≤ 0.1%p.
> * Deliver the `icd` CLI, Python API, and HTML/CSV report artifacts.
> * External reviewers can reproduce all results within **24 hours** using the repro scripts.

## 6) Requirements (Functional / Non-functional)

### Functional

* `repermute.fit(W|trace)`: Produce a permutation from the shared-access weight matrix W (or directly from the trace).
* `transform.apply({sparsity|quant|kv})`: Execute the state transformation and update metadata.
* `icd run …`: Orchestrate the pipeline (logging, profiling, reporting).
* `collect_correlations/cluster_graph`: After transforms, build activation-based correlation matrices and Louvain clusters to seed re-permutation.
* `measure.builtin=benchmark`: Built-in GPU benchmark that aggregates Latency/Throughput/EpT with Nsight/NVML options.
* Caching/reuse: Reuse permutations under the same conditions with version tagging.

### Non-functional

* **Performance**: Re-permute in **≤ 5 minutes** for D=2.5k dimensions (offline) with overhead much smaller than accumulated savings.
* **Reproducibility**: Lock environments with Docker/conda and enforce seed/clock SOPs.
* **Observability**: Structured event/metric schema with < 1% overhead.
* **Instrumentation automation**: Persist benchmark/cluster results to `metrics.json` and `correlation/` artifacts; automatically evaluate PRD gates (Lat −20%, L2 +10%p, EpT −15%).
* **Safety**: On failure, automatically roll back to the previous layout and record root causes in logs.

## 7) Architecture Overview (Top-level Blocks)

* **core/**: Graph construction (access co-occurrence → W), cost model (C(π), Q), solver (spectral + local). Convergence/approximation/complexity guarantees are documented in `docs/Theoretical_Analysis.md`.
* **adapters/**: S/Q/K transformation adapters (HDS 2:4, PTQ/FP8, KV compression).
* **runtime/**: Execution pipeline, permutation cache, failure recovery.
* **measure/**: Nsight/NVML wrappers and HTML/CSV reporters.
* **bridge/**: Optional TVM/StableHLO pass PoC.

## 8) Milestones & Schedule (3-week baseline)

* **W1**: core/mock/measure scaffolding, cost-model + solver unit tests, CLI draft.
* **W2**: Integrate the S/Q/K loop, complete L2/Latency/EpT E2E measurement, automate reporting.
* **W3**: Validate model snapshots (SSM/Transformer), finalize documentation/repro package, audit performance targets.

## 9) Dependencies & Preconditions

* **Environment**: Python 3.10, (mps|cuda|cpu) guard, CUDA 11.x+, Nsight Compute, NVML.
* **Resources**: ≥1 A100/H100 GPU (profiling recommended), public datasets (PTB/WikiText/SST-2).
* **Risk sharing**: Control driver/clock drift via the SOP.

## 10) Risks & Mitigations

* **R1 Measurement variability**: Power/cache noise → fix clocks, warm up, repeat runs, and gate via CI microbenchmarks.
* **R2 Scale**: Spectral methods are O(D³) → use partial eigenvectors/sampling/block parallelism and enforce time caps.
* **R3 Generalization**: Bias toward two workloads → run ablation sweeps (sparsity/precision/sequence length) and optionally add a third model.
* **R4 Integration cost**: Framework dependencies → keep API/CLI non-intrusive and limit the IR bridge to a PoC.

## 11) Non-functional Quality & Governance

* **Documentation**: Deliver SAS/ICD/Pass-Doc/Cost-Spec/SOP/Repro-Pack.
* **Testing**: Cover unit, integration, E2E, performance, and regression cases, including determinism and numerical stability.
* **Security/Licensing**: SBOM, OSS license compliance, and data usage policies.

## 12) Decision Matrix (Approach Options)

| Option                                           | Impact                    | Risk            | Cost | Complexity | Decision  |
| ------------------------------------------------ | ------------------------- | --------------- | ---- | ---------- | --------- |
| **A** Iterative re-permute (S/Q/K) + measurement pipeline | Direct L2/Latency/EpT improvement | Solver scalability | Medium | Medium     | **Selected** |
| B Add only a cost model (keep sequential pipeline)       | Easy to implement        | Limited gains    | Low  | Low        | Deferred  |
| C Lead with compiler integration (IR/pass)               | Long-term expansion      | Slower initial velocity | Medium–High | High | Follow-up |

## 13) Execution Checklist

* [ ] PRD approved (Owner/Reviewer signatures)
* [ ] SOP/test plan finalized (fixed clocks, repeats, warmup)
* [ ] Latency/L2/EpT auto-report scripts complete
* [ ] Two representative workloads meet targets
* [ ] Repro package (with raw logs) passes external review

## 14) Roles & Responsibilities (RACI)

* **Owner/PM**: Approve goals/scope/metrics, manage risk
* **Tech Lead**: Own architecture/core design, performance
* **Systems Eng.**: Runtime/measurement, observability
* **ML Eng.**: S/Q/K modules, data/model cards
* **Compiler Eng.** (optional): IR-bridge PoC
* **QE**: SOP, reproducibility, regression testing

---

### Appendix A — Glossary (Quick)

* **S/Q/K**: Sparsity / Quantization / KV-cache compression
* **Q (modularity)**: Metric for intra-community density
* **EpT**: Energy per Token

---

**Completion summary**
Ship the **MVP** once both representative workloads hit **Latency −20%, L2 +10%p, EpT −15%**, and the reports/repro artifacts/documentation set are complete.
