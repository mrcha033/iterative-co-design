# Literature Review & NeurIPS Paper Outline

## Purpose
This document summarizes the relevant literature, highlights where the Iterative Co-Design (ICD) system diverges from prior work, and anchors the future NeurIPS submission plan to the infrastructure that already exists in this repository. Each section links the writing plan to available code, experiments, and documentation so the paper drafting process can reuse the current assets.

## 1. Related Work Landscape
### 1.1 Iterative Layout & Memory Optimization
- **Graph partitioning for data locality**: Traditional approaches such as METIS and PaToH focus on offline partitioning without adaptive feedback loops.
- **Tensor permutation for accelerator pipelines**: Prior systems generally rely on fixed permutations calibrated during compilation time, lacking runtime rollback mechanisms.
- **Dynamic sparsity scheduling**: Research on structured sparsity (e.g., N:M pruning) typically optimizes kernels independently from layout choices.

**ICD Novelty Hooks**
- The Louvain + spectral hybrid solver stack (`icd/core/solver.py`) provides multi-level refinement with quality fallback logic (`icd/core/cost.py`).
- Runtime rollback and acceptance gates (`icd/runtime/orchestrator.py`, `icd/measure/gates.py`) enable experimentation with adaptive permutations beyond static compile-time approaches.

### 1.2 Hardware/Software Co-Design Frameworks
- **Co-design through model retraining**: Works such as HW-aware NAS or quantization-aware training focus on adjusting model weights rather than memory layouts.
- **Graph-based co-optimization**: Joint hardware-software studies often target cache line alignment or scheduling but rarely integrate measurement-driven rollback.

**ICD Novelty Hooks**
- ICD couples hardware profiling (`icd/measure/latency.py`, `icd/measure/l2_ncu.py`, `icd/measure/power.py`) with software transforms (`icd/adapters/`) under a unified orchestration loop.
- Acceptance policies use production-quality gates configured via `docs/Observability_Spec.md` and `docs/Runtime_Memory_Plan.md`, offering reproducible deployment pathways.

### 1.3 Baseline Comparisons & Regression Frameworks
- **Standard benchmarking**: Most memory-layout studies compare a single baseline against a proposed method, often lacking regression safeguards.
- **Benchmark suites**: Works like MLPerf provide comprehensive benchmarks but do not target iterative layout adaptation.

**ICD Novelty Hooks**
- Multi-baseline comparisons are scripted through the regression harness (`docs/Regression_Baseline_Guide.md`, `tests/regression/`) with executable configs (`configs/mock.json`, `configs/bert.json`, `configs/mamba.json`).
- Observability and logging specs (`docs/Observability_Spec.md`, `docs/Completeness_Scoring.md`) tie each experiment to deterministic replay artifacts.

### 1.4 Solver Variants & Adaptors
- **Community detection solvers**: Research demonstrates improvements from Louvain or Leiden algorithms but rarely integrates fallback solvers for stability.
- **Adapter stacks**: Sparse/quantized transformer works typically evaluate adapters in isolation.

**ICD Novelty Hooks**
- ICD exposes solver variants (Louvain + spectral fallback) with configurable quality thresholds (`docs/Correlation_Solver_Design.md`).
- Adapter coverage (sparsity, quantization, KV cache, HDS) is spec’d in `docs/S_Q_K_Adapter_Spec.md` and `icd/adapters/`, enabling cross-transform ablations within the same pipeline.

## 2. Highlighting Novel Contributions
1. **Adaptive re-permutation loop** that fuses graph-driven solvers with runtime measurement gates, providing a deterministic rollback path absent in related work.
2. **Unified regression and baseline infrastructure** enabling simultaneous evaluation of linear, iterative, and hybrid pipelines (`docs/Regression_Baseline_Guide.md`, `tests/runtime/`).
3. **Production-grade observability** that supports NeurIPS reproducibility checklists through documented schemas (`docs/schema/run_config.schema.json`, `docs/Observability_Spec.md`).

## 3. Proposed NeurIPS Paper Structure
### Introduction
- Problem framing: iterative layout re-optimization for memory locality.
- Motivation grounded in the existing workflow (`README.md`, Section “System Overview & Architecture Analysis”).
- Contributions list referencing the adaptive solver stack and regression infrastructure.

### Methodology
1. **Graph Construction & Solvers** – Summarize `icd/core/graph.py`, `icd/core/solver.py`, and the solver design document (`docs/Correlation_Solver_Design.md`).
2. **Transform Adapters** – Detail sparsity, quantization, KV cache, and HDS adapters using `docs/S_Q_K_Adapter_Spec.md` and `icd/adapters/` implementations.
3. **Runtime Orchestration & Acceptance** – Describe state machine (`icd/runtime/orchestrator.py`), gates (`icd/measure/gates.py`), and rollback flow (`docs/rollback_flow.mmd`).

### Experiments
1. **Baselines** – Linear vs iterative comparisons executed via configs in `configs/` and guided by `docs/Regression_Baseline_Guide.md`.
2. **Solver Ablations** – Compare Louvain-only vs fallback-enabled runs leveraging testing hooks in `tests/core/`.
3. **Adapter Studies** – Evaluate combinations of sparsity, quantization, and KV cache using orchestration support documented in `docs/S_Q_K_Adapter_Spec.md`.
4. **Observability & Reproducibility** – Report instrumentation defined in `docs/Observability_Spec.md`, `docs/Repro_Plan.md`, and results storage from `logs/`.

### Discussion & Future Work
- Interpret measurement trade-offs using the cost spec (`docs/Cost_Spec.md`) and runtime plan (`docs/Runtime_Memory_Plan.md`).
- Identify extensions such as native IR passes (planned in `docs/docs_required.md`) and accelerator-specific kernels (`docs/Kernel_Contract.md`).
- Outline deployment considerations informed by `docs/SOP.md` and `docs/Resource_Plan.md`.

## 4. Writing Plan & Asset Mapping
| Paper Section | Repository Assets | Notes |
|---------------|-------------------|-------|
| Introduction | `README.md`, `docs/PRD.md`, `docs/ICD.md` | Use architecture diagrams and hypothesis statements already curated. |
| Methodology | `icd/core/`, `icd/runtime/`, `docs/Correlation_Solver_Design.md`, `docs/S_Q_K_Adapter_Spec.md` | Ensure algorithm descriptions align with solver + adapter specs. |
| Experiments | `configs/*.json`, `tests/regression/`, `docs/Regression_Baseline_Guide.md`, `docs/Completeness_Scoring.md` | Leverage regression harness for multi-baseline comparison tables. |
| Discussion | `docs/Cost_Spec.md`, `docs/Runtime_Memory_Plan.md`, `docs/SOP.md` | Tie operational learnings to broader co-design implications. |
| Appendix | `docs/Repro_Plan.md`, `docs/schema/run_config.schema.json`, `docs/Observability_Spec.md` | Provide reproducibility checklists and schema definitions. |

## 5. Next Steps
1. **Literature citation pass**: Populate references for each subsection, prioritizing graph partitioning, adaptive sparsity, and co-design frameworks.
2. **Experiment script consolidation**: Align regression harness outputs with target NeurIPS tables and figures.
3. **Draft introduction & methodology**: Use this outline to begin the manuscript, referencing existing diagrams and specs for figure/table preparation.
