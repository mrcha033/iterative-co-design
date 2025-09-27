# Iterative HW–SW Co-Design — Layout Re-Optimization (ICD)

This repository implements a mock, testable pipeline for iterative layout optimization (permute → transform(S/Q/K) → re-permute), focusing on determinism, observability, and CI portability.

- Quick start and CLI usage: see `docs/USAGE.md`
- Product requirements and interfaces: see `docs/PRD.md` and `docs/ICD.md`
- Architecture and operating procedures: see `docs/SAS.md` and `docs/SOP.md`
- Graph/Adapters/Kernel/Runtime specs: see `docs/Graph_Construction_Spec.md`, `docs/S_Q_K_Adapter_Spec.md`, `docs/Kernel_Contract.md`, `docs/Runtime_Memory_Plan.md`
- Observability/Contrib/Schema: see `docs/Observability_Spec.md`, `docs/SBOM_Contrib.md`, `docs/schema/run_config.schema.json`
- Executable experiments: `configs/mock.json` (mock), `configs/bert.json` (HF BERT-base), `configs/mamba.json` (HF Mamba-130M)
- IR bridge PoC: see `bridge/README.md`
- Contribution guidelines: see `CONTRIBUTING.md`

Run tests:

```bash
pytest -q tests/unit
pytest -q tests/integration
pytest -q tests/ir
```

Quick start (mock):

```bash
python -m icd.cli.main run -c configs/mock.json --override pipeline.mode=iterative --out runs/mock_iter
```

Trace example:

```bash
python -m icd.cli.main run -c configs/trace.json --out runs/trace_iter
```

Repo layout:

- `icd/core`: graph, cost, solver
- `icd/runtime`: orchestrator and compare logic
- `icd/measure`: latency, ncu parsing, power sampling, reports
- `icd/adapters`: sparsity/quant/kv stubs
- `scripts/`: utilities (e.g., `repro_smoke.sh`, IR PoC tool)
- `tests/`: unit/integration/IR tests

Notes

- Most features are CI‑portable mocks with deterministic behavior; optional hooks enable real profiling/power.
- See USAGE for flags like `--dry-run`, `--print-schema`, `--no-measure`, and `--reuse-perm`.

## Install

From source (development):

```bash
pip install -e .[dev]
```

PyPI (planned):

```bash
# distribution name: repermute ; import and CLI remain `icd`
pip install repermute
```

Why “repermute”?
- The project centers on re‑permutation after state transforms (S/Q/K) to improve memory locality. The distribution name `repermute` is descriptive and discoverable on PyPI, while the import package and CLI remain the short, memorable `icd` to reflect the broader “Iterative HW–SW Co‑Design” scope.

## Reproducing Results (End‑to‑End)

This section shows how to reproduce the core experiment (linear vs iterative) from a fresh checkout, what artifacts to expect, and how to enable optional profiling and power logging.

Prereqs
- Python 3.10+
- GPU not required for the mock pipeline. Nsight Compute and NVML are optional for L2/EpT.

1) Smoke Reproduction (recommended first)
- Run the prewired smoke script to generate linear and iterative runs:
  - `bash scripts/repro_smoke.sh`
- Outputs: `runs/smoke/{linear,iter}/` with:
  - `W.csr.npz`, `w.meta.json` — co-access graph snapshot
- `perm_before.json`, `stats_before.json` — baseline permutation + stats
- `perm_active.json` — currently active permutation after acceptance/rollback
  - `perm_after.json`, `stats_after.json` — iterative permutation + stats
  - `metrics.json` — latency/L2/EpT (nulls if disabled), acceptance gate info
  - `report.{html,csv}`, `run.log`, `config.lock.json`

2) Linear vs Iterative (pair mode)
- Runs both modes and writes a verdict:
  - `python -m icd.cli.main pair -c configs/mock.json --out runs/pair01`
- Inspect:
  - `runs/pair01/compare.json` — acceptance decision and deltas
  - `runs/pair01/{linear,iter}/metrics.json` — per‑run metrics and acceptance (trial is updated with verdict)

3) Single Run with Overrides
- Iterative (mock, no external tools):
  - `python -m icd.cli.main run -c configs/mock.json --override pipeline.mode=iterative --out runs/iter`
- Linear baseline:
  - `python -m icd.cli.main run -c configs/mock.json --override pipeline.mode=linear --out runs/linear`
- Replace the mock runner with your own inference loop by setting `--override pipeline.runner="my.module:runner"` (see USAGE for details).

4) Executable Experiments (BERT / Mamba)
- Install HuggingFace dependencies (CPU example):
  - `pip install -e .[experiments]`
  - `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Run BERT-base sequence classification (library+runner load real model):
  - `python -m icd.cli.main run -c configs/bert.json --out runs/bert_iter`
- Run Mamba-130M causal LM experiment (requires `mamba-ssm` wheel):
  - `pip install mamba-ssm`
  - `python -m icd.cli.main run -c configs/mamba.json --out runs/mamba_iter`
See `docs/USAGE.md` for GPU notes and runner customization.

4) Trigger Re‑permute from Transforms in Linear Mode
- Demonstrates adapter metadatas and delta‑layout trigger:
  - `python -m icd.cli.main run -c configs/mock.json \
    --override pipeline.mode=linear \
    --override pipeline.repermute_on_delta=true \
    --override transform.sparsity.enable=true \
    --override transform.sparsity.rate=0.5 \
    --out runs/linear_delta`
- Expect `perm_after.json` and `metrics.json.transform_meta.delta_layout=true`.

5) Optional: Profiling and Power
- Nsight Compute (L2 hit): set an external command in `ICD_NCU_CMD` that writes JSON or outputs JSON to stdout. Example (stubbed for CI):
  - `export ICD_NCU_CMD='nv-nsight-cu-cli --section MemoryWorkloadAnalysis --section MemoryChart --export json --export-file {out} ./your_runner'`
  - Add `--override measure.ncu_enable=true` to your run.
- NVML power (EpT): enable and set sample rate:
  - `--override measure.power_enable=true --override measure.power_sample_hz=10`
- Results are written to `ncu.json` and `power.csv`; `metrics.json` is updated with `l2_hit_pct` and `ept_j_per_tok` when available.

6) Caching and Reuse
- Enable cache to reuse the baseline permutation in subsequent runs:
  - `--override cache.enable=true --override cache.cache_dir=.icd_cache`
- Or reuse a specific prior permutation:
  - `--reuse-perm runs/linear` (auto‑picks `perm_before.json`) or a direct file path.

7) Determinism & Config Validation
- Print and validate the input schema without running:
  - `python -m icd.cli.main run --print-schema -c configs/mock.json --out /tmp/ignore`
  - `python -m icd.cli.main run --dry-run -c configs/mock.json --out /tmp/ignore`
- Determinism is seed‑driven; `graph.mock.seed` and `solver.rng_seed` fix mock graph and solver behavior respectively.

8) Interpreting Acceptance
- The mock pipeline computes a cost `J` (from `docs/Cost_Spec.md`) and uses a simple ΔJ‑based acceptance with rollback semantics in iterative mode. Pair mode adds relative latency/L2 deltas in `compare.json`.
- PRD gates and SOP are documented in `docs/PRD.md` and `docs/SOP.md`; CI uses relaxed smoke thresholds.

9) Full Test Sweep
- Unit: `pytest -q tests/unit`
- Integration: `pytest -q tests/integration`
- IR PoC: `pytest -q tests/ir`

Tips
- For quick iteration on solver changes, add `--no-measure` to skip report generation.
- To control output formats, set `--override report.formats=["html"]` or `["csv"]`.

## Makefile Shortcuts

If you prefer one-liners, a Makefile is provided:
- `make repro-smoke`: run smoke scenario (`scripts/repro_smoke.sh`)
- `make pair`: run baseline+trial and write `runs/pair01`
- `make test`: run unit, integration, and IR tests
- `make schema`: print the input JSON Schema
- `make clean-runs`: remove `runs/` and `.icd_cache/`

## License

Apache License 2.0 — see `LICENSE`.

## Citation

If you use this software, please cite:

```
Yunmin Cha. Iterative HW–SW Co-Design — Layout Re-Optimization (ICD), 2025.
Software. https://github.com/mrcha033/iterative-co-design
```

See `CITATION.cff` for a citation file (CFF) that GitHub can render and export to BibTeX.
