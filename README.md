# Iterative HW–SW Co-Design — Layout Re-Optimization (ICD)

This repository implements a mock, testable pipeline for iterative layout optimization (permute → transform(S/Q/K) → re-permute), focusing on determinism, observability, and CI portability.

- Quick start and CLI usage: see `docs/USAGE.md`
- Product requirements and interfaces: see `docs/PRD.md` and `docs/ICD.md`
- Architecture and operating procedures: see `docs/SAS.md` and `docs/SOP.md`
- Graph/Adapters/Kernel/Runtime specs: see `docs/Graph_Construction_Spec.md`, `docs/S_Q_K_Adapter_Spec.md`, `docs/Kernel_Contract.md`, `docs/Runtime_Memory_Plan.md`
- Observability/Contrib/Schema: see `docs/Observability_Spec.md`, `docs/SBOM_Contrib.md`, `docs/schema/run_config.schema.json`
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
