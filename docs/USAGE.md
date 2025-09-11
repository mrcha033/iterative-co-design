# USAGE — ICD CLI & Library

This guide shows how to run the iterative layout optimization pipeline from source, what it produces, and common options. It reflects the current repository behavior and tests.

## Requirements
- Python 3.10+
- No GPU is required for the mock pipeline; Nsight/NVML are optional for profiling/power.

## Quick Start (Mock)

Run linear vs iterative and compare metrics:

```bash
python -m icd.cli.main run -c configs/mock.json --override pipeline.mode=linear   --out runs/mock_linear
python -m icd.cli.main run -c configs/mock.json --override pipeline.mode=iterative --out runs/mock_iter
```

Outputs per run (directory):
- `W.csr.npz` — CSR payload as JSON (`indptr, indices, data, shape, meta`)
- `w.meta.json` — graph meta snapshot
- `perm_before.json`, `stats_before.json` — π0 and stats
- `perm_after.json`, `stats_after.json` — π1 and stats (iterative mode)
- `metrics.json` — latency/L2/EpT (nulls when disabled) + acceptance info
- `report.csv`, `report.html` — simple summaries
- `run.log` — stage events (File-per-line JSON)
- `config.lock.json` — resolved config used for the run
- `power.csv` — when `measure.power_enable=true`

Pair mode (runs baseline+trial and writes a compare verdict):

```bash
python -m icd.cli.main pair -c configs/mock.json --out runs/pair01
```

## Validation & Schema

Validate a config without running, or print the input schema skeleton/JSON Schema file:

```bash
python -m icd.cli.main run -c configs/mock.json --out /tmp/ignore --dry-run
python -m icd.cli.main run --print-schema -c configs/mock.json --out /tmp/ignore
```

Schema file lives at `docs/schema/run_config.schema.json` and mirrors ICD.md.

## Useful Overrides
- `pipeline.mode=linear|iterative`
- `solver.time_budget_s=1.5`
- `solver.refine_steps=500`
- `transform.sparsity.enable=true` (and `transform.sparsity.rate=0.5`)
- `transform.quant.enable=true` (and `transform.quant.dtype=int8`)
- `pipeline.repermute_on_delta=true` (re-permute when transforms change layout, even in linear mode)
- `measure.ncu_enable=false` (no L2 profile)
- `measure.power_enable=false` (no NVML sampling)
- `cache.enable=true` and `cache.cache_dir=.icd_cache`
- `--no-measure` (CLI flag) to skip measurement and report when iterating on solver-only changes
 - `--reuse-perm path/to/perm_before.json` to reuse an existing permutation (skips baseline solve)

Example:

```bash
python -m icd.cli.main run -c configs/mock.json \
  --override pipeline.mode=iterative \
  --override solver.time_budget_s=2.0 \
  --override measure.ncu_enable=false \
  --override cache.enable=true \
  --override cache.cache_dir=.icd_cache \
  --out runs/tuned
```

## Repro Script

Run the smoke scenario and write outputs under `runs/smoke/`:

```bash
bash scripts/repro_smoke.sh
```

## Reuse an Existing Permutation

You can reuse a previously computed permutation to skip the baseline solve:

```bash
# pass a prior run directory (auto-picks perm_before.json)
python -m icd.cli.main run -c configs/mock.json --reuse-perm runs/mock_linear --out runs/reuse_example

# or pass a specific JSON file
python -m icd.cli.main run -c configs/mock.json --reuse-perm runs/mock_linear/perm_before.json --out runs/reuse_example2
```

## Trace Source Example

Run with a tiny JSONL trace (see `tests/data/trace_demo.jsonl`):

```bash
python -m icd.cli.main run -c configs/trace.json --out runs/trace_iter
```

## IR PoC (FileCheck‑like)

You can run the text‑based attach/verify shim on MLIR files:

```bash
python -m scripts.icd_mlir_opt --icd-attach-metadata tests/ir/mlir/attach_basic.mlir
python -m scripts.icd_mlir_opt --icd-attach-metadata --icd-verify tests/ir/mlir/attach_verify_mix.mlir
```

## Configuration Notes
- Configs are JSON in this repo (YAML planned). See `configs/mock.json` for a minimal example.
- Graph sources: `mock` and basic `trace` (JSONL/tuples) are implemented; `pytorch` path is partial.
- Normalization: `sym` and `row` are implemented (`row` is approximate with upper-triangle storage).

## Troubleshooting
- L2/EpT are `null` in `metrics.json` unless enabled and the tools are available.
- Determinism in CI: mock graph/solver are seeded; expect stable results with the same config.
- If NVML is not present, `power.csv` contains a single NaN sample.
- Cache is disabled unless `cache.enable=true` and `cache.cache_dir` is set; cache keys are simplistic and may change between versions.
- Nsight: set `ICD_NCU_CMD` to a shell command that produces JSON to stdout or writes `{out}` to a file (e.g., `ICD_NCU_CMD='nv-nsight-cu-cli ... --export json --export-file {out}'`). When unset or failing, a stub `ncu.json` is written.
 - EpT: when `measure.power_enable=true`, a naive EpT is computed by integrating sampled power over the window and dividing by `pipeline.repeats` (smoke proxy).

## Library Entrypoints (Heads‑up)
- CLI uses `icd.runtime.orchestrator.run`/`run_pair` under the hood.
- Lower‑level APIs live in `icd/core` (graph, solver) and `icd/measure` (reporting).
