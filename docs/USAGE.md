# USAGE — ICD CLI & Library

This guide shows how to run the iterative layout optimization pipeline from source, what it produces, and common options. It reflects the current repository behavior and tests.

## CPU-Only Quickstart (5 minutes)

This quickstart walks through reproducing a latency improvement on a toy
Mamba layer in under five minutes using CPU-only resources.

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 2. Generate synthetic correlation data

```bash
python - <<'PY'
import torch
from icd.graph.streaming_correlation import compute_streaming_correlation
from icd.graph import CorrelationConfig

rng = torch.Generator().manual_seed(7)
samples = [torch.randn(64, 32, generator=rng) for _ in range(8)]
cfg = CorrelationConfig(threshold=0.05, normalize="sym")
csr, meta = compute_streaming_correlation(samples, feature_dim=32, cfg=cfg)
print("Correlation nnz:", csr.nnz())
print("Meta:", meta)
PY
```

### 3. Fit a permutation

```bash
python - <<'PY'
from icd.core.cost import CostConfig, eval_cost
from icd.core.solver import fit_permutation
from icd.core.graph import CSRMatrix
from icd.graph.streaming_correlation import compute_streaming_correlation
from icd.graph import CorrelationConfig
import torch

rng = torch.Generator().manual_seed(13)
samples = [torch.randn(64, 32, generator=rng) for _ in range(6)]
cfg = CorrelationConfig(threshold=0.05, normalize="sym")
csr, _ = compute_streaming_correlation(samples, feature_dim=32, cfg=cfg)
pi, stats = fit_permutation(csr, time_budget_s=0.1, refine_steps=64)
identity = list(range(csr.shape[0]))
cfg_cost = CostConfig()
baseline = eval_cost(csr, identity, identity, cfg_cost)
improved = eval_cost(csr, pi, pi, cfg_cost)
print("Permutation:", pi[:10], "...")
print("Cost delta:", baseline["J"] - improved["J"])
print("Stats:", stats)
PY
```

### 4. Next steps

- Replace the synthetic samples with captures from your own model using
  `icd.graph.collect_correlations`.
- Export the permutation via the CLI: `icd export --config configs/mock.json`.
- Validate determinism with `python scripts/check_cuda_env.py`.

---

### Quantization/Permutation strategies

The following override sets make the three supported strategies explicit (combine with a config such as `configs/mamba.json` that enables a quantizable loader):

* **Quant → Permute** (build `W` from quantized weights):

  ```bash
  --override transform.quant.enable=true \
  --override pipeline.transform_stage=pre \
  --override pipeline.post_transform_repermute=never
  ```

* **Permute → Quant** (legacy order, skip re-permutation):

  ```bash
  --override transform.quant.enable=true \
  --override pipeline.transform_stage=post \
  --override pipeline.post_transform_repermute=never
  ```

* **Permute → Quant → Re-permute** (quantize after the first permutation, then refine):

  ```bash
  --override transform.quant.enable=true \
  --override pipeline.transform_stage=post \
  --override pipeline.post_transform_repermute=always
  ```

## Requirements
- Python 3.10+
- No GPU is required for the mock pipeline; Nsight/NVML are optional for profiling/power.
- For executable HuggingFace experiments install:
  - `pip install -e .[experiments]`
  - `pip install torch --index-url https://download.pytorch.org/whl/cpu` (pick the wheel that matches your CUDA/cuDNN stack if targeting GPU)
  - `pip install mamba-ssm` (required for the Mamba configs)

### PyTorch troubleshooting on macOS

Several contributors have reported an `ImportError` similar to the
following when running `pytest` on Apple Silicon machines:

```
ImportError: dlopen(.../torch/_C.cpython-313-darwin.so, 0x0002): Symbol not found: __ZN6google8protobuf5Arena18CreateMaybeMessageIN10onnx_torch10GraphProtoEJEEEPT_PS1_DpOT0_
```

The error is triggered when a virtual environment inadvertently links
against a Homebrew installation of PyTorch (or libtorch) instead of the
wheel that was installed in the environment. To resolve the issue, remove
any conflicting Homebrew PyTorch packages and reinstall PyTorch directly
from the official wheels inside your virtual environment:

```bash
brew uninstall pytorch libtorch  # remove conflicting libraries
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu
```

The CPU wheels are compatible with Apple Silicon for the purposes of the
unit tests in this repository. If you intend to run GPU workflows, replace
the `cpu` wheel URL with the variant that matches your CUDA toolkit.

Install (PyPI, planned):

```bash
pip install repermute   # distribution name; import and CLI are `icd`
```

## Real Model Experiments (HuggingFace)

With the dependencies above installed you can run the executable configs that load real models:

```bash
# BERT base sequence classification (CPU defaults shown)
python -m icd.cli.main run -c configs/bert.json --out runs/bert_iter

# Mamba-130M causal LM (requires mamba-ssm wheel)
python -m icd.cli.main run -c configs/mamba.json --out runs/mamba_iter
```

Both configs rely on two new fields:

- `graph.loader` / `loader_kwargs`: dotted path + kwargs to instantiate a PyTorch model and example inputs for graph construction.
- `pipeline.runner` / `runner_context`: dotted path + context for the measurement loop (reuses the same loader by default).
- Iterative guard: enable at least one transform (S/Q/K) **or** opt into the correlation path via `graph.correlation.enable=true`. The reference configs ship with correlation enabled so they satisfy the guard out of the box.

Set `"device": "cuda"` in the loader kwargs to target a GPU (ensure the correct PyTorch wheel is installed).
When running on shared machines consider setting `HF_HOME`/`TRANSFORMERS_CACHE` to control HuggingFace download paths.

## Vision & Graph Experiments (TorchVision / PyG)

Additional configs cover computer-vision and graph workloads. Install the optional dependencies first:

```bash
pip install torchvision
pip install torch-geometric ogb
```

TorchVision will download pretrained checkpoints on first use (set `TORCH_HOME` to control the cache path). The `ogbn-arxiv`
dataset is fetched through `ogb`; use `OGB_CACHE_ROOT` to relocate the dataset cache if needed.

Run the configs once dependencies are available:

```bash
# ResNet-50 vision trace (mock runner context)
python -m icd.cli.main run -c configs/resnet50.json --out runs/resnet50_iter

# GCN on OGBN-ArXiv (graph loader + mock runner context)
python -m icd.cli.main run -c configs/gcn_arxiv.json --out runs/gcn_iter
```

Both configs reuse the `graph.loader`/`pipeline.runner_context.model_loader` pattern introduced for HuggingFace models. Swap the
mock runner for your measurement harness when collecting real latency or accuracy numbers.

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
- `pipeline.runner=icd.runtime.runners.mock_inference`
- `pipeline.runner_context={"tokens":512}`
- `graph.loader=icd.experiments.hf.load_hf_sequence_classifier`
- `graph.loader_kwargs={"model_name":"bert-base-uncased","sequence_length":128}
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

Makefile shortcuts (optional):

```bash
make repro-smoke         # bash scripts/repro_smoke.sh
make repro-codesign      # bash scripts/repro_codesign.sh
make pair                # pair run to runs/pair01
make test                # quick unit+integration+ir sweep
make schema              # print input JSON Schema
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
- Configs are JSON in this repo (YAML planned). See `configs/mock.json` for a minimal example, `configs/bert.json` / `configs/mamba.json` for executable experiments.
- Graph sources: `mock`, `trace` (JSONL/tuples), and `pytorch` (via dotted loaders) are supported.
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