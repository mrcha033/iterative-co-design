# TVM Evaluation Playbook

This document explains how to reproduce the TVM baseline comparisons that are
referenced throughout the ICD documentation set.  It covers three supported
entrypoints:

1. **Standalone TVM export** via `scripts/run_autotvm.py`.
2. **Automatic TVM baselines** from the ICD runtime (`icd.cli run`).
3. **Result comparison** using `scripts/compare_tvm_results.py`.

The goal is to make the evaluation workflow described in the paper and
`docs/TVM_Integration_Guide.md` fully executable from this repository without
requiring any undocumented steps.

## 1. Standalone AutoTVM/Ansor runs

Use `scripts/run_autotvm.py` to export any PyTorch module to TVM, perform
optional AutoTVM/Ansor tuning, verify numerical correctness, and capture
latency statistics.

```bash
# Export BERT-base with 300 tuning trials and measure TVM latency
python scripts/run_autotvm.py icd.experiments.hf:load_bert_base \
    --example-shape 1 128 \
    --target "cuda -arch=sm_80" \
    --tuning-trials 300 \
    --artifacts runs/tvm_bert_base \
    --measure-repeats 200 \
    --measure-warmup 20
```

Key behaviour:

- Artifacts are stored in the target directory (`deploy_lib.tar`,
  `deploy_graph.json`, `deploy_params.bin`).
- `metadata.json` is updated with the tuning configuration, verification result,
  and latency summary (mean, standard deviation, percentiles, confidence
  interval, warmup/repeat counts).
- `summary.json` mirrors this information for quick inspection.
- Latency is collected using `icd.measure.latency.LatencyMeasurer`, matching the
  statistics documented in `docs/API_Reference.md`.

## 2. Automatic TVM baselines inside ICD runs

The ICD runtime can evaluate TVM baselines alongside the standard measurement
pipeline.  Enable this by adding a `measure.tvm_*` block to the run configuration
(or via CLI overrides):

```json
{
  "measure": {
    "tvm_enable": true,
    "tvm_target": "cuda -arch=sm_80",
    "tvm_trials": 3000,
    "tvm_use_ansor": false,
    "tvm_repeats": 200,
    "tvm_warmup": 20,
    "tvm_log": "runs/tvm/bert_auto.log"
  }
}
```

When `tvm_enable` is set, the orchestrator:

1. Reuses the graph/model pair resolved for measurement and exports it to TVM
   via `icd.adapters.tvm_export`.
2. Saves artifacts to `<out_dir>/tvm/` (or `measure.tvm_artifacts_dir` when
   provided).
3. Measures TVM latency with the configured warmup/repeat counts using
   `LatencyMeasurer.measure_callable`.
4. Verifies outputs against the PyTorch model when possible.
5. Stores the result in `metrics.json` under `tvm_baseline` with fields:
   `status`, `latency`, `target`, `tuning_trials`, `use_ansor`,
   `artifacts_dir`, and `verification`.

TVM integration is optional—missing dependencies result in a
`status: "missing_dependencies"` entry rather than failing the run.

### CLI override example

```bash
python -m icd.cli.main run -c configs/bert.json \
    --override measure.tvm_enable=true \
    --override measure.tvm_target="cuda -arch=sm_80" \
    --override measure.tvm_trials=1000 \
    --override measure.tvm_repeats=200 \
    --override measure.tvm_warmup=20 \
    --out runs/bert_vs_tvm
```

## 3. Comparing ICD vs TVM results

After running both the ICD pipeline and the TVM export, use the comparison tool
referenced in the documentation to generate a concise report:

```bash
python scripts/compare_tvm_results.py \
    runs/bert_vs_tvm/metrics.json \
    runs/tvm_bert_base/metadata.json \
    --output runs/bert_vs_tvm/tvm_comparison.json
```

The output contains:

- `icd_latency_ms`: mean latency from the ICD run.
- `tvm_latency_ms`: mean latency from `metadata.json`.
- `speedup_vs_tvm`: ratio (`tvm / icd`) showing the improvement of ICD over TVM.
- `delta_ms`: absolute difference (positive values indicate ICD is faster).
- `tvm_config` and `tvm_verification` fields copied from the metadata for audit
  purposes.

CSV output is also supported by passing a filename that ends with `.csv`.

## Troubleshooting

- **Missing TVM installation**: both the script and runtime emit
  `missing_dependencies` messages without terminating the run.  Install TVM via
  `pip install apache-tvm` or follow the source build instructions in
  `docs/TVM_Integration_Guide.md`.
- **No example inputs**: automatic baselines require example tensors.  Ensure
  your configuration loads a model/example pair via `graph.loader` or the runner
  context.
- **Device mismatches**: inputs are automatically moved to CPU for export.
  Ensure your model supports CPU execution during export or provide a
  CPU-compatible variant for TVM evaluation.

With these pieces in place the repository now fully realises the TVM evaluation
workflow promised in the design documents.
