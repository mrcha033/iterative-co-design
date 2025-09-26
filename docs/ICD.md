# ICD — Interface Control Document (Update)

## Added Measurement Harness
- Introduced `icd.measure.runner_gpu` providing `BenchmarkConfig` and `benchmark_inference` for built-in benchmarking.
- Updated orchestrator to accept `measure.builtin="benchmark"` which leverages the internal harness when no runner is supplied.
- Latency metrics (`latency_ms_mean`, `latency_ms_p50`, `latency_ms_p95`, `latency_ms_ci95`) now sourced directly from harness results and recorded in `metrics.json`.

## Correlation & Clustering
- New `graph.correlation` configuration block enables activation-based correlation collection. The orchestrator writes correlation artifacts under `runs/<tag>/correlation/`.
- Solver now accepts Louvain/spectral clusters via `solver.clustering` configuration, capturing modularity statistics in `stats_after.json` and `metrics.json`.

## Transform Meta and Metrics
- `metrics.json` includes `correlation` and `clustering` sections when enabled.
- Events emitted: `CORRELATION`, `CLUSTERING`, and enriched `REPERMUTED` stage with cluster counts and modularity.

