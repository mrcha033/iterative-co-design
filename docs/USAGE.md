# ICD Usage (Developer Quick Start)

- Switch graph source:
  - Mock (CI-safe):
    `python -m icd.cli.main run -c configs/mock.json --out runs/mock_iter`
  - PyTorch (trace v0.9, last-dim heuristic):
    Provide `graph.source="pytorch"` and pass `model` and `example_inputs` to `build_w(...)` in code, or wire a loader.

- Inspect W artifacts (when `source=pytorch`):
  - `runs/<tag>/W.csr.npz` (JSON payload)
  - `runs/<tag>/w.meta.json` (D, nnz, normalize, band kernel, op weights, seed, trace_hash)
  - `runs/<tag>/w.ops.json` (used_ops, skipped_ops_count, hops, notes)

- Notes:
  - v0.9 uses last-dim feature heuristic and banded kernel; attention-aware mapping arrives next.
  - Enable measurement locally by setting `measure.ncu_enable=true` or `measure.power_enable=true` and providing tool access.

