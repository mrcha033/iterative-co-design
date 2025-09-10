**Self‑Review Summary for Newly Added Specs**

- Cross‑checked spec requirements with current code paths:
  - Graph construction spec aligns with `icd/core/graph.py` and `graph_pytorch.py`; noted open items (row norm, trace schema).
  - S/Q/K adapter spec mirrors stubs in `icd/adapters/*`; defines delta_layout triggers consistent with code.
  - Kernel contract forward‑looking, maps Nsight metrics to measurement layer intent.
  - Runtime/memory plan extends orchestrator with caching/rollback policy without breaking current tests.
  - Observability matches `metrics.json` and `run.log` fields already emitted.
  - SBOM & Contrib integrates cleanly with existing CI; does not alter runtime behavior.
- Risks/Assumptions:
  - Some parts are aspirational (trace loader, CUDA kernels); explicitly marked.
  - File naming follows snake_case where prior docs used mixed case; link updates may be needed.

