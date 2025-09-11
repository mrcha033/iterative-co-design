# StableHLO Bridge — A0/V1 PoC

This directory documents the StableHLO bridge proof‑of‑concept for the ICD layout optimization pipeline.

Goal: validate IR plumbing and invariants early (metadata attach + verify) without depending on a native MLIR build in CI. We ship a lightweight, text‑based transformer that mimics A0/V1 behavior and comes with FileCheck‑style tests.

What’s included
- scripts/icd_mlir_opt.py: a tiny "opt" tool with:
  - `--icd-attach-metadata` (A0): attaches `icd.layout_tag = "icd/v1"` and a default `icd.layout_perm` per tensor rank on StableHLO ops.
  - `--icd-verify` (V1): appends an `icd.metrics` marker (e.g., `pi_valid=true`) as a placeholder.
- tests/ir/mlir/*.mlir: 5 skinny cases to exercise attach/idempotency/verify/no‑op/mixed.
- tests/ir/test_filecheck.py: pytest harness that runs the tool and regex‑checks output (FileCheck‑like).

Run locally
- Attach only:
  - `python -m scripts.icd_mlir_opt --icd-attach-metadata tests/ir/mlir/attach_basic.mlir`
- Attach + verify:
  - `python -m scripts.icd_mlir_opt --icd-attach-metadata --icd-verify tests/ir/mlir/attach_verify_mix.mlir`
- Tests:
  - `pytest -q tests/ir`

Roadmap to native passes
- Replace the Python tool with actual MLIR passes:
  - A0: AttachICD (adds layout_tag/layout_perm module/tensor attrs)
  - V1: VerifyICD (perm rank/type checks; annotate `icd.metrics`)
- Provide a minimal CMake in `bridge/stablehlo/` to build against an MLIR toolchain.
- Keep the Python PoC and tests in CI for portability; add a separate job for native builds when GPU runners/toolchains are available.

Invariants (from Pass Design Doc)
- Semantic equivalence is preserved by metadata attach/verify (no functional transforms in this PoC).
- `layout_perm` length = tensor rank; tag is idempotent; verification emits metrics marker.

Links
- docs/Pass_Doc.md: pass DAG and invariants
- tests/README.md: CI gates and layering
