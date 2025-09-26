#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT=${1:-runs/codesign}
CORR_CFG=${2:-configs/mock.json}

python -m icd.cli.main run -c "$CORR_CFG" \
  --override pipeline.mode=iterative \
  --override graph.correlation.enable=true \
  --override solver.clustering.enable=true \
  --override measure.builtin=benchmark \
  --out "${OUT_ROOT}/iterative"

python -m icd.cli.main run -c "$CORR_CFG" \
  --override pipeline.mode=linear \
  --override measure.builtin=benchmark \
  --out "${OUT_ROOT}/linear"

echo "Codesign repro completed. Outputs in ${OUT_ROOT}/{iterative,linear}."
