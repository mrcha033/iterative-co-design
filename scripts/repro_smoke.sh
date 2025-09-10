#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT=${1:-runs/smoke}

python -m icd.cli.main run -c configs/mock.json --override pipeline.mode=linear   --out "${OUT_ROOT}/linear"
python -m icd.cli.main run -c configs/mock.json --override pipeline.mode=iterative --out "${OUT_ROOT}/iter"

echo "Smoke run completed. Outputs in ${OUT_ROOT}/{linear,iter}".

