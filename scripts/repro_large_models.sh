#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT=${1:-runs/large_models}
mkdir -p "${OUT_ROOT}"

run_pair() {
  local name="$1"
  local config="$2"
  echo "=== Running ${name} (linear baseline) ==="
  python -m icd.cli.main run -c "${config}" --override pipeline.mode=linear --out "${OUT_ROOT}/${name}_linear"
  echo "=== Running ${name} (iterative) ==="
  python -m icd.cli.main run -c "${config}" --override pipeline.mode=iterative --out "${OUT_ROOT}/${name}_iter"
  echo "=== Metrics delta for ${name} ==="
  python scripts/validate_results.py "${OUT_ROOT}/${name}_linear" "${OUT_ROOT}/${name}_iter" || echo "Validation thresholds not met; inspect metrics manually."
}

run_pair bert_large configs/bert_large.json
run_pair mamba_2p8b configs/mamba_2p8b.json

echo "Large-model repro run completed. Outputs in ${OUT_ROOT}/{bert_large_*,mamba_2p8b_*}."
