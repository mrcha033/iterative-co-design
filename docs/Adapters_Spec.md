# S/Q/K Adapters Spec — Inputs/Outputs and Triggers

One-line: Define contracts for sparsity, quantization, and KV-cache transforms with `transform_meta` and `delta_layout` rules.

## Common
- Input: tensor/weights + layout meta.
- Output: transformed tensor + `transform_meta` JSON.
- Failure: no-op + warning, record `quality_delta_pp: NaN`.

## Sparsity
- Params: `{type:"2:4|HDS|unstructured", rate:[0,1]}`
- Trigger: `delta_layout=true` if `rate >= 0.25` (tunable), or structural pattern changes block alignment.
- Quality impact: bound |Δacc| ≤ 0.1pp (PRD), else rollback.

## Quantization
- Params: `{dtype:"int8|fp8", method:"ptq-minmax|..."}`
- Trigger: `delta_layout=true` on dtype change affecting vector width/packing.
- Meta: store per-tensor scale/zero-point (if any), calibration hash.

## KV-cache
- Params: `{block:int, drop:float}`
- Trigger: `delta_layout=true` if `block` crosses cacheline stride or >64.
- Meta: effective capacity, eviction policy tag.

