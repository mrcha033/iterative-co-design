# Graph Construction Spec — Co-access Weight Matrix W

One-line: Define trace/mock inputs, normalization, noise/filtering, CSR bounds, and determinism rules for W.

## Input schema
- source: `trace | mock`
- trace (TBD):
  - record: `{op, tensor, i, j, reuse_distance, bytes, kernel, ts}`
  - merge: producer–consumer paths, reuse-distance buckets → weights
  - constraints: NaN/negatives forbidden, time windowed aggregation
- mock:
  - `{d:int, blocks:int, noise:float, seed:int, normalize:"sym|row|none"}`

## Normalization
- Default: `sym` (D^{-1/2} W D^{-1/2} approx.)
- Row: optional alternative; `none` only for debugging.

## Noise/filtering
- Jitter: bounded |ε| ≤ noise; zero out w<ε.
- Diagonal excluded; store upper triangle only.

## CSR bounds/validation
- `nnz <= 0.05 * d^2` (mock). Trace-bound for real.
- JSON `.npz` payload: `{indptr, indices, data, shape:[d,d], meta}`
- Determinism: same config → identical payload hash.

