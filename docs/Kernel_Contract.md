# Kernel Contract — Layout, Align, Coalescing

One-line: Specify tensor layout/alignment, vector widths, prefetch/shared mem use, and Nsight mapping.

## Layout & Alignment
- Alignment: ≥32B default (64B preferred on A100/H100).
- Vector width: {8,16,32} elems depending on dtype.
- Stride: forbid alias/overlap; legalize in pass T3.

## Coalescing & Tiling
- Group size g = line_bytes / elem_size (Cost Spec `g`).
- Encourage contiguous within g; reflect in `R_align`.

## Prefetch/Shared Memory
- Use shared tiles when reuse distance < 2g.
- Prefetch ahead distance tunable per kernel.

## Nsight Mapping
- L2 hit: `l2_tex__t_sector_hit_rate.pct` (section-based preferred).
- Latency: wall-clock; Throughput: tokens/s.

