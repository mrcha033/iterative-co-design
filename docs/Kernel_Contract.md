**Kernel Contract (Layout-Oriented, V1)**

- **Scope:** Define the contract between layout decisions (π, transform_meta) and kernels. Includes layout/alignment/tiling/coalescing requirements, prefetch/shared usage expectations, and Nsight metrics mapping used for validation.

**Layout & Indexing**
- **Permutation π:** Kernels must accept a row/column permutation applied to tensor layouts; IR bridge attaches `icd.layout_perm=[...]` attributes (PoC) with `icd.layout_tag="icd/v1"`.
- **Stride/Align:**
  - Minimum alignment: 16 bytes for vectorized loads/stores; 32 bytes preferred for `fp8/int8`.
  - Stride constraints: last-dimension contiguous preferred; if permuting breaks this, kernels must switch to gather/scatter path with coalescing fallback.
- **Tiling:** Parameterize tile shapes to respect vector width and shared-memory size (e.g., 128×K for matmul-like ops). π changes may alter optimal tiles; kernels must expose tile selection via config.

**Coalescing & Memory Access**
- **Global:** 128‑byte coalesced segments expected for bulk loads; misaligned cases penalized and surfaced via `icd.metrics`/Nsight.
- **Shared Memory:**
  - Use when stride misalignment exceeds 2× cacheline; transpose tiles to recover coalescing.
  - Bank‑conflict avoidance required (pad to multiple of 32 as needed).
- **Prefetch:**
  - Enable prefetch distance tuned per kernel; if π changes increase stride, increase prefetch distance or switch to double‑buffering.

**Nsight Metric Map**
- **Latency proxy:** `sm__cycles_elapsed.avg` (contextual only).
- **L2 hit:** `lts__t_sectors_srcunit_tex_op_read_lookup_hit_rate.pct` (or MemoryWorkloadAnalysis L2 metrics parsed by `ncu_wrapper`).
- **DRAM bytes:** `dram__bytes.sum`.
- **SM busy:** `sm__throughput.avg.pct_of_peak_sustained_elapsed`.
- Mapping must be stable across kernels to compare pre/post π.

**Error Handling & Fallbacks**
- If align/stride constraints violated (P‑02), IR bridge must either legalize by inserting transposes/copies (minimized) or mark kernel as slow‑path; never proceed with undefined behavior.
- Quality gate: if kernel can’t meet minimum coalescing, emit diagnostic in `icd.metrics` and allow runtime rollback.

**Configuration Interface**
- Expose kernel params via IR attributes or runtime knobs: `vec_width`, `tile_m/n/k`, `prefetch_dist`, `use_shared`, `accum_dtype`.
- Record chosen params and their rationale into run log for RCA.

**Self‑Review**
- Contract is forward‑looking; current repo does not host CUDA kernels. The Nsight metric mapping aligns with `measure/l2_ncu.py` intent and bridges well with the `icd.metrics` placeholder produced by IR PoC.

