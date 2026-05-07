# BF16 Tensor-Core GEMM on RTX 4090: Design and Results

## Abstract

We implement a series of BF16 GEMM kernels using CUDA tensor cores, reaching
**~97% of cuBLAS BF16** at N=4096 and **~96% at N=8192**, building directly
on the template established by the FP32 cuda-core kernel (s6). The progression
has three stages: TC1 replaces the scalar FMA loop with WMMA; TC2 replaces the
WMMA API with raw PTX (`ldmatrix` + `mma.sync`) and adds XOR swizzling on the B
tile to eliminate shared-memory bank conflicts; TC2b extends this to the A tile
as well. The template itself — tile sizes, double-buffered `cp.async`, warp
partitioning, autotuner — is inherited intact from s6.

---

## 1. Inherited Template

The s6 kernel (see `fp32_matmul_report_minimal.md`) established the following
structure, which all tensor-core variants carry forward unchanged:

- **CTA tile** `BM × BN` partitioned across warps in a 2D inter-warp grid
  `WARP_M × WARP_N = (NUM_WARPS/2) × 2`.
- **Double-buffered async prefetch** via `cp.async` / `__pipeline_*`: while the
  current smem tile is being computed, the next tile is DMA'd from global memory.
- **Dynamic shared memory** (`extern __shared__`): allows tiles exceeding the 48 KB
  static limit; actual size passed at launch via `smem_bytes`.
- **Template parameters** `(BM, BN, BK, NUM_WARPS)` autotuned empirically per
  problem size (2 warmup + 3 timed runs per config, best cached in memory).

The UNROLL parameter from s6 is dropped: the tensor-core inner loop has no
scalar FMA body to unroll — the hardware executes the matrix operation as a
single instruction, so unroll adds no scheduling benefit.

---

## 2. Kernel Variants

### TC1 — WMMA API

TC1 replaces the per-thread scalar FMA register tile with warp-level WMMA
fragments. Each warp holds `WM_TILES × WN_TILES` float32 accumulator fragments
(where `WM_TILES = WARP_TILE_M/16`, `WN_TILES = WARP_TILE_N/16`), and computes
using `wmma::load_matrix_sync` + `wmma::mma_sync`.

The COMPUTE_TILE macro uses outer-product order: for each k-step (`_kk`), all A
fragments for the warp are loaded first, then all B fragments, then the full
`WM_TILES × WN_TILES` mma grid. This loads each B fragment once per k-step
rather than `WM_TILES` times.

Output is written directly from the float32 accumulator fragment layout
(no scratch buffer): thread `t`, element `e` maps to:

```
row = (t/4) + row_off[e],  row_off = {0,0,8,8,0,0,8,8}
col = (t%4)*2 + col_off[e], col_off = {0,1,0,1,8,9,8,9}
```

**Limitation:** the WMMA API is opaque to the compiler — `wmma::fragment` objects
are treated as unanalyzable blobs, causing conservative register allocation. At
BM=128, BN=128, BK=16, NW=4 (the best TC1 config at most sizes), the compiler
allocates ~192 registers/thread → only 2 CTAs/SM → ~16% occupancy.

---

### TC2 — Raw PTX + B-tile XOR swizzle

TC2 replaces WMMA with raw PTX, exposing the register layout to the compiler:

| Operation | PTX instruction | Purpose |
|-----------|----------------|---------|
| A load | `ldmatrix.sync.aligned.x4.m8n8` | 16×16 bf16 from smem → 4 uint32 per thread |
| B load | `ldmatrix.sync.aligned.x2.m8n8.trans` | 16×8 bf16 from smem → 2 uint32 per thread, transposed to col-major register layout |
| Compute | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | one 16×8 output tile per call |

The native mma width is 16×8 (not 16×16), so `WN_TILES = WARP_TILE_N/8` and
each `(mt, nt, kk)` triple maps to exactly one `mma.sync` call. Accumulators
shrink from 8 to 4 floats per tile. The write-back offsets simplify accordingly:

```
row_off = {0, 0, 8, 8},  col_off = {0, 1, 0, 1}
gc_base = block_col + warp_col * WARP_TILE_N + nt * 8
```

**B-tile XOR swizzle.** With B stored row-major (`[BK][BN]`), all 16 k-rows
covered by one `ldmatrix.x2.trans` call start at the same set of banks
(row stride = `BN * 2` bytes; for BN=128, stride = 256 bytes = 64 banks mod 32 = 0
→ every row aliases to bank 0). This creates a 16-way conflict.

The XOR swizzle permutes 8-bf16 column chunks on writes and undoes the
permutation on reads:

```
physical_chunk = logical_chunk ^ (row % B_SWZ),   B_SWZ = BN/8
```

The 16 consecutive k-rows each land in a distinct column chunk → at most 2-way
conflict (the hardware minimum for 16×16-byte loads across 32 banks).

**Occupancy improvement.** Raw PTX lets the compiler see the full register
dataflow. At the same BM=64, BN=128, BK=16, NW=4 config, TC2 uses ~122
registers/thread vs TC1's ~192, enabling 4 CTAs/SM → ~31% occupancy — a 2×
improvement.

---

### TC2b — A-tile XOR swizzle

TC2b adds the same XOR treatment to the A tile.

**A-tile conflict analysis.** With A stored `[BM][BK]`, the row stride is
`BK * 2` bytes. The bank-conflict period (rows until the pattern repeats) is:

```
A_SHIFT = 32 banks / (BK * 2 / 4 banks-per-byte) = 64 / BK
```

| BK | row stride | bank period | conflict degree (16-row ldmatrix.x4) |
|----|-----------|-------------|--------------------------------------|
| 16 | 32 bytes  | 4 rows      | 4-way (rows 0,4,8,12 all → bank 0)  |
| 32 | 64 bytes  | 2 rows      | 8-way (rows 0,2,4,...,14 → bank 0)  |

The A swizzle uses the same XOR formula with `A_SWZ = BK/8` chunks:

```
physical_chunk = logical_chunk ^ ((row / A_SHIFT) % A_SWZ)
```

For BK=16 (A_SWZ=2, A_SHIFT=4): rows 0–3 get key 0, rows 4–7 get key 1,
rows 8–11 get key 0, rows 12–15 get key 1. Bank layout after swizzle:

| Rows | physical chunk | banks (left-half thread) |
|------|---------------|--------------------------|
| 0–3  | 0             | 0–3, 8–11, 16–19, 24–27 |
| 4–7  | 1             | 4–7, 12–15, 20–23, 28–31 |
| 8–11 | 0             | 0–3, … (2-way with 0–3) |
| 12–15| 1             | 4–7, … (2-way with 4–7) |

Result: 4-way → 2-way for BK=16; 8-way → 2-way for BK=32.

The COMPUTE_TILE A address un-swizzles with the same formula:

```c
const int _lg   = _kk * 2 + (lane / 16);          // logical chunk
const int _phys = _lg ^ ((_ar / A_SHIFT) % A_SWZ); // physical chunk
ldmatrix_x4(..., &A_shared[buf][_ar][_phys * 8]);
```

---

## 3. Template and Parameters

All three variants share the same template signature `(BM, BN, BK, NUM_WARPS)`
and the same config search space:

**Shared memory** (double-buffered, no padding needed with swizzling):
```
A_shared[2][BM][BK]   (BM*BK*2 bytes per buffer)
B_shared[2][BK][BN]   (BK*BN*2 bytes per buffer)
Total: (2*BM*BK + 2*BK*BN) * 2 bytes
```

**Tunable parameters:**

| Parameter | Candidates | Notes |
|-----------|-----------|-------|
| BM | 64, 128, 256 | CTA row tile |
| BN | 64, 128, 256 | CTA col tile |
| BK | 16, 32 | k-step; must be multiple of 16 (mma k-dim) |
| NUM_WARPS | 4, 8 | 4 → 2×2 inter-warp (128 threads); 8 → 4×2 (256 threads) |

**Constraints:**
- `smem(BM, BN, BK) ≤ 100352` bytes (98 KB, hardware max with `cudaFuncAttributeMaxDynamicSharedMemorySize`)
- `BM × BN ≤ 4096 × NUM_WARPS` (keeps per-warp tile from overflowing registers)

Total valid configs: **28** (fewer than s6's 112 because UNROLL is absent and
BK only needs to be a multiple of 16 rather than also driving the unroll schedule).

---

## 4. Results

**Hardware:** NVIDIA RTX 4090 (Ada Lovelace, sm_89), 128 SMs.  
**Precision:** BF16 inputs, FP32 accumulators, BF16 output.  
**Peak BF16 tensor-core throughput:** 330 TFLOPS (dense).

### Performance

| Size | TC1 (TFLOPS) | TC2 (TFLOPS) | TC2b (TFLOPS) | cuBLAS BF16 (TFLOPS) | TC2b / cuBLAS |
|------|-------------|-------------|--------------|----------------------|---------------|
| 2048 | 103.6       | 107.5       | **110.4**    | 135.3                | 82%           |
| 4096 | 123.4       | 129.2       | **131.2**    | 134.8                | **97%**       |
| 8192 | 133.6       | 137.2       | **138.5**    | 144.1                | **96%**       |

TC2 over TC1 reflects the occupancy improvement from raw PTX (2× more CTAs/SM).
TC2b over TC2 reflects reduced A-tile bank conflicts, with the most visible
gain at BK=32 (where A-tile conflicts were 8-way before swizzling).

### Best Config Selected by Autotuner

| Size | TC1 best | TC2 best | TC2b best |
|------|----------|----------|-----------|
| 2048 | BM=128, BN=128, BK=16, NW=4 | BM=64, BN=128, BK=16, NW=4 | BM=64, BN=128, BK=32, NW=4 |
| 4096 | BM=128, BN=128, BK=16, NW=4 | BM=64, BN=128, BK=16, NW=4 | BM=64, BN=128, BK=32, NW=4 |
| 8192 | BM=128, BN=128, BK=16, NW=4 | BM=128, BN=128, BK=32, NW=4 | BM=128, BN=128, BK=32, NW=4 |

TC1 consistently prefers BK=16, NW=4 — the register pressure from WMMA
fragments penalises larger tiles. TC2/TC2b shift toward BK=32 once the A-tile
swizzle removes the associated bank-conflict penalty.

---

## 5. Conclusion

Three BF16 tensor-core kernels, built by layering optimizations onto the s6
FP32 template:

- **TC1 (WMMA):** drop-in tensor-core upgrade — warp tiling and double-buffer
  prefetch unchanged, scalar FMA body replaced by `wmma::mma_sync`.
- **TC2 (raw PTX + B swizzle):** raw PTX exposes register layout → 2× occupancy;
  XOR swizzle on B eliminates 16-way smem bank conflicts.
- **TC2b (A+B swizzle):** same XOR treatment on A, tuned to the row-stride-
  dependent conflict period (`A_SHIFT = 64/BK`), reducing 4-way (BK=16) or
  8-way (BK=32) A-tile conflicts to 2-way.
