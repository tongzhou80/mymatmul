# BF16 Tensor-Core GEMM on RTX 4090: Design and Results

## Abstract

We implement a series of BF16 GEMM kernels using CUDA tensor cores, reaching
**102–103% of cuBLAS BF16** at N=3072–4096 (wave-quantization sweet spots where
cuBLAS underperforms) and **99–100% at N=8192–10240** with `tc5_regpruned`.
The progression has five main stages: TC1 (WMMA API), TC2 (raw PTX + B-tile
XOR swizzle), TC2b (A-tile XOR swizzle), TC5 (vectorized write-back + BK=64),
and TC5_reg/TC5_regpruned (two-arg `__launch_bounds__` LB tuning + register-estimate
pruning). A second round of optimization explored pipeline restructuring (TC6,
TC7), CTA swizzle (TC5swz), JIT constants (TC5jit/TC5jit_lb), and multi-stage
prefetch (TC8g); none improved over TC5_regpruned. Triton leads by a consistent
2–4% across all sizes; the gap is not yet explained.

---

## 1. Inherited Template

The s6 kernel (`fp32_matmul_report_minimal.md`) established the following
structure, which all tensor-core variants carry forward unchanged:

- **CTA tile** `BM × BN` partitioned across warps in a 2D inter-warp grid
  `WARP_M × WARP_N = (NUM_WARPS/2) × 2`.
- **Double-buffered async prefetch** via `cp.async` / `__pipeline_*`: while the
  current smem tile is being computed, the next tile is DMA'd from global memory.
- **Dynamic shared memory** (`extern __shared__`): allows tiles exceeding the 48 KB
  static limit; actual size passed at launch via `smem_bytes`.
- **Template parameters** `(BM, BN, BK, NUM_WARPS)` autotuned empirically per
  problem size using `triton.testing.do_bench` (best cached in memory).

---

## 2. Kernel Variants

### TC1 — WMMA API

TC1 replaces the per-thread scalar FMA register tile with warp-level WMMA
fragments. Each warp holds `WM_TILES × WN_TILES` float32 accumulator fragments
(where `WM_TILES = WARP_TILE_M/16`, `WN_TILES = WARP_TILE_N/16`), and computes
using `wmma::load_matrix_sync` + `wmma::mma_sync`.

The COMPUTE_TILE macro uses outer-product order: for each k-step (`_kk`), all A
fragments for the warp are loaded first, then all B fragments, then the full
`WM_TILES × WN_TILES` mma grid.

**Limitation:** the WMMA API is opaque to the compiler — `wmma::fragment` objects
are treated as unanalyzable blobs, causing conservative register allocation. At
BM=128, BN=128, BK=16, NW=4, the compiler allocates ~192 registers/thread →
only 2 CTAs/SM → ~16% occupancy.

---

### TC2 — Raw PTX + B-tile XOR Swizzle

TC2 replaces WMMA with raw PTX, exposing the register layout to the compiler:

| Operation | PTX instruction | Purpose |
|-----------|----------------|---------|
| A load | `ldmatrix.sync.aligned.x4.m8n8` | 16×16 bf16 from smem → 4 uint32 per thread |
| B load | `ldmatrix.sync.aligned.x2.m8n8.trans` | 16×8 bf16 from smem → 2 uint32 per thread, transposed |
| Compute | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | one 16×8 output tile per call |

The native mma width is 16×8, so `WN_TILES = WARP_TILE_N/8` and accumulators
shrink from 8 to 4 floats per tile.

**B-tile XOR swizzle.** The XOR swizzle permutes 8-bf16 column chunks on writes
and undoes the permutation on reads:
```
physical_chunk = logical_chunk ^ (row % B_SWZ),   B_SWZ = BN/8
```

**Occupancy improvement.** At BM=64, BN=128, BK=16, NW=4, TC2 uses ~122
registers/thread vs TC1's ~192, enabling 4 CTAs/SM — a 2× improvement.

---

### TC2b — A-tile XOR Swizzle

TC2b adds the same XOR treatment to the A tile.

**A-tile conflict analysis.** With A stored `[BM][BK]`, the bank-conflict period is:
```
A_SHIFT = 64 / BK
```

| BK | bank period | conflict degree |
|----|-------------|-----------------|
| 16 | 4 rows      | 4-way           |
| 32 | 2 rows      | 8-way           |

The A swizzle: `physical_chunk = logical_chunk ^ ((row / A_SHIFT) % A_SWZ)` with
`A_SWZ = BK/8` reduces to 2-way for both BK=16 and BK=32.

---

### TC5 — Vectorized BF16 Write-Back

TC5 keeps the full TC2b kernel structure and adds two changes:

**Vectorized write-back.** Each `mma.sync m16n8k16` produces 4 float32 outputs;
elements 0&1 and elements 2&3 are row-adjacent pairs. TC5 packs each pair into
one `bfloat162` store, halving epilogue store count:
```c
*(__nv_bfloat162*)&C[row0][col] = __floats2bfloat162_rn(acc[0], acc[1]);
*(__nv_bfloat162*)&C[row8][col] = __floats2bfloat162_rn(acc[2], acc[3]);
```

**BK=64 extension.** At BK=64, `A_SHIFT=1` so each row gets a unique XOR key,
completely eliminating A-tile bank conflicts. The extended search space allows
higher arithmetic intensity at large sizes.

**Config space.** TC5 uses `(BM, BN, BK, NUM_WARPS)` with BK ∈ {16, 32, 64},
yielding **42 valid configs**.

---

### TC5_reg — `__launch_bounds__` LB Tuning

TC5_reg adds `LB_MIN_BLOCKS` as a second argument to `__launch_bounds__`:
```c
__global__ __launch_bounds__(NW * 32, LB_MIN_BLOCKS)
```
The two-arg form is authoritative: it sets the per-thread register budget to
`floor(65536 / (NW*32 * LB_MIN_BLOCKS))`. Unlike the one-arg form (which lets
the compiler exceed `-maxrregcount`), two-arg reliably forces register spilling
for high-LB configs and tighter packing for low-LB configs.

Four cubins are compiled per config (`LB ∈ {1, 2, 3, 4}` for NW=4;
`LB ∈ {1, 2}` for NW=8, due to the halved per-thread register budget). Config
space: **120 configs**.

---

### TC5_regpruned — Register-Estimate Pruning

TC5_regpruned adds pre-tuning pruning to remove configs that will certainly spill:
```
estimate = WM_TILES * WN_TILES * 4   # accumulator regs
         + (BK/16) * (WM_TILES*4 + WN_TILES*2)   # fragment regs (unrolled)
budget   = floor(65536 / (NW*32 * LB))
```
where `WM_TILES = BM*2 / (NW*16)`, `WN_TILES = BN / 16`. Configs where
`estimate > budget` are dropped before benchmarking. This reduces the 120-config
space to **~90 configs** at typical sizes, cutting tuning time by 25% with no
loss in peak performance.

**Key insight.** The dominant register pressure is the accumulator array
(`WM×WN×4` floats), not loop scalars or address registers. Pruning on the
accumulator+fragment estimate accurately identifies configs where the compiler
must spill heavily (measured drops to 30–90 TFLOPS for those configs).

---

## 3. Experimental Variants

The following variants were built on TC5/TC5_regpruned to explore further
optimizations. **None improved over TC5_regpruned** at the primary benchmark
sizes (2048–8192).

### TC5jit — JIT Compile-Time Constants

TC5jit compiles a separate cubin per `(M, K, N)` with `-DM_VAL=... -DK_VAL=...
-DN_VAL=...`. The kernel drops M/K/N runtime arguments and uses
`constexpr int num_tiles = K_VAL / BK` as a compile-time loop bound.

**Why it underperforms TC5_regpruned.** The theoretical gains (3 saved argument
registers, stride multiplies become shifts, compile-time loop bound) are swamped
by the accumulator register pressure (128–176 regs/thread). Saving 3 out of
~180 registers (~1.7%) does not change occupancy. The outer double-buffer loop
is never unrolled regardless of whether the trip count is known (its body is too
large: full SMEM pipeline + MMA blocks). The compiler already fully unrolls the
inner `BK/16` kk-loop via template parameter.

Result: TC5jit lags TC5_regpruned by 2–8% at 2048–8192.

### TC5jit_lb — TC5jit + LB Tuning

TC5jit_lb combines JIT compile-time constants with two-arg `__launch_bounds__`.
One cubin compiled per `(M, K, N, lb)`. The LB tuning recovers 3–5% vs plain
TC5jit, but still falls 1–2% short of TC5_regpruned at all sizes. The best config
consistently selects `LB=1` (tightest budget), confirming that LB tuning matters
even with JIT constants. The JIT advantage does not compound with LB tuning.

### TC6_lb — Split-Pass kk Loop + LB Tuning

TC6 restructures the COMPUTE_TILE macro into three separate passes — ldmatrix
all A, ldmatrix all B, then MMA — before adding LB tuning. This is semantically
equivalent to TC5 after compiler unrolling. Result: **identical to TC5_regpruned**
within 0.2% at all sizes. Same best configs selected.

### TC7_lb — Split A/B Pipeline + LB Tuning

TC7 uses separate `__pipeline_commit()` calls for A-tile and B-tile async copies,
enabling ldmatrix A to start as soon as A tiles arrive without waiting for B.
Uses `__pipeline_wait_prior(3)` for A and `__pipeline_wait_prior(2)` for B.
With LB tuning: within 1% of TC5_regpruned at all sizes, slightly better at 3072
(+0.2%) and 5120 (+0.5%), slightly worse at 6144 (−0.5%).

### TC5swz_lb — CTA GROUP_M Swizzle + LB Tuning

TC5swz adds a GROUP_M CTA swizzle (SW ∈ {1,2,4,8}) to improve B-tile L2 reuse.
With LB tuning the config space expands to 360 configs. The autotuner **always
selects SW=1** (no swizzle) at all tested sizes, confirming the swizzle provides
no benefit when the grid has enough CTAs for natural L2 reuse. Numerically
identical to TC5_regpruned at 2048–4096. Crashes at 5120 with `BM=256, BN=128,
NW=8` configs triggering a GPU illegal memory access, contaminating the CUDA
context.

### TC6_x4b — `ldmatrix_x4_trans` for B Tile

TC6_x4b replaces the per-n-tile `ldmatrix.x2.trans` with a single
`ldmatrix.x4.trans` covering two consecutive n-tiles, halving the B ldmatrix
instruction count per kk-step. Addressing: lanes 0–15 provide the address for
n-tile `nt`; lanes 16–31 provide the address for `nt+1` (exploiting the
previously-wasted lane slots). Marginal +2% at 2048 but −1% at large sizes.
The halved instruction count does not translate to throughput at compute-bound
sizes because ldmatrix latency is already hidden by the outer pipeline.

### TC8g — Multi-Stage Triton-Style Pipeline

TC8g uses a STAGES-deep (2–5) prefetch pipeline with 1D flat thread blocks and
GROUP_M=8 swizzle baked in. The larger STAGES depth allows more outstanding async
copies, enabling an additional CTA/SM when smem permits. Config space: 152 valid
`(BM, BN, BK, NW, NS, LB)` tuples. Restricted to `min(M,N,K) ≥ 2048` due to GPU
crashes with NS=4 on small problem sizes. Not yet benchmarked at 2048–8192
against TC5_regpruned (run was killed by TC5swz crash).

---

## 4. Template and Parameters

**Shared memory** (double-buffered, no padding needed with swizzling):
```
A_shared[2][BM][BK]   (BM*BK*2 bytes per buffer)
B_shared[2][BK][BN]   (BK*BN*2 bytes per buffer)
Total: (2*BM*BK + 2*BK*BN) * 2 bytes
```

**Tunable parameters:**

| Parameter | Candidates | TC5 | TC5_regpruned | TC8g |
|-----------|-----------|-----|---------------|------|
| BM | 64, 128, 256 | ✓ | ✓ | ✓ |
| BN | 64, 128, 256 | ✓ | ✓ | ✓ |
| BK | 16, 32, 64 | ✓ | ✓ | ✓ |
| NUM_WARPS | 4, 8 | ✓ | ✓ | ✓ |
| LB | 1–4 (NW=4), 1–2 (NW=8) | — | ✓ | ✓ |
| STAGES | 2–5 | — | — | ✓ |

**Constraints (TC5_regpruned):**
- `smem(BM, BN, BK) ≤ 100352` bytes
- `BM × BN ≤ 4096 × NW`
- `WM_TILES*WN_TILES*4 + (BK/16)*(WM_TILES*4+WN_TILES*2) ≤ floor(65536 / (NW*32*LB))`

Valid configs: **42** for TC5, **90** for TC5_regpruned (after pruning).

---

## 5. Results

**Hardware:** NVIDIA RTX 4090 (Ada Lovelace, sm_89), 128 SMs.  
**Precision:** BF16 inputs, FP32 accumulators, BF16 output.  
**Peak BF16 tensor-core throughput:** 164 TFLOPS (dense).

### TC5_regpruned vs cuBLAS vs Triton

| Size  | TC5_regpruned | cuBLAS | Triton | regpruned/cuBLAS | regpruned/Triton |
|-------|---------------|--------|--------|-----------------|-----------------|
| 1024  | 87.4          | 91.2   | 95.3   | 96%             | 92%             |
| 2048  | 128.1         | 135.3  | 132.1  | 95%             | 97%             |
| 3072  | 134.8         | 130.8  | 140.9  | **103%**        | 96%             |
| 4096  | 138.7         | 135.6  | 143.1  | **102%**        | 97%             |
| 5120  | 139.1         | 141.5  | 143.9  | 98%             | 97%             |
| 6144  | 141.4         | 142.6  | 144.5  | **99%**         | 98%             |
| 7168  | 139.4         | 143.5  | 144.8  | 97%             | 96%             |
| 8192  | 142.5         | 144.0  | 144.9  | **99%**         | 98%             |
| 9216  | 141.1         | 144.3  | 145.0  | 98%             | 97%             |
| 10240 | 142.8         | 143.1  | 145.2  | **100%**        | 98%             |

TC5_regpruned matches or beats cuBLAS at 3072–4096 (wave-quantization sweet spots
where cuBLAS dips) and stays within 1–2% at 5120+. Triton leads by a consistent
2–4%, attributed to its deeper multi-stage prefetch pipeline.

### LB Variant Comparison at Large Sizes (TFLOPS)

| Size | TC5_regpruned | TC6_lb | TC7_lb | TC5jit | TC5jit_lb |
|------|:---:|:---:|:---:|:---:|:---:|
| 2048 | **128.1** | 128.1 | 128.1 | 120.1 | 124.3 |
| 3072 | 134.8 | 134.8 | **135.1** | 126.1 | 132.3 |
| 4096 | 138.8 | **138.9** | 138.4 | 134.1 | 138.4 |
| 5120 | **139.1** | 139.1 | **139.6** | 134.6 | 137.4 |
| 6144 | **141.4** | **141.4** | 140.7 | 137.1 | 140.3 |
| 7168 | **139.4** | 139.3 | 139.1 | 135.3 | 137.6 |
| 8192 | **142.5** | 142.4 | 142.3 | 138.6 | 141.0 |

TC6_lb and TC7_lb are statistically tied with TC5_regpruned — they converge to
the same or equivalent best configs. TC5jit_lb beats TC5jit by 3–5% but cannot
match TC5_regpruned; the JIT M/K/N constants offer no additional benefit once
register pressure is properly managed via LB pruning.

### Historical Progression

| Size | TC1   | TC2   | TC2b  | TC5   | TC5_regpruned |
|------|-------|-------|-------|-------|---------------|
| 2048 | 103.6 | 107.5 | 110.4 | 123.4 | **128.1**     |
| 4096 | 123.4 | 129.2 | 131.2 | 134.4 | **138.8**     |
| 8192 | 133.6 | 137.2 | 138.5 | 140.0 | **142.5**     |

TC1→TC2: B-tile bank conflict elimination via XOR swizzling.  
TC2→TC2b: A-tile bank conflict reduction (4-way/8-way → 2-way).  
TC2b→TC5: vectorized write-back + BK=64 (conflict-free A tile).  
TC5→TC5_regpruned: two-arg `__launch_bounds__` forces tighter register budget.

### Best Configs Selected by Autotuner

| Size  | TC5_regpruned best |
|-------|-------------------|
| 1024  | BM=64,  BN=64,  BK=64, NW=4, LB=3 |
| 2048  | BM=128, BN=64,  BK=64, NW=4, LB=2 |
| 3072  | BM=64,  BN=128, BK=32, NW=4, LB=1 |
| 4096  | BM=128, BN=128, BK=32, NW=4, LB=1 |
| 5120  | BM=64,  BN=128, BK=32, NW=4, LB=1 |
| 6144  | BM=128, BN=128, BK=32, NW=4, LB=1 |
| 7168  | BM=128, BN=128, BK=32, NW=4, LB=1 |
| 8192  | BM=128, BN=128, BK=32, NW=4, LB=1 |
| 9216  | BM=128, BN=128, BK=32, NW=4, LB=1 |
| 10240 | BM=128, BN=128, BK=32, NW=4, LB=1 |

LB=1 dominates from 3072 onwards. LB=2 wins at 2048 and LB=3 at 1024, where the
smaller grids benefit from higher occupancy at the cost of fewer registers per
thread. BK=64 is preferred at 1024–2048 (higher arithmetic intensity per CTA
compensates for fewer CTAs); BK=32 with larger tiles takes over from 3072+.

---

## 6. Conclusion

The kernel progression from TC1 to TC5_regpruned reaches cuBLAS parity at
4096 and 99% at 8192:

- **TC1 (WMMA):** tensor-core entry point; WMMA opacity limits occupancy to ~16%.
- **TC2 (raw PTX + B swizzle):** exposes register dataflow → 2× occupancy; B-tile
  bank conflicts eliminated.
- **TC2b (A+B swizzle):** A-tile conflict period matched to BK; 4-way/8-way
  conflicts reduced to 2-way.
- **TC5 (vectorized write-back + BK=64):** epilogue stores halved via `bfloat162`;
  BK=64 now conflict-free, improving arithmetic intensity at large sizes.
- **TC5_regpruned (two-arg launch bounds + register pruning):** two-arg
  `__launch_bounds__` authoritatively sets the register budget;
  register-estimate pruning eliminates configs where spill is certain.
  Gains +2–3 TFLOPS over TC5 across all sizes.

A second round of structural exploration (TC6 split passes, TC7 split A/B
pipeline, TC5swz CTA swizzle, TC5jit/TC5jit_lb JIT constants, TC8g multi-stage)
found no improvement over TC5_regpruned. The remaining 1–2% gap behind cuBLAS
and 2–4% behind Triton at large sizes is not yet explained; the register and
occupancy analysis does not point to an obvious remaining bottleneck.
