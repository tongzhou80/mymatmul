# GEMM Optimization on RTX 4090: FP32 CUDA Cores and BF16 Tensor Cores

## Abstract

We implement high-performance GEMM kernels for both FP32 (CUDA cores) and BF16
(tensor cores) on the RTX 4090 (Ada Lovelace, sm_89), sharing a common design
template and reaching competitive performance against cuBLAS and Triton in both
precision regimes.

**FP32 (s6_lb):** reaches **106–107% of cuBLAS** at N=2048–3072 and N=5120, and
**91–98% at N=4096/6144–8192**, consistently outperforming Triton's autotuned FP32
SIMT kernel at all sizes ≥ 2048. No tensor cores, inline PTX, or assembly tricks
are used.

**BF16 (tc5_regpruned):** reaches **102–103% of cuBLAS BF16** at N=3072–4096
(wave-quantization sweet spots) and **99–100% at N=8192–10240**. On small-M skewed
shapes (M=64–128), beats cuBLAS BF16 by **5–9%** and matches Triton within 1%.
Numerically, the FP32 accumulator path makes tc5_regpruned near-lossless vs a full
FP32 reference (mean error 0.002 vs cuBLAS BF16's 0.214 at K=16384).

Both kernels share the same foundation: double-buffered `cp.async` prefetch,
warp-level output tiling, vectorized shared-memory loads, register prefetching, and
two-argument `__launch_bounds__` tuning. The BF16 kernel adds raw PTX tensor-core
instructions, XOR swizzle for bank-conflict elimination, and vectorized BF16
write-back.

---

---

# Part I — FP32 CUDA Core Kernel

## 1. Design Principles

**Double-buffered asynchronous prefetch.** While the current tile's FMAs are in flight,
the next tile is loaded from global memory asynchronously via `cp.async`. This overlaps
DRAM latency with compute.

**Vectorized shared memory loads.** B-tile loads from shared memory use `float4`
(128-bit), reducing instruction count and using the full 128-bit smem data path.

**Maximize arithmetic intensity — global memory.** Use the largest (BM × BN) tile that
fits in shared memory. Larger tiles amortize the DRAM cost of each A and B element
over more FMAs.

**Maximize arithmetic intensity — shared memory.** Assign each thread a large (TM × TN)
output sub-tile. More output elements per thread = more FMAs per smem load = less smem
bandwidth pressure.

**Eliminate shared memory bank conflicts.** Adopt a strided thread-to-output mapping
so that all 32 threads in a warp access distinct smem banks for both A-tile row loads
and B-tile float4 loads.

**Warp-level output tiling.** The thread block's output tile is partitioned across
warps in a 2D inter-warp grid (WARP_M × WARP_N = 2×2 or 4×2). Within each warp,
threads are arranged in a fixed 4×8 intra-warp layout (LWARP_M=4, LWARP_N=8), with
each thread owning a TM×TN sub-tile of the warp tile. Warps operate independently
with no cross-warp smem traffic.

**Register prefetching.** Within the BK inner loop, the next row of A and chunk of B
are pre-loaded into registers while the current round of FMAs executes, hiding smem
read latency behind compute.

**Autotune over unroll factor; let nvcc schedule.** Rather than manually analyzing how
many unroll steps are needed to hide smem latency, we unroll the inner K-loop heavily
and let nvcc interleave loads and FMAs across the enlarged code region. A larger
unrolled body gives the compiler a wider window for instruction scheduling and register
assignment.

---

## 2. Kernel Architecture

The kernel is a single C++ template instantiated over `(BM, BN, BK, UNROLL, NUM_WARPS)`.

**Fixed per instantiation:**

| Parameter | Value |
|-----------|-------|
| Inter-warp layout | WARP_M × WARP_N = (NUM_WARPS/2) × 2 |
| Intra-warp layout | LWARP_M × LWARP_N = 4 × 8 (fixed) |
| Thread output tile | TM × TN = (BM / WARP_M / 4) × (BN / WARP_N / 8) |
| Pipeline | 2-stage double buffer (cp.async) |
| B smem load | float4 (128-bit, 4 floats per instruction) |

**Shared memory layout** (double-buffered, dynamic allocation):
```
A_shared[2][BM][BK]   (first half of smem)
B_shared[2][BK][BN]   (second half of smem)
```
Dynamic allocation (`extern __shared__`) is required for configs exceeding the 48 KB
static limit (e.g., BM=256, BN=128, BK=32 → 50 KB per block).

**Algorithm overview:**

```
Each thread block computes C[block_row : block_row+BM, block_col : block_col+BN].
Each warp owns a contiguous WARP_TILE_M×WARP_TILE_N sub-tile of that output.
Each thread owns acc[TM][TN] in registers (the thread's portion of the warp tile).

issue async load: A_shared[0] ← A[block_row:, 0:BK]      (cp.async)
                  B_shared[0] ← B[0:BK, block_col:]      (cp.async)

for k_tile = 0 .. K/BK - 1:                               ← outer K loop
    issue async load: A_shared[next] ← A[block_row:, (k_tile+1)*BK : ...]
                      B_shared[next] ← B[(k_tile+1)*BK : ..., block_col:]
    wait for A_shared[cur], B_shared[cur]        (pipeline_wait_prior)
    __syncthreads()                              ← all threads see the loaded tile

    pre-load a_reg[0][0..TM]   from A_shared[cur][warp_row_base, kk=0]
    pre-load b_reg[0][0..TN/4] from B_shared[cur][kk=0, warp_col_base]  (float4)

    #pragma unroll UNROLL
    for kk = 0 .. BK-1:                                   ← inner BK loop
        pre-load a_reg[next] ← A_shared[cur][warp_row_base, kk+1]
        pre-load b_reg[next] ← B_shared[cur][kk+1, warp_col_base]  (float4)
        acc[0..TM][0..TN] += outer_product(a_reg[cur], b_reg[cur]) ← TM×TN FMAs

    __syncthreads()                              ← done reading smem[cur]; safe to overwrite

write acc[TM][TN] → C  (float4 vectorized stores)
```

The outer loop uses a 2-stage `cp.async` double buffer to overlap DRAM loads with
compute. The inner loop over the BK columns of each smem tile uses a register
ping-pong buffer (pre-loading the next A row and B float4 chunk while the current
round of FMAs executes) to hide smem read latency. The `UNROLL` factor controls how
many inner iterations are unrolled into a single code block, widening the compiler's
scheduling window for interleaving loads and FMAs.

**Tunable parameters:**

| Parameter | Candidates | Notes |
|-----------|-----------|-------|
| BM | 64, 128, 256 | row tile; larger → higher global-mem arithmetic intensity |
| BN | 64, 128, 256 | col tile; larger → higher global-mem arithmetic intensity |
| BK | 16, 32 | k-step size |
| UNROLL | 2, 4, 8, 16 | inner loop unroll factor |
| NUM_WARPS | 4, 8 | 4 → 2×2 inter-warp (128 threads); 8 → 4×2 inter-warp (256 threads) |

BM=BN=256 is excluded: TM×TN = 256×256/(NUM_WARPS×32) > 128 registers, causing spill.
BM×BN ≤ 4096×NUM_WARPS is enforced to keep TM×TN ≤ 128.

Total valid configs: **112**. All kernels are compiled together by nvcc into a single
`.cubin`, cached on disk and reused across runs.

**Autotuning:** on the first call for a given (M, N, K), all valid configs are timed
(2 warmup + 3 measured runs each). The best is cached in memory for subsequent calls.
The unroll factor order is [16, 8, 4, 2] so high-unroll configs are measured first
(before GPU cache state is perturbed), avoiding a systematic bias toward low unroll.

**Stage 7 — shape-specific JIT compilation.** A variant compiles the kernel template
per (M, N, K) via `nvcc -DM_VAL=M -DN_VAL=N -DK_VAL=K`, baking the three dimensions
as `constexpr`. This allows the compiler to treat `num_tiles = K/BK` as a known loop
count for better scheduling, and statically eliminates the bounds-check branch in the
store epilog. The compiled `.cubin` is cached on disk by (M, N, K). This yields
meaningful gains at small sizes (≥+8% at N=1024 where `num_tiles` is small) and
marginal gains at large sizes.

**Stage 8 — `__launch_bounds__` LB tuning (s6_lb).** The two-argument form
`__launch_bounds__(NW*32, LB_MIN_BLOCKS)` instructs the compiler to guarantee at least
`LB_MIN_BLOCKS` concurrent blocks per SM, which sets an authoritative register budget
of `floor(65536 / (NW*32 * LB))` registers per thread. The one-argument form used in
s6 defaults to `LB=2` on modern GPUs, unnecessarily capping registers for NW=8 kernels
to 128. Adding `LB ∈ {1,2,3,4}` (for NW=4) and `LB ∈ {1,2}` (for NW=8) as a sixth
tunable axis lets the autotuner trade occupancy for ILP as appropriate per size.
Register-estimate pruning discards configs where the accumulator + prefetch-buffer
footprint exceeds the LB budget before benchmarking. All compiled cubins are cached on
disk by LB value.

---

## 3. Results

**Hardware:** NVIDIA RTX 4090 (Ada Lovelace, sm_89), 128 SMs, 82.6 TFLOPS FP32 peak.
**Precision:** FP32, no TF32, no tensor cores.

**Measurement methodology.** Each kernel is timed with `triton.testing.do_bench`
(100 ms warmup budget, 500 ms timed budget). `do_bench` flushes the GPU L2 cache
between iterations using a built-in cache-flushing buffer. GFLOPS is reported from
the best (minimum) kernel time; FLOP count = 2·M·N·K.

Triton FP32 SIMT autotuned with configs: BM, BN ∈ {64,128,256}, BK ∈ {16,32},
num_stages ∈ {3,4}, num_warps=8 (fixed), GROUP_M=8 (fixed).

### Performance

| Size | **s6** (TFLOPS) | **s6_lb** (TFLOPS) | Triton (TFLOPS) | cuBLAS (TFLOPS) | s6_lb / cuBLAS |
|------|-----------------|---------------------|-----------------|-----------------|----------------|
| 1024 | 33.8 | **37.4** | 38.1 | 38.9 | 96% |
| 2048 | 49.1 | **51.6** | 46.2 | 48.5 | **107%** |
| 3072 | 49.3 | **50.6** | 46.1 | 47.9 | **106%** |
| 4096 | 50.4 | **51.3** | 47.4 | 52.2 | 98% |
| 5120 | 43.5 | **49.6** | 45.9 | 46.7 | **106%** |
| 6144 | 49.7 | **49.8** | 46.3 | 52.3 | 95% |
| 7168 | 48.7 | 48.4 | 46.0 | 50.8 | 95% |
| 8192 | 48.9 | **49.1** | 46.4 | 53.9 | 91% |

s6_lb beats s6 at all sizes, with the largest gains at 1024 (+11%), 5120 (+14%), and
2048 (+5%). Both beat Triton FP32 SIMT at all sizes ≥ 2048. s6_lb exceeds cuBLAS at
2048, 3072, and 5120 — wave-quantization sweet spots where cuBLAS's tile selection
underperforms. cuBLAS leads at 4096+ by 2–9%, likely due to a deeper async pipeline
(≥3 stages) or CTA swizzling for L2 reuse.

The 5120 dip in s6 (43.5 vs ~49–50 TFLOPS for neighbors) illustrates the tile-size
interaction with register pressure: at 5120, tiles that normally deliver high
arithmetic intensity (e.g. BM=128, BN=128) require more registers per thread than the
implicit LB=2 budget allows, causing the s6 autotuner to fall back to BM=64, BN=128
which is register-safe but arithmetically lighter. s6_lb with LB=1 unlocks the full
register file for BM=128, BN=128 and recovers the performance.

### Best Configs Selected by Autotuner

**s6:**

| Size | BM | BN | BK | UNROLL | NW |
|------|----|----|----|--------|----|
| 1024 | 64 | 128 | 16 | 8 | 4 |
| 2048 | 128 | 128 | 16 | 2 | 4 |
| 3072 | 64 | 128 | 16 | 16 | 4 |
| 4096 | 256 | 128 | 16 | 8 | 8 |
| 5120 | 64 | 128 | 16 | 16 | 4 |
| 6144 | 256 | 128 | 16 | 8 | 8 |
| 7168 | 128 | 128 | 16 | 8 | 4 |
| 8192 | 256 | 128 | 32 | 4 | 8 |

**s6_lb:**

| Size | BM | BN | BK | UNROLL | NW | LB |
|------|----|----|----|--------|----|----|
| 1024 | 64 | 128 | 32 | 8 | 4 | 1 |
| 2048 | 128 | 128 | 16 | 2 | 4 | 1 |
| 3072 | 64 | 128 | 16 | 16 | 4 | 1 |
| 4096 | 256 | 128 | 16 | 8 | 8 | 1 |
| 5120 | 128 | 128 | 16 | 8 | 4 | 1 |
| 6144 | 256 | 128 | 32 | 8 | 8 | 1 |
| 7168 | 256 | 128 | 16 | 8 | 8 | 1 |
| 8192 | 256 | 128 | 32 | 4 | 8 | 1 |

All sizes select LB=1 — at every scale the grid is large enough that maximizing
registers for ILP wins over occupancy-based latency hiding.

**Triton best configs:**

| Size | BM | BN | BK | num_stages |
|------|----|----|----|------------|
| 1024 | 64 | 128 | 32 | 4 |
| 2048 | 128 | 128 | 32 | 3 |
| 4096 | 128 | 256 | 16 | 4 |
| 8192 | 128 | 256 | 16 | 4 |

Triton selects BN=256 at large sizes and a 4-stage pipeline, neither of which our
2-stage design replicates.

---

## 4. Conclusion

A pure CUDA FP32 matmul kernel, built from first principles, reaches 91–107% of cuBLAS
performance across N=1024–8192 and consistently outperforms Triton's autotuned FP32
SIMT kernel at all sizes ≥ 2048. The key techniques — warp-level output tiling,
register prefetching, float4 vectorized smem loads, and compiler-driven unroll
scheduling — are each individually well-understood, but their combination in a single
autotuned template proves highly effective.

Adding two-argument `__launch_bounds__` as a tunable axis (s6_lb) yields consistent
gains by removing the implicit per-block occupancy floor that otherwise caps the
register file. All sizes favor LB=1 (maximum registers), confirming that for matrix
sizes where the grid fully saturates the GPU, ILP from a larger register file
outweighs the latency-hiding benefit of higher occupancy.

The remaining gap vs cuBLAS at large sizes (6144+, ~5–9%) most likely stems from a
deeper async pipeline (3–4 stages vs our 2-stage double buffer) and possibly CTA
swizzling for L2 reuse. Closing that gap would require implementing a deeper software
pipeline or exploring BN=256 tiles with a 3-stage prefetch.

---

---

# Part II — BF16 Tensor Core Kernel

## 1. Inherited Template

The s6 kernel (Part I) established the following structure, which all tensor-core
variants carry forward unchanged:

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

**Measurement methodology.** Each kernel is timed with `triton.testing.do_bench`
(100 ms warmup budget, 500 ms timed budget). `do_bench` flushes the GPU L2 cache
between iterations using a built-in cache-flushing buffer. GFLOPS is reported from
the best (minimum) kernel time; FLOP count = 2·M·N·K.

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

## 6. Skewed (Non-Square) Shapes

All square-size results above use `M = N = K`. This section evaluates small-M shapes
representative of batch matrix-multiply workloads (e.g. token-length × model-dim ×
vocab in LLM inference).

### Performance vs cuBLAS and Triton

| Shape (M×K×N) | TC5_regpruned | cuBLAS BF16 | Triton BF16 | tc5/cuBLAS | tc5/Triton |
|---------------|:-------------:|:-----------:|:-----------:|:----------:|:----------:|
| 64×16384×65536 | 60.9 | 58.0 | 60.9 | **105%** | 100% |
| 128×16384×65536 | 119.7 | 109.7 | 119.9 | **109%** | 100% |
| 256×16384×65536 | 139.9 | 142.7 | 141.9 | 98% | 99% |
| 64×8192×65536 | 59.4 | 54.4 | 59.2 | **109%** | 100% |
| 128×8192×65536 | 114.5 | 106.4 | 116.3 | **108%** | 98% |

TC5_regpruned beats cuBLAS by 5–9% at M=64 and M=128, and is statistically tied
with Triton across all small-M shapes. The one exception is M=256, where both fall
~1–2% short: with BM=256 the occupancy constraint `BM×BN ≤ 4096×NW` forces BN=64
(narrow-N tile), reducing arithmetic intensity per CTA while cuBLAS can likely
select a wider asymmetric tile.

### LB Behavior at Small M

| Shape | TC5_regpruned best config |
|-------|--------------------------|
| 64×16384×65536 | BM=64 BN=128 BK=32 NW=4 **LB=4** |
| 64×8192×65536 | BM=64 BN=128 BK=32 NW=4 **LB=4** |
| 128×16384×65536 | BM=128 BN=256 BK=32 NW=8 **LB=1** |
| 128×8192×65536 | BM=128 BN=128 BK=16 NW=4 **LB=1** |
| 256×16384×65536 | BM=256 BN=64 BK=32 NW=4 **LB=1** |

At M=64 the grid has only one row of M-tiles, so the total block count is
`N/BN × 1`. With so few blocks per SM, occupancy matters more than ILP: LB=4 (4
blocks/SM enforced, tighter register budget) wins over LB=1. This is the opposite
of the large-square-matrix regime. At M=128+ the grid is large enough to saturate
the GPU, and LB=1 resumes dominance.

### Numerical Accuracy

With M=64, K=16384, N=65536 and standard-normal BF16 inputs (output std ≈ 128):

| Comparison | Max abs err | Mean abs err |
|------------|------------|-------------|
| cuBLAS BF16 vs FP32 ref | 4.0 | 0.214 |
| TC5_regpruned vs FP32 ref | 4.0 | **0.002** |
| TC5_regpruned vs cuBLAS BF16 | 4.0 | 0.214 |

Our kernel uses WMMA `float` accumulators throughout — the BF16 inputs are only
loaded into fragment registers; all intermediate accumulation happens in FP32. The
result is bit-identical to a full FP32 matmul at the accumulator level, with error
only from the final BF16 output cast (max ~4.0 at magnitude ~128, i.e. one BF16
ULP). cuBLAS BF16 introduces ~100× more mean error — likely using a different
internal accumulation path.

The max abs err of 4.0 in all cases is expected BF16 quantization of the output:
at values ~128, the BF16 step size is ~1.0, so rounding error up to ~4.0 is normal.

---

## 7. Conclusion

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

On small-M skewed shapes (M=64–128, K/N ∼ 8192–65536), TC5_regpruned **beats
cuBLAS BF16 by 5–9%** and matches Triton within 1%. The autotuner shifts to
higher LB (LB=4 at M=64) to favor occupancy over ILP when the grid has only one
M-tile row. Numerically, the FP32 accumulator path means TC5_regpruned is
effectively lossless vs a full FP32 reference (mean error 0.002 vs cuBLAS BF16's
0.214 at K=16384).
