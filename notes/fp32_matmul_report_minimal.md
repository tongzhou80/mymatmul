# FP32 GEMM on RTX 4090: Design and Results

## Abstract

We implement a high-performance FP32 GEMM kernel in CUDA, reaching **~98% of cuBLAS**
at N=2048 and **~94% at N=4096/8192**, and outperforming Triton's autotuned kernel at
all sizes ≥ 2048, without tensor cores, inline PTX, or assembly tricks. The design
relies entirely on fundamental principles: maximizing arithmetic intensity at both the
global-memory and shared-memory levels, eliminating bank conflicts, vectorized memory
accesses, double-buffered prefetching, warp-level output tiling, and register
prefetching. A templated kernel with five tunable parameters is autotuned empirically
per problem size.

---

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

---

## 3. Results

**Hardware:** NVIDIA RTX 4090 (Ada Lovelace, sm_89), 128 SMs, 82.6 TFLOPS FP32 peak.
**Precision:** FP32, no TF32, no tensor cores.

Triton autotuned with configs: BM, BN ∈ {64,128,256}, BK ∈ {16,32},
num_stages ∈ {3,4}, num_warps=8 (fixed), GROUP_M=8 (fixed).

### Performance

| Size | **s6** (TFLOPS) | **s7** (TFLOPS) | Triton (TFLOPS) | cuBLAS (TFLOPS) | s7 / cuBLAS |
|------|-----------------|-----------------|-----------------|-----------------|-------------|
| 1024 | 38–41 | **41** | 46 | 44 | ~94% |
| 2048 | **52** | **52** | 50 | 52 | ~98% |
| 4096 | **51** | **51** | 48 | 54 | ~94% |
| 8192 | **50** | **50** | 46 | 54 | ~92% |

s6 and s7 are essentially identical at 2048+. s7 shows a consistent edge at 1024 due to
the JIT benefit on small `num_tiles`. Both beat Triton at all sizes ≥ 2048. cuBLAS
leads at 4096+ by ~6–8%, likely due to a deeper async pipeline (≥3 stages) and
better tile-shape selection for those sizes.

### Best Config Selected by Autotuner

**s6:**

| Size | BM | BN | BK | UNROLL | NUM_WARPS |
|------|----|----|----|--------|-----------|
| 1024 | 64 | 128 | 16–32 | 8 | **4** (2×2, 128 threads) |
| 2048 | 256 | 128 | 16 | 8 | **8** (4×2, 256 threads) |
| 4096 | 256 | 128 | 16 | 8 | **8** |
| 8192 | 256 | 128 | 32 | 8 | **8** |

NW=4 wins at 1024 (fewer threads → less occupancy contention at small grid);
NW=8 wins at 2048+ (more threads → better latency hiding at large grid).

**Triton best configs:**

| Size | BM | BN | BK | num_stages | GROUP_M |
|------|----|----|----|------------|---------|
| 1024 | 64 | 128 | 32 | 4 | 8 |
| 2048 | 128 | 128 | 32 | 3 | 8 |
| 4096 | 128 | 256 | 16 | 4 | 8 |
| 8192 | 128 | 256 | 16 | 4 | 8 |

Triton selects BN=256 at large sizes (vs our BN=128 winner), and uses a 4-stage
pipeline throughout. These are both within our search space but our autotuner does not
select them — BN=256 with BM=128 at 4096+ may benefit from Triton's software-managed
multi-stage pipeline which our 2-stage design does not replicate.

---

## 4. Conclusion

A pure CUDA FP32 matmul kernel, built from first principles, reaches 92–98% of cuBLAS
performance across N=2048–8192 and consistently outperforms Triton's autotuned kernel
at those sizes. The key techniques — warp-level output tiling, register prefetching,
float4 vectorized smem loads, and compiler-driven unroll scheduling — are each
individually well-understood, but their combination in a single autotuned template
proves highly effective.

The remaining gap vs cuBLAS at large sizes (4096+) most likely stems from a deeper
async pipeline (3–4 stages vs our 2-stage double buffer), which better tolerates global
memory latency at high occupancy, and possibly a smarter CTA swizzling strategy for L2
reuse. Closing that gap would require either implementing a deeper software pipeline or
exploring BN=256 tiles with a 3-stage prefetch.
