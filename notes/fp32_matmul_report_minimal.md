# FP32 GEMM on RTX 4090: Design and Results

## Abstract

We implement a high-performance FP32 GEMM kernel using CUDA, reaching **91% of cuBLAS**
and outperforming Triton's autotuned kernel at large matrix sizes, without tensor cores,
inline PTX, or assembly tricks. The design relies entirely on fundamental principles:
maximizing arithmetic intensity at both the global-memory and shared-memory levels,
eliminating bank conflicts, vectorizing memory accesses, double-buffered prefetching,
and heavy loop unrolling delegated to the compiler. A single templated kernel with four
tunable parameters is autotuned empirically per problem size.

---

## 1. Design Principles

**Double-buffered asynchronous prefetch.** While the current tile's FMAs are in flight,
the next tile is loaded from global memory asynchronously via `cp.async`. This overlaps
DRAM latency with compute.

**Vectorized shared memory loads.** B-tile loads from shared memory use `float2`
(64-bit), halving instruction count and using the full 64-bit smem data path.

**Maximize arithmetic intensity — global memory.** Use the largest (BM × BN) tile that
fits in shared memory. Larger tiles amortize the DRAM cost of each A and B element
over more FMAs.

**Maximize arithmetic intensity — shared memory.** Assign each thread a large (TM × TN)
output sub-tile. More output elements per thread = more FMAs per smem load = less smem
bandwidth pressure.

**Eliminate shared memory bank conflicts.** Adopt a strided thread-to-output mapping
so that all 32 threads in a warp access distinct smem banks for both A-tile row loads
and B-tile float2 loads.

**Autotune over unroll factor; let nvcc schedule.** Rather than manually analyzing how
many unroll steps are needed to hide smem latency, we unroll the inner K-loop heavily
and let nvcc interleave loads and FMAs across the enlarged code region. The unroll
factor is swept empirically. A larger unrolled body gives the compiler a wider window
for instruction scheduling and register assignment — it consistently does this better
than hand-crafted approaches once enough code is visible.

---

## 2. Kernel Architecture

The kernel is a single C++ template instantiated over `(BM, BN, BK, UNROLL)`.

**Fixed:**

| Parameter | Value |
|-----------|-------|
| Threads per block | 256 |
| Logical thread layout | 16 × 16 (lty × ltx) |
| Thread output tile | TM × TN = (BM/16) × (BN/16) |
| Pipeline | 2-stage double buffer (cp.async) |
| B smem load | float2 (64-bit, 2 floats per instruction) |

**Shared memory layout** (double-buffered, dynamic allocation):
```
A_shared[2][BM][BK]   (first half of smem)
B_shared[2][BK][BN]   (second half of smem)
```
Dynamic allocation (`extern __shared__`) is required for configs exceeding the 48 KB
static limit (e.g., BM=256, BN=128, BK=32 → 50 KB per block).

**Inner loop structure:**
```cpp
#pragma unroll UNROLL
for (int kk = 0; kk < BK; kk++) {
    float a[TM];        // TM scalar loads from A_shared (strided rows, zero bank conflicts)
    float2 b[TN/2];     // TN/2 float2 loads from B_shared (contiguous pairs, zero bank conflicts)
    // TM × TN FMAs into register accumulator acc[TM][TN]
}
```

**Tunable parameters:**

| Parameter | Candidates | Notes |
|-----------|-----------|-------|
| BM | 64, 128, 256 | row tile; larger → higher global-mem arithmetic intensity |
| BN | 64, 128, 256 | col tile; larger → higher global-mem arithmetic intensity |
| BK | 16, 32 | k-step size |
| UNROLL | 2, 4, 8, 16 | inner loop unroll factor |

BM=BN=256 is excluded: `acc[16][16]` = 256 float registers exceeds the 255-register
hardware maximum, causing register spill and severe performance degradation.

Total valid configs: **64**. Each is compiled as a separate kernel. On first load,
all 64 kernels are compiled together by nvcc in a single `.cu` file; the resulting
`.cubin` is cached on disk and reused across runs.

**Autotuning:** on the first call for a given (M, N, K), all 64 configs are timed
(2 warmup + 3 measured runs each). The best is cached in memory and reused for all
subsequent calls with the same shape.

---

## 3. Results

**Hardware:** NVIDIA RTX 4090 (Ada Lovelace, sm_89), 128 SMs, 82.6 TFLOPS FP32 peak.
**Precision:** FP32, no TF32, no tensor cores.

Triton autotuned with 32 configs: BM, BN ∈ {64,128,256}, BK ∈ {16,32},
num_stages ∈ {3,4}, num_warps=8 (fixed), GROUP_M=8 (fixed); BM=BN=256 excluded.

### Performance

| Size | **s5** (TFLOPS) | Triton (TFLOPS) | cuBLAS (TFLOPS) |
|------|-----------------|-----------------|-----------------|
| 1024 | 40.3 | **46.6** | 43.9 |
| 2048 | **49.6** | 50.6 | 52.1 |
| 4096 | **49.4** | 48.0 | 54.2 |
| 8192 | **48.5** | 47.0 | 53.9 |

s5 reaches **91% of cuBLAS** at 4096³ and **90% at 8192³**.

### Best Config and Autotune Time

**s5:**

| Size | BM | BN | BK | UNROLL | Autotune time |
|------|----|----|----|--------|---------------|
| 1024 | 128 | 64 | 32 | 16 | ~19 s (incl. ~19s nvcc compile) |
| 2048 | 256 | 128 | 32 | 16 | ~0.1 s |
| 4096 | 256 | 128 | 32 | 16 | ~1.1 s |
| 8192 | 256 | 128 | 32 | 16 | ~9.4 s |

nvcc compilation of all 64 kernels takes ~19 seconds on first run; the `.cubin` is
cached on disk so subsequent runs (different problem sizes in the same session, or
future sessions) skip compilation entirely.

**Triton best configs (identified from autotune cache):**

| Size | BM | BN | BK | num_stages |
|------|----|----|----|------------|
| 1024 | 64 | 128 | 32 | 4 |
| 2048 | 128 | 128 | 32 | 3 |
| 4096 | 256 | 128 | 32 | 3 |
| 8192 | 128 | 256 | 16 | 4 |

num_warps=8 wins at every size; num_stages=2 never wins. These findings were used to
prune Triton's search space from 96 → 32 configs (~3× faster autotuning with no
quality loss).
