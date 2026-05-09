# High-Performance FP32 Matrix Multiplication on RTX 4090: Design Philosophy and Results

## Abstract

We implement a high-performance FP32 GEMM kernel for NVIDIA RTX 4090 using CUDA,
reaching **49.4 TFLOPS at 4096³** — 91% of cuBLAS (54.2 TFLOPS) and above the
best Triton autotuned result (48.0 TFLOPS) — without tensor cores, inline PTX, or
compiler-specific assembly tricks. The kernel is derived from four fundamental
principles: maximizing arithmetic intensity at both the global-memory and
shared-memory levels, eliminating shared memory bank conflicts, vectorizing memory
instructions, and using double-buffered asynchronous prefetching. A deliberate
choice to abandon manual instruction scheduling in favor of heavy loop unrolling
combined with autotuning allows the compiler (nvcc) to handle all register
allocation and instruction interleaving — a collaboration that outperforms every
hand-crafted scheduling attempt.

---

## 1. Introduction

Writing a fast GEMM kernel is often portrayed as a sequence of increasingly arcane
optimizations: hand-written SASS, custom register file tricks, undocumented hardware
behaviors. This report argues the opposite: **the fundamental optimizations are
sufficient, and the compiler is a better scheduler than the programmer.**

The target is square FP32 GEMM (C = A × B, no TF32, no tensor cores) on RTX 4090
(Ada Lovelace, sm_89). The theoretical SIMT FP32 peak is 82.6 TFLOPS. cuBLAS
achieves 54.2 TFLOPS at 4096³ — likely via tensor-core FP16 accumulation
internally. Our goal is to understand how close pure SIMT FP32 can get and, more
importantly, *why*.

---

## 2. Background: Tiling and the Memory Hierarchy

Standard tiled GEMM maps a (BM × BN) output tile to each thread block. Each block
iterates over the K dimension in steps of BK, loading (BM × BK) from A and
(BK × BN) from B into shared memory, then computing the partial product. Threads
within the block each own a (TM × TN) sub-tile of the output.

The memory hierarchy has three levels relevant to performance:

| Level | Bandwidth (RTX 4090) | Latency |
|-------|---------------------|---------|
| Global (DRAM) | ~1 TB/s | ~400 cycles |
| L2 cache | ~5 TB/s | ~200 cycles |
| Shared memory | ~20 TB/s | ~30 cycles |
| Registers | essentially free | 0 cycles |

Every byte fetched from DRAM must fund as many FLOPs as possible. Every byte
loaded from shared memory must equally fund as many FMAs as possible. These are the
two arithmetic intensity constraints that drive all design decisions.

---

## 3. Design Philosophy

### 3.1 Common Practices Applied

Two techniques are standard in any serious GEMM implementation and are applied here
without modification:

**Double-buffered asynchronous prefetching (cp.async).** While the current
(BM × BK) A-tile and (BK × BN) B-tile are being consumed by the FMA pipeline,
the *next* tile's data is loaded asynchronously from global memory into a second
shared memory buffer. This overlaps DRAM latency with compute and keeps the FMA
units fed. On Ada Lovelace, `cp.async` (`__pipeline_memcpy_async`) allows
in-flight global loads without blocking the warp.

**Vectorized memory instructions.** Wherever the access pattern permits, loads and
stores use 128-bit (float4) or 64-bit (float2) instructions rather than scalar
floats. This reduces instruction count, better utilizes the memory bus width, and
lowers register pressure per element moved. Specifically, B-tile loads from shared
memory use `float2` to load two consecutive elements per instruction.

These two techniques are the foundation. Everything else is an extension.

### 3.2 Maximize Arithmetic Intensity — Global Memory

Arithmetic intensity for global memory is:

```
AI_global = (2 * M * N * K) / bytes_from_DRAM
           ≈ (BM * BN * K/BK * 2 * BK) / ((BM * BK + BK * BN) * 4)
           = 2 * BM * BN / (4 * (BM + BN))
```

This increases monotonically with BM and BN. A thread block with BM=256, BN=128
has roughly twice the arithmetic intensity of BM=64, BN=64, for the same number of
threads. **The primary design choice is therefore to use the largest tiles that fit
in shared memory and registers.**

The shared memory constraint is:

```
smem = 2 × (BM × BK + BK × BN) × 4 bytes   (factor 2 for double buffering)
```

RTX 4090 supports up to 100 KB dynamic shared memory per block (`MAX_DYNAMIC_SHARED_SIZE_BYTES`).
This permits BM=256, BN=128, BK=32 (50 KB) while excluding BM=BN=256 (128 KB > 100 KB).

### 3.3 Maximize Arithmetic Intensity — Shared Memory

Each thread owns a (TM × TN) output sub-tile. For each k-step, it loads TM
elements from A-smem and TN elements from B-smem and performs TM×TN FMAs. The
arithmetic intensity from shared memory is:

```
AI_smem ≈ (TM * TN) / (TM + TN)
```

This increases with tile size. With a fixed 256-thread block and a 16×16 logical
thread layout, TM = BM/16 and TN = BN/16. For BM=256, BN=128: TM=16, TN=8,
giving 128 FMAs per thread per k-step from only 24 smem loads — the highest ratio
achievable without exceeding the register limit.

### 3.4 Bank Conflict Analysis

Shared memory on NVIDIA GPUs is organized into 32 banks (4-byte stride). When
multiple threads in a warp access the same bank, accesses serialize. This is
orthogonal to arithmetic intensity — a high-intensity kernel can still be smem-bound
if its access pattern causes conflicts.

We analyze the thread-to-smem-bank mapping for both the A and B tiles under each
layout choice:

- **A-tile (strided row mapping):** Thread `(lty, ltx)` reads `A_smem[lty + i*LROWS][kk]`.
  For a fixed kk, the 32 threads of a warp span `lty` values spaced by 1 (within a
  half-warp). With LROWS=16, adjacent threads in lty access consecutive rows, which
  maps to different smem banks → zero conflicts.

- **B-tile (float2 load):** Thread `(lty, ltx)` reads
  `B_smem[kk][2*ltx + j*2*LCOLS]` as a float2. The 16 threads in ltx spread across
  32 consecutive floats (one full smem row of BN=128). With float2, each thread owns
  a 2-float (8-byte) aligned slot, keeping all 16 accesses to distinct bank pairs →
  zero conflicts.

Choosing the strided output layout (s4st) over the contiguous layout (s4) was the
key change that eliminated the bank conflict bottleneck and allowed the smem pipe to
clear, making the kernel compute-bound rather than smem-bound.

### 3.5 Abandon Manual Scheduling — Autotune Instead

The remaining degree of freedom is the **unroll factor** of the inner compute loop
(iterating over BK k-steps). This is where our philosophy diverges most sharply from
traditional hand-tuning.

The conventional analysis goes: "the smem load latency is ~30 cycles; each kk
iteration has T_load + T_fma ≈ 30+5 cycles; to hide the load latency, we need at
least N warps or U unroll steps such that..." This analysis is tractable for a
single warp but becomes unreliable as register pressure grows, occupancy drops to
1 block/SM, and ptxas's interference graph produces register counts that differ from
analytical prediction by 15–20%.

**We abandon this analysis entirely.** Instead:

1. Unroll aggressively (U ∈ {2, 4, 8, 16}).
2. Let nvcc see a large straight-line code region and find the best instruction
   interleaving and register assignment on its own.
3. Autotune over (BM, BN, BK, U) to find the best combination empirically.

This works because: with heavy unrolling, ptxas has full visibility into all
in-flight loads and FMAs across multiple k-iterations. It can interleave them
optimally — something impossible at U=1 where each iteration is a separate loop
body. The compiler's register allocator also has freedom to choose assignments that
avoid pipeline stalls, whereas manual PTX forces specific register choices that may
conflict with the allocator's preferences for other instructions.

Empirical confirmation: at U=1, writing the inner loop in inline PTX to force
front-loading (loads first, then FMAs) achieves **81.5% SM utilization**. At U=8
in plain C++, nvcc produces **~65% SM utilization** — lower occupancy — but
achieves **47.4 TFLOPS vs 42.9 TFLOPS** because the deeper intra-warp pipeline
(more instructions in flight per warp) more than compensates for the lower SM%.

**The compiler is a better scheduler than the programmer, once given enough
unrolled code to work with.**

---

## 4. Implementation

### 4.1 Thread Block Layout

| Parameter | Value |
|-----------|-------|
| Threads per block | 256 (fixed) |
| Logical layout | 16 × 16 (lty × ltx) |
| TM = BM / 16 | tunable |
| TN = BN / 16 | tunable |

Each thread accumulates a TM × TN register tile. Float2 B loads means TN must be
even (satisfied for all BN ∈ {64, 128, 256}).

### 4.2 Tunable Parameters (Stage 5)

| Parameter | Values | Notes |
|-----------|--------|-------|
| BM | 64, 128, 256 | row-tile per block |
| BN | 64, 128, 256 | col-tile per block |
| BK | 16, 32 | k-step per tile |
| UNROLL | 2, 4, 8, 16 | inner loop unroll |

BM=BN=256 excluded: acc[16][16] = 256 float registers > 255 hardware max → spill.
Total: **64 valid configurations** per (M, N, K).

### 4.3 Shared Memory Layout

```
smem[0..2*BM*BK-1]         → A_shared[2][BM][BK]   (double buffer for A)
smem[2*BM*BK..end]         → B_shared[2][BK][BN]   (double buffer for B)
```

Declared as `extern __shared__ float smem[]` (dynamic allocation) to support
configurations exceeding the 48 KB static limit.

### 4.4 Pipeline Structure

```
ISSUE_TILE(k=0, buf=0)           // prefetch first tile
for k = 0 .. num_tiles-2:
    ISSUE_TILE(k+1, buf=1-cur)   // prefetch next tile
    __pipeline_wait_prior(1)      // wait for current tile
    __syncthreads()
    COMPUTE_TILE(cur)             // compute from current buffer
    __syncthreads()
__pipeline_wait_prior(0)
COMPUTE_TILE(last)
```

`COMPUTE_TILE` is a macro with `#pragma unroll UNROLL` over the BK inner loop,
giving the compiler the full unrolled body as a straight-line code region.

---

## 5. Evaluation

Hardware: NVIDIA RTX 4090 (Ada Lovelace, sm_89), 128 SMs, 82.6 TFLOPS FP32 peak,
72 MB L2 cache. All results are FP32, no TF32, no tensor cores.

### 5.1 Progressive Optimization Results (4096³)

| Stage | Kernel | TFLOPS | vs cuBLAS |
|-------|--------|--------|-----------|
| Baseline | naive global memory | ~2 | 4% |
| s1 | shared memory tiling | ~10 | 18% |
| s2 | thread tile (TM×TN) | ~20 | 37% |
| s3 | smem double buffer | ~33 | 61% |
| s4 | cp.async prefetch | ~37 | 68% |
| s4st | strided layout (no bank conflicts) | ~44 | 81% |
| s4st2 (TN=16, float2 B) | vectorized smem load | 47.4 | 87% |
| **s5 autotuned** | **template + autotune** | **49.4** | **91%** |
| cuBLAS | reference | 54.2 | 100% |

### 5.2 Three-way Comparison: s5 vs Triton vs cuBLAS

| Size | s5 (TFLOPS) | Triton (TFLOPS) | cuBLAS (TFLOPS) |
|------|-------------|-----------------|-----------------|
| 1024 | 40.3 | **46.6** | 43.9 |
| 2048 | 49.6 | **50.6** | 52.1 |
| 4096 | **49.4** | 48.0 | 54.2 |
| 8192 | **48.5** | 47.0 | 53.9 |

### 5.3 Best Autotuned Configurations

**s5 (our CUDA kernel):**

| Size | BM | BN | BK | UNROLL |
|------|----|----|----|--------|
| 1024 | 128 | 64 | 32 | 16 |
| 2048 | 256 | 128 | 32 | 16 |
| 4096 | 256 | 128 | 32 | 8 |
| 8192 | 256 | 128 | 32 | 16 |

**Triton autotuned (32 configs: num_warps=8 fixed, num_stages ∈ {3,4}):**

| Size | BM | BN | BK | num_stages |
|------|----|----|----|------------|
| 1024 | 64 | 128 | 32 | 4 |
| 2048 | 128 | 128 | 32 | 3 |
| 4096 | 256 | 128 | 32 | 3 |
| 8192 | 128 | 256 | 16 | 4 |

---

## 6. Discussion

### 6.1 Why s5 Trails Triton at Small Sizes

At N=1024, both s5 and Triton land roughly 128 active thread blocks (1 per SM).
The bottleneck shifts from compute to pipeline depth: Triton uses up to 4 software
pipeline stages (overlapping 4 tiles in flight), while s5 has only 2-stage double
buffering. With lower arithmetic intensity at small sizes, hiding DRAM latency
matters more, and Triton's deeper pipeline wins (+15% at 1024).

### 6.2 Why s5 Beats Triton at Large Sizes

At N=4096+, the kernel is fully compute-bound. The number of in-flight tiles matters
less than the FMA density per clock. s5's aggressive unrolling (U=8/16) gives nvcc
a large code region to achieve deep intra-warp instruction-level parallelism. Triton
generates PTX directly and has less freedom to reschedule across stage boundaries.
The result: s5 is +2–3% at 4096 and 8192.

### 6.3 The cuBLAS Gap

cuBLAS is 10% ahead at large sizes regardless of our tile choices. This gap is not
closeable with SIMT FP32 kernels. cuBLAS uses HMMA (tensor core) instructions
internally, even for FP32 inputs, accumulating in FP16 or BF16 with higher precision
fallback. Tensor cores provide 8× the raw FLOP throughput per clock of SIMT float
units on Ada Lovelace — the gap is a hardware path, not an algorithmic one.

### 6.4 The Role of nvcc as Co-author

At U=1, manual PTX scheduling (forcing loads before FMAs) beats the C++ compiler
by ~16% because the compiler cannot see across loop iterations. At U=2+, the
compiler has enough code visibility to discover and apply the same pattern — and
does so better, because it has full knowledge of the register interference graph and
can assign registers to minimize stalls in ways that hand-crafted PTX cannot.

The crossover point is the unroll factor. Below it, the programmer is a better
scheduler. Above it, nvcc is. The practical conclusion: don't stay below the
crossover — unroll aggressively and autotune.

---

## 7. Conclusion

91% of cuBLAS performance for FP32 GEMM on RTX 4090 is achievable with:

1. **Large tiles** (high BM, BN) — maximizes global-memory arithmetic intensity.
2. **Large per-thread output tiles** (high TM, TN) — maximizes shared-memory arithmetic intensity.
3. **Bank-conflict-free layouts** (strided output, float2 B loads) — removes the smem serialization bottleneck.
4. **Double-buffered cp.async** — overlaps DRAM latency with compute.
5. **Vectorized memory instructions** (float2) — standard practice.
6. **Heavy unrolling + autotuning** — replaces manual instruction scheduling with compiler-driven ILP discovery.

No inline PTX, no SASS, no undocumented hardware features. The remaining 9% gap to
cuBLAS is attributable to tensor cores — a hardware capability, not a missed
software optimization.

The broader lesson: GPU kernel optimization is a collaboration between the programmer
and the compiler. The programmer's job is to express the *structure* — tile shape,
memory layout, pipeline pattern, unroll factor. The compiler's job is to fill in the
*schedule*. Trying to do both by hand eventually creates friction without gain.
