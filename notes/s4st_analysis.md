# Stage 4 Strided (s4st): Analysis and NCU Profiling

## What s4st Is

s4st is a variant of the double-buffered cp.async kernel (s4) that changes the
thread-to-output mapping from "row-major block" to a strided layout.

**s4 (standard)**: thread `(lty, ltx)` computes rows `lty, lty+1, ..., lty+(TM-1)` — consecutive.

**s4st (strided)**: thread `(lty, ltx)` computes rows `lty, lty+LCOLS, lty+2*LCOLS, ...` — strided
by the number of logical thread columns.

The motivation is to eliminate shared memory bank conflicts in B tile reads.

---

## Why s4st Eliminates Bank Conflicts

### B tile: always zero conflicts

With the strided layout, thread `ltx` owns output columns `ltx, ltx+LCOLS, ltx+2*LCOLS, ...`.
Per `kk`, it reads `B_shared[kk][ltx], B_shared[kk][ltx+LCOLS], ...` — all on the same row.
A warp's 32 threads read 32 *consecutive* columns of a single row → always coalesced, zero conflicts.

### A tile: depends on tile shape

For **bm128_bn128** (LCOLS=16, 8 warps, 16×16 logical thread grid):
- Each warp contains threads with only **2 distinct lty values**
- The 2 rows accessed per kk fall into banks `0..15` and `16..31` respectively → zero conflict

For **bm64_bn64** (LCOLS=8, 4 warps, 8×8 logical thread grid):
- Each warp has 4 distinct lty values → rows 0 & 2 alias → 2-way A conflict

**bm128_bn128 is the sweet spot**: both A and B have zero bank conflicts.

---

## Benchmark Results @ 4096³ (RTX 4090, fp32)

| Kernel | GFLOPS |
|---|---|
| cuBLAS (no TF32) | ~55,742 |
| s4st bm128_bn128 u16 | ~46,084 |
| s4st bm128_bn64 u16 | ~41,000 |
| s4st bm64_bn64 u16 | ~34,000 |
| s4 bm128_bn128 u16 | ~40,000 |
| s4sw bm128_bn128 u16 | ~40,000 |

s4st bm128_bn128 u16 is our best result: **~83% of cuBLAS**.

---

## NCU Profiling: s4st vs cuBLAS @ 4096³

### Speed-of-Light & Occupancy

| Metric | s4st bm128_bn128 u16 | cuBLAS (no TF32) |
|---|---|---|
| SM% | 76.6 | 80.5 |
| **L1/TEX%** | **76.6** | **42.7** |
| L2% | 29.7 | 30.4 |
| DRAM% | 18.9 | 17.6 |
| Occupancy% | 16.7 | 16.7 |
| Smem LD conflicts | 0 | 0 |
| Smem ST conflicts | 0 | 0 |

### Resource Usage

| | s4st | cuBLAS |
|---|---|---|
| Threads/block | 256 | 256 |
| Registers/thread | 242 | 202 |
| Smem/block | 32 KB | 48 KB |
| Occupancy limiter | regs | regs |

---

## Bottleneck Analysis

Both kernels are at identical 16.7% occupancy (1 block/SM), both register-limited.
With only 2 warps per scheduler, there is minimal latency hiding from warp switching.

The critical difference is **L1/TEX throughput**: 76.6% vs 42.7%.

This means:
- Our kernel saturates L1/TEX at the same SM throughput as cuBLAS
- cuBLAS does **more FMAs per shared memory read** — it achieves higher arithmetic intensity
  from shared memory by reusing data already in registers (larger effective tile per smem access)
- We issue 256 smem reads × 16 kk iterations per tile per warp; cuBLAS issues far fewer per FMA

### What warp-level shuffle tiling showed

We explored reducing smem reads via `__shfl_sync` (load once per warp, broadcast to 32 threads).
Theoretically ~5× fewer smem ops. In practice, ~3× *slower* than s4st because:
- Shuffle latency (~20 cycles) adds overhead per kk iteration
- Reduced smem reads didn't compensate for the shuffle synchronization cost
- With only 2 warps/scheduler, no other warp hides the shuffle stalls

### Path to closing the gap

To approach cuBLAS performance from the cuda-core side:
1. **Larger BK** (e.g. BK=32): halves the number of `__syncthreads()` and tile loops,
   amortizes pipeline overhead — at the cost of a 2-way A bank conflict (BK=32 with BM=128,
   stride 32 → rows 0 & 32 alias)
2. **More registers/reuse per smem load**: load A/B once and accumulate across multiple kk
   without re-reading smem — requires unrolling the BK loop and keeping fragments in registers
3. **Higher occupancy**: currently bottlenecked by 242 regs/thread; register tiling (smaller
   TM×TN with more blocks) trades per-block arithmetic intensity for more warps/SM

cuBLAS likely uses a combination of these plus architecture-specific optimizations
(e.g. LDMATRIX, HMMA, asynchronous copy pipelines with larger staging buffers).

---

## BK=32 Experiment: The "Optimize One Thing, Break Another" Pattern

### Analysis before benchmarking

Increasing BK from 16 to 32 has two competing effects:

**Benefit**: Half as many tile iterations, so half the `__syncthreads()` calls and pipeline
overhead. For K=4096 with BM=128: 32 tiles instead of 64. Each tile has 2 syncs, so 64
fewer barriers per block.

**Cost**: A bank conflict is introduced.

With BK=32, `A_shared` is `[BM][32]`, so each row has exactly 32 floats — one bank per
column (`bank = col % 32 = col`). When two threads in a warp read column `kk`:
- With bm128_bn128: warp has 2 lty values (2w, 2w+1). Both read `A_shared[row][kk]`.
  `row*32 + kk` maps both to bank `kk`. **2-way A bank conflict** (vs 0 for BK=16).

With BK=16, `A_shared` is `[BM][16]`, so `bank = (row*16 + kk) % 32`. The two lty rows
differ by 1, so their bank values differ by 16 — no conflict.

**Smem budget hard wall**: For bm128_bn128:
- `A_shared[2][128][32]` = 32768 bytes
- `B_shared[2][32][128]` = 32768 bytes
- Total = 65536 bytes = 64 KB, exceeding the 48 KB per-block default limit.
- bm128_bn128_bk32 cannot be instantiated without dynamic smem + `cudaFuncSetAttribute`.

So our main BK=32 candidate fails: the config that had zero bank conflicts (bm128_bn128)
cannot use BK=32 due to smem budget. The configs that can (bm128_bn64, bm64_bn64) already
have more limited tile coverage.

### Benchmark results @ 4096^3 (RTX 4090)

| Kernel | GFLOPS | vs BK=16 best |
|---|---|---|
| s4st bm128_bn64 bk16 u16  | 44,165 | baseline |
| s4st bm128_bn64 bk32 u8   | 42,407 | -4% |
| s4st bm128_bn64 bk32 u16  | 41,020 | -7% |
| s4st bm128_bn64 bk32 u32  | **45,298** | **+2.5%** |
| s4st bm64_bn64  bk16 u16  | 42,337 | baseline |
| s4st bm64_bn64  bk32 u16  | 39,406 | -7% |
| s4st bm64_bn64  bk32 u32  | 43,352 | +2.4% |

Full unroll (u32) with BK=32 wins slightly: the syncthreads reduction outweighs the A bank
conflicts only when the loop is fully unrolled. At lower unroll factors the conflicts
dominate.

**Takeaway**: +2.5% for a non-trivial change, and only available for configs that can't
achieve zero bank conflicts anyway (bm128_bn128 is smem-limited at BK=32).

---

## Vectorized B Loads: Why Triton Can, We Can't

### What Triton does

The best Triton config (BM=128, BN=128, BK=32, 8 warps) emits `ld.shared.v4.b32`
for B tile reads — loading 4 consecutive floats (128 bits) per instruction, giving ~4x
instruction-level throughput for B.

### Why our strided layout blocks this

In s4st, thread `ltx` reads B at positions:
```
B_shared[kk][ltx + 0*LCOLS],  B_shared[kk][ltx + 1*LCOLS], ...
```
These addresses have stride LCOLS=16 floats = 64 bytes between elements. `float4` loads
require 16-byte-aligned **consecutive** addresses. Our stride is 64 bytes — cannot use
vector loads.

### Why Triton's layout enables it

Triton assigns threads so that each thread owns consecutive output columns within its
block. At B-read time, threads in a warp collectively cover consecutive B columns, making
`ld.shared.v4.b32` natural.

### The coupling problem

This is a direct illustration of how optimizations interact:

| Optimization | What it enables | What it breaks |
|---|---|---|
| Strided output (s4st) | Zero B bank conflicts | Non-consecutive B reads, no float4 |
| Contiguous output (s4) | Consecutive B reads, float4 possible | 2-way B bank conflicts at BN=64 |
| BK=16 bm128_bn128 | Zero A and B conflicts | Smem too small for BK=32 |
| BK=32 bm128_bn64 | Fewer barriers | 2-way A conflicts |

There is no layout that simultaneously achieves zero bank conflicts, vector B loads, and
large BK without a fundamentally different access pattern — e.g., swizzle mapping or
LDMATRIX instructions (which operate on a different memory model designed for tensor cores).

### Why Triton avoids bank conflicts despite vector loads

Triton applies a swizzle (XOR pattern) on shared memory layout automatically during code
generation. Internally, `tl.dot` maps to `mma` tensor core instructions that use LDMATRIX,
which is designed to work with the swizzle pattern. This is architecture-specific
optimization that bypasses the manual bank-conflict reasoning we do for scalar loads.

---

## TODO: Triton Best Config NCU Bank Conflicts

To verify that Triton's swizzle achieves zero bank conflicts, measure with NCU.
Since `cuda_kernel_name()` returns None for Triton, the existing `profile_ncu_sol.py`
skips it. A custom NCU invocation without `--kernel-name` (using SM% heuristic) would
be needed to capture the Triton matmul kernel's conflict counters.
