# Stage 4 Strided (s4st): Analysis and NCU Profiling

## What s4st Is

s4st is a variant of the double-buffered cp.async kernel (s4) that changes the
thread-to-output mapping from "row-major block" to a strided layout.

**s4 (standard)**: thread `(lty, ltx)` computes rows `lty, lty+1, ..., lty+(TM-1)` — consecutive.

**s4st (strided)**: thread `(lty, ltx)` computes rows `lty, lty+LROWS, lty+2*LROWS, ...` — strided
by the number of logical thread rows.

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

| Kernel | GFLOPS | Notes |
|---|---|---|
| cuBLAS (no TF32) | ~55,742 | reference |
| **s4st2 bm128_bn128 bk16 u16** | **~45,777** | **best custom kernel** |
| s4st bm128_bn128 bk32 u32 | ~45,653 | dynamic smem |
| s4st bm128_bn128 bk16 u16 | ~44,577 | |
| s4st bm128_bn64 bk32 u32 | ~45,298 | |
| s4st bm128_bn64 bk16 u16 | ~44,165 | |
| s4 bm128_bn128 u16 | ~40,000 | |
| s4sw bm128_bn128 u16 | ~40,000 | |

s4st2 bm128_bn128 bk16 u16 is our best result: **~82% of cuBLAS**.

---

## NCU Profiling @ 4096³: Cross-Kernel Comparison

### Speed-of-Light, Occupancy, and Smem Instructions

| Kernel | GFLOPS | SM% | L1/TEX% | Occ% | Regs | smem_ld_inst |
|---|---|---|---|---|---|---|
| s4st bk16 u16 | 44,577 | 76.6 | 76.6 | 16.7 | 242 | 342M |
| s4st bk32 u32 | 45,653 | 77.5 | 77.5 | 16.7 | 248 | 339M |
| s4st2 (old unpack) u16 | 44,311 | 67.9 | 48.7 | 16.7 | 164 | 208M |
| **s4st2 (direct) u16** | **45,777** | **70.5** | **50.7** | **16.7** | **168** | **208M** |
| Triton w8s4 | ~48,000 | 70.2 | 48.1 | 16.7 | 188 | 171M |
| cuBLAS (no TF32) | 55,742 | 80.5 | 42.7 | 16.7 | 202 | — |

### What SM% means

`sm__throughput.avg.pct_of_peak_sustained_elapsed` measures how much of the SM's
instruction-issue capacity is utilized, averaged over the kernel's elapsed time.

Key observation: for s4st bk16/bk32, **SM% equals L1/TEX%**. This means every
instruction-issue cycle is fully occupied processing smem requests — the smem pipe
and the SM are in lockstep. The bottleneck is memory-side.

For s4st2 and Triton, **SM% > L1/TEX%** (70 vs 50). The smem pipe is no longer
the bottleneck; the remaining throughput gap is due to compute-side stalls
(register dependency latency, warp switch overhead).

---

## BK=32 Experiment

### Bank conflict analysis

Increasing BK from 16 to 32 has two competing effects:

**Benefit**: Half as many tile iterations — for K=4096 with BM=128: 32 tiles instead
of 64. Each tile has 2 `__syncthreads()`, so 64 fewer barriers per block.

**Cost**: A bank conflict is introduced.

With BK=32, `A_shared` is `[BM][32]` — exactly one bank per column (`bank = col % 32 = col`).
When two threads in a warp read column `kk` from different lty rows: both map to bank `kk`
→ **2-way A bank conflict** (vs 0 for BK=16).

With BK=16: `bank = (row*16 + kk) % 32`. The two lty rows in a warp differ by 1,
so their bank values differ by 16 — no conflict.

### Smem budget hard wall

For bm128_bn128:
- `A_shared[2][128][32]` = 32768 bytes
- `B_shared[2][32][128]` = 32768 bytes
- Total = 65536 bytes = 64 KB, exceeding the 48 KB per-block default limit.

**Solution**: dynamic shared memory + `cudaFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, 65536)`.
Ada Lovelace (RTX 4090) supports up to ~100 KB per block. This unlocks bm128_bn128_bk32
and its zero-bank-conflict property transfers to BK=32 too (same bank arithmetic).

The dynamic smem kernel is in `cuda_core/_matmul_cuda_ext_s4st_bk32.cu`, wrapped by
`matmul_cuda_s4st_bk32.py` which passes `smem_bytes=65536`.

### Benchmark results @ 4096³

| Kernel | GFLOPS | vs BK=16 best |
|---|---|---|
| s4st bm128_bn128 bk16 u16 | 44,577 | baseline |
| s4st bm128_bn128 bk32 u32 | **45,653** | **+2.4%** |
| s4st bm128_bn64 bk16 u16 | 44,165 | baseline |
| s4st bm128_bn64 bk32 u32 | 45,298 | +2.5% |
| s4st bm64_bn64 bk16 u16 | 42,337 | baseline |
| s4st bm64_bn64 bk32 u32 | 43,352 | +2.4% |

Full unroll (u32) wins: the syncthreads reduction only outweighs the A bank conflicts
when the loop is fully unrolled and the register pressure from a larger unroll is
acceptable.

NCU for bk32 shows SM%=77.5, L1/TEX%=77.5 — still in lockstep, smem is the
bottleneck, same profile as bk16 just with slightly fewer total wavefronts.

---

## Vectorized B Loads: Why Triton Can, We Can't

### What Triton does

The best Triton config (BM=128, BN=128, BK=32, 8 warps, 4 stages) emits
`ld.shared.v4.b32` for B tile reads — loading 4 consecutive floats (128 bits)
per instruction. Its smem_ld_inst is 171M vs our 342M (bk16) — half the instructions.

### Why our strided layout blocks this

In s4st, thread `ltx` reads B at positions:
```
B_shared[kk][ltx + 0*LCOLS],  B_shared[kk][ltx + 1*LCOLS], ...
```
These addresses have stride LCOLS=16 floats = 64 bytes between elements.
Float4 loads require 16-byte-aligned **consecutive** addresses. Our stride is
64 bytes — vector loads are impossible.

### The coupling problem

| Optimization | What it enables | What it breaks |
|---|---|---|
| Strided output (s4st) | Zero B bank conflicts | Non-consecutive B reads, no float4 |
| Contiguous output (s4) | Consecutive B reads, float4 possible | 2-way B bank conflicts at BN=64 |
| BK=16 bm128_bn128 | Zero A and B conflicts | Smem too small for BK=32 (static) |
| BK=32 bm128_bn64 | Fewer barriers | 2-way A conflicts |

There is no layout that simultaneously achieves zero bank conflicts, vector B loads,
and large BK without a fundamentally different access pattern — e.g., swizzle mapping
or LDMATRIX instructions (designed for tensor cores).

### How Triton avoids bank conflicts despite vector loads

Triton applies a swizzle (XOR pattern) on shared memory layout automatically during
code generation. Internally, `tl.dot` maps to `mma` tensor core instructions that
use LDMATRIX, which is designed to work with the swizzle pattern. This is an
architecture-specific optimization that bypasses the manual bank-conflict reasoning
required for scalar loads.

---

## Float2 B Loads Experiment (s4st2)

### Hypothesis

With the s4st strided layout, thread ltx reads B_shared at stride LCOLS=16 — non-consecutive,
blocking float2 loads. Change to a "2-contiguous" layout: each thread ltx owns column pairs
`{2*ltx, 2*ltx+1}`, `{2*ltx+32, 2*ltx+33}`, ... (stride 2*LCOLS=32 between pairs). At step
j (0..3), thread ltx reads `float2` from `B_shared[kk][2*ltx + j*32]`.

**Bank conflict analysis** (BK=16, BN=128, LCOLS=16): bank = c%32. At step j:
- 16 threads load float2 at cols 2*ltx + j*32, 2*ltx + j*32 + 1
- ltx=0 → banks 0,1; ltx=1 → banks 2,3; ...; ltx=15 → banks 30,31
- All 32 banks distinct; lanes with same ltx (across the warp's other half) access
  the same address → broadcast, not a conflict.
- **Zero conflicts**. ✓

This is implemented as `s4st2` (see `cuda_core/_matmul_cuda_ext_s4st2.cu`).

### Attempt 1: unpack into intermediate array (worse)

Initial COMPUTE_TILE unpacked `float2 _bv` into `float _b[TN]`, then ran the TM×TN
FMA loop over the `_b` array:
```c
float _b[TN];
for (int _j = 0; _j < TN/2; _j++) {
    float2 _bv = *reinterpret_cast<const float2*>(...);
    _b[2*_j] = _bv.x;  _b[2*_j+1] = _bv.y;
}
for (int _i = 0; _i < TM; _i++)
    for (int _j = 0; _j < TN; _j++)
        acc[_i][_j] += _a[_i] * _b[_j];
```

Results: 44,311 GFLOPS (-1.3% vs s4st bk16 u16). NCU: SM%=67.9, L1/TEX%=48.7, 164 regs.

**Why it was slower**: The intermediate `_b[TN]` array broke the dependency chain between
load and FMA. The compiler had to hold all 8 loaded values before issuing any FMA,
worsening the schedule. SM% dropped from 76.6 to 67.9 — the SM itself became underutilized.

Also, the core insight about wavefronts applies here:

> The L1/TEX pipeline is measured in **wavefronts** (32-bank cycles), not instructions.
> A float2 load issues 1 instruction but consumes 2 wavefronts.
> 4×float2 = 8 wavefronts = same as 8×scalar.
> Halving instruction count left memory traffic (wavefronts) unchanged.

### Attempt 2: direct float2 FMA — the fix

Instead of unpacking into `_b[]`, use `bv.x` and `bv.y` directly inside the j-loop,
consuming each value immediately in TM FMAs before loading the next:
```c
for (int _j = 0; _j < TN/2; _j++) {
    float2 _bv = *reinterpret_cast<const float2*>(...);
    for (int _i = 0; _i < TM; _i++) {
        acc[_i][2*_j]   += _a[_i] * _bv.x;
        acc[_i][2*_j+1] += _a[_i] * _bv.y;
    }
}
```

Results: **45,777 GFLOPS** (+2.7% vs s4st bk16 u16, new best). NCU: SM%=70.5, L1/TEX%=50.7, 168 regs.

**Why it works**: Eliminating the intermediate array shortens the load→FMA dependency
chain. The compiler can interleave the float2 load with the TM FMAs that follow, keeping
the execution units busy. SM% rises to 70.5 — now matching Triton's 70.2.

The L1/TEX% drops from 76.6 (s4st) to 50.7 (s4st2) because the wavefront count is now
lower — the loads are fewer (4 float2 instructions vs 8 scalar per kk per thread) and
the smem pipe is no longer the hard bottleneck.

### Comparison with Triton

| Kernel | SM% | L1/TEX% | Regs | smem_ld_inst |
|---|---|---|---|---|
| s4st bk16 u16 | 76.6 | 76.6 | 242 | 342M |
| s4st bk32 u32 | 77.5 | 77.5 | 248 | 339M |
| **s4st2 (direct) u16** | **70.5** | **50.7** | **168** | **208M** |
| Triton w8s4 | 70.2 | 48.1 | 188 | 171M |

s4st2 and Triton now have the **same SM%/L1/TEX% profile**: both are out of the
smem-bottleneck regime and into the compute-stall regime. The remaining performance
gap is:
- Triton uses v4 smem loads (`ld.shared.v4.b32`) for smem_ld_inst=171M vs our 208M
- Triton uses 3-stage pipeline (96 KB dynamic smem) vs our 2-stage
- Triton uses 188 regs/thread; we use 168 with fewer accumulators

To fully close the gap from the CUDA-core side would require either v4 B loads
(which requires restructuring to 4-contiguous output assignment — 4×LCOLS stride
between owned columns) or a 3-stage pipeline for more memory latency hiding.

### Key lessons

1. **Instruction count ≠ bandwidth**: reducing smem instructions doesn't reduce L1/TEX
   wavefronts if the vector width is smaller than the bank group size.
2. **Dependency chain length matters**: an intermediate accumulator array between
   a smem load and subsequent FMAs can hurt the compiler's scheduling freedom enough
   to reduce SM utilization by ~12%.
3. **SM% vs L1/TEX% reveals the bottleneck**: when they're equal, smem is the
   rate-limiter; when SM% > L1/TEX%, the smem pipe is cleared and instruction-level
   stalls dominate.

---

## Bottleneck Analysis: s4st vs cuBLAS

Both kernels are at 16.7% occupancy (1 block/SM), both register-limited.

The critical difference is **L1/TEX throughput**: s4st is at 76.6%, cuBLAS at 42.7%.
cuBLAS does more FMAs per shared memory read — higher arithmetic intensity from smem by
reusing data across a larger effective tile. cuBLAS likely uses tensor cores (HMMA) with
LDMATRIX for structured smem reads, achieving 4× the FLOPs per smem access.

Our best custom kernel (s4st2) is now in a different regime — L1/TEX% ~50%, same as Triton.
The remaining ~15% gap to cuBLAS comes from cuBLAS using tensor cores (8× FLOPs per smem
access) rather than SIMT FP32 FMAs.

### What warp-level shuffle tiling showed

Explored reducing smem reads via `__shfl_sync` (load once per warp, broadcast to 32 threads).
~3× slower than s4st because:
- Shuffle latency (~20 cycles) adds overhead per kk iteration
- Reduced smem reads didn't compensate for the shuffle synchronization cost
- With only 2 warps/scheduler, no other warp hides the shuffle stalls

---

## Summary: Optimization History

| Stage | Change | GFLOPS | Change |
|---|---|---|---|
| s4 | double-buffered cp.async | ~40,000 | baseline |
| s4st | strided layout, zero bank conflicts | ~44,577 | +11% |
| s4st bk32 u32 | dynamic smem, larger tiles | ~45,653 | +2.4% |
| s4st2 (direct) | float2 B loads, no temp array | **~45,777** | **+2.7%** |
| Triton w8s4 | v4 loads, 3-stage pipeline | ~48,000 | reference |
| cuBLAS | tensor cores | ~55,742 | reference |
