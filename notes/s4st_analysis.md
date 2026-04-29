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

---

## Float4 B Loads Experiment (s4st4 / s4st4_xor)

### Motivation

s4st2 achieves smem_ld_inst=208M vs Triton's 171M. The gap is 37M instructions — all from
B-side smem loads (Triton uses v4 = float4 loads). Hypothesis: switch from float2 to float4 B
loads to close this instruction gap.

### Why float4 requires a different warp layout

With the existing 16×2 warp layout (LCOLS=16 distinct ltx per warp):
- float2 B reads: 16 threads × 2 floats = 32 floats → exactly 32 banks → zero conflicts ✓
- float4 B reads: 16 threads × 4 floats = 64 floats → wraps 32 banks twice → 2-way B conflict ✗

Fix: **8×4 warp layout** (LCOLS_W=8 distinct ltx per warp):
- float4 B reads: 8 threads × 4 floats = 32 floats → exactly 32 banks → zero B conflicts ✓
- Same THREADS=256, TM=8, TN=8, BM=128, BN=128 — only the warp-internal grouping changes.
- B read stride: `4*ltx + j*4*LCOLS` (j=0,1) — correctly covers all 128 BN columns without overlap.

Thread mapping:
```c
warp_id = tid / 32;  lane = tid % 32
ltx = (warp_id % 2) * 8 + lane % 8   // 0..15
lty = (warp_id / 2) * 4 + lane / 8   // 0..15
```

### A bank conflict with 8×4 layout

Warp 0 has lty ∈ {0,1,2,3} (consecutive rows). With BK=16:
- bank(A_shared[row][kk]) = (row * 16 + kk) % 32
- row=0: bank=kk;  row=1: bank=(kk+16)%32;  row=2: bank=kk ← conflict;  row=3: bank=(kk+16)%32 ← conflict
- → 2-way A bank conflict (rows {0,2} alias, {1,3} alias).

This is the same 2-way A conflict that s4st bk32 has and that s4st2 bk16 avoids.

### XOR swizzle attempt (s4st4_xor)

Apply `physical_col = kk XOR ((row & 2) * 4)` at both store and load time. This rotates
alternate row-pairs by 8 columns, giving banks {kk, (kk+16)%32, kk^8, (kk^8+16)%32} for
rows 0,1,2,3 — all distinct, zero A conflicts.

### Benchmark results @ 4096³ (3-run average)

| Kernel | GFLOPS | Notes |
|---|---|---|
| s4st2 bk16 u16 | ~45,763 | reference |
| s4st4 (no XOR) u16 | ~44,600 | −2.5% vs s4st2 |
| s4st4_xor u16 | ~40,700 | −11% vs s4st2 |

### NCU profile @ 4096³

| Kernel | SM% | L1/TEX% | smem_ld_wf | smem_ld_inst | LD-conf |
|---|---|---|---|---|---|
| s4st2 bk16 u16 | 70.4 | 50.7 | 402M | 207M | 0 |
| s4st4 (no XOR) u16 | 71.4 | 49.8 | 402M | 140M | 0 |
| s4st4_xor u16 | 72.5 | 55.6 | 469M | 241M | 0 |

### What the data shows

**s4st4 (no XOR)**: Float4 B loads reduce smem_ld_inst by 32% (207M→140M) but wavefronts
are unchanged (402M). Why? Float2 generates 1 wf per step (16-unique-ltx × 2 = 32 floats =
32 banks). Float4 also generates 1 wf per step (8-unique-ltx × 4 = 32 floats = 32 banks).
With half as many steps, total B wf halves — but the 2-way A bank conflict adds wf that
exactly compensates. Net: same smem bandwidth, fewer instructions. Despite fewer instructions,
GFLOPS is −2.5%, because we're in compute-stall regime (SM% > L1/TEX%), not
instruction-issue-bound.

**s4st4_xor**: The XOR swizzle was supposed to eliminate A conflicts, but it causes MORE
instructions (241M, up from 140M) and MORE wavefronts (469M, up from 402M) than s4st4.
The root cause: `_kk ^ ((_row & 2) * 4)` involves a runtime-valued lty, so the compiler
cannot vectorize A reads into float4. It falls back to scalar or float2 loads, multiplying
A-side instructions and erasing all gains.

Crucially, the conflict counter (LD-conf) reads 0 for s4st4 despite the theoretical 2-way A
conflict. The hardware likely handles the two broadcast-groups-to-same-bank as sequential
wavefronts without incrementing the conflict register. This is consistent with smem_ld_wf
being the same as s4st2 (the extra wavefronts appear in the base wf count, not the conflict
delta).

### Conclusion

Float4 B loads are not viable from the CUDA C side:
- Without swizzle: instruction reduction doesn't translate to throughput gain (same wf, A conflict latency).
- With XOR swizzle: compiler can't vectorize A reads, making things worse.

Triton achieves float4 smem reads via LDMATRIX / mma instructions that are architecture-aware
and swizzle-compatible. That path requires tensor core instructions (WMMA/PTX mma), not SIMT FP32.

---

## s4st2 bk32 Re-evaluation (3-run average @ 4096³)

| Kernel | Avg GFLOPS |
|---|---|
| s4st  bk16 u16 | 44,751 |
| s4st  bk32 u32 | 45,559 |
| s4st2 bk16 u16 | 45,763 |
| s4st2 bk32 u16 | 44,384 |
| s4st2 bk32 u32 | 45,740 |

**s4st2 bk16 u16 ≈ s4st2 bk32 u32** (within 25 GFLOPS = noise). BK=32 does not compound
with float2 because s4st2 already left the smem-bottlenecked regime. The simpler s4st2 bk16
(static smem, no cudaFuncSetAttribute) is the preferred best kernel.

---

## Summary: Optimization History (up to s4st2)

| Stage | Change | GFLOPS | Change |
|---|---|---|---|
| s4 | double-buffered cp.async | ~40,000 | baseline |
| s4st | strided layout, zero bank conflicts | ~44,751 | +12% |
| s4st bk32 u32 | dynamic smem, larger tiles | ~45,559 | +1.8% |
| s4st2 bk16 u16 | float2 B loads, no temp array | **~45,763** | **+2.3%** |
| s4st2 bk32 u32 | float2 + larger tile | ~45,740 | ≈ s4st2 bk16 |
| s4st4 (float4) | 8×4 warp layout, float4 B | ~44,600 | −2.5% vs s4st2 |
| Triton w8s4 | v4 loads, 3-stage pipeline | ~48,000 | reference |
| cuBLAS | tensor cores | ~55,742 | reference |

---

---

## TN=16 Kernel Family (s4st_tn16)

### Motivation: Higher Arithmetic Intensity from Smem

Increasing TN from 8 to 16 (and BN from 128 to 256) changes the per-kk smem load count:

| Config | FMAs/kk | A loads/kk | B loads/kk | FMA/load ratio |
|---|---|---|---|---|
| TN=8, BN=128 (s4st2) | 64 | 8 | 8 (float2) | 4.0 |
| TN=16, BN=256 (s4st_tn16) | 128 | 8 | 16 (scalar) | **5.33** |

More FMAs per smem load means the FMA pipe can stay busier relative to the smem pipe.
The cost: BN=256 halves the number of blocks for small matrices (bad for N<2048), and
the 16 scalar B loads per kk add register pressure.

### Inline PTX Implementation

`_matmul_cuda_ext_s4st_tn16.cu` uses a `LD_S(reg, base, byte_off)` macro:
```c
asm("ld.shared.f32 %0, [%1+" #byte_off "];" : "=f"(reg) : "r"(base));
```
All per-element offsets are compile-time constants; only `A_addr` and `B_addr` base
pointers increment per-kk. This means ptxas never needs extra registers for offset arithmetic.

### Profiling Results @ 4096³

| Kernel | Regs | smem_ld_wf | SM% | TFLOPS |
|---|---|---|---|---|
| tn16_u1 | 194 | 402M | 81.5 | 42.9 |
| tn16_u2 | 255 | 335M | ~64 | 41.9 |
| tn16_u4 | 255 | 335M | ~64 | ~42.5 |
| s4st2_u16 (prev best) | 168 | 208M inst | 70.5 | 45.5 |

tn16_u1 achieves 81.5% SM — the highest of any variant — because 194 registers stay
below the hardware register-pressure cliff. But u2+ all jump to 255 registers and drop
to ~64% SM, losing the advantage.

---

## The 255-Register Cliff

Every kernel that reaches 255 registers (the sm_89 maximum) converges to ~64% SM,
regardless of instruction ordering (u2, u4, m2 hand-crafted interleaving, all identical
within 1%). The drop is ~17 percentage points from the ~81% achievable at 194 registers.

Key evidence that this is hardware-level, not scheduling quality:
- All 255-reg kernels land at 63–65% SM regardless of load/FMA interleaving style
- Occupancy is the same (1 block/SM → 8 warps/SM) for both 194-reg and 255-reg kernels
- Register file utilization: 255 regs × 256 threads = 65,280 entries out of 65,536 (99.6%)

Most plausible cause: the physical register file has banked read ports. At near-full
utilization, multiple concurrent FMA operand reads conflict on the same bank, serializing
what should be parallel register reads. This is consistent with the consistent floor value
and independence from instruction scheduling.

### Alternative Pipeline Variants

Two variants were designed to escape the 255-reg cliff while improving latency hiding:

**m2 (hand-crafted 2-way interleaving)**: `_matmul_cuda_ext_s4st_tn16_m2.cu`
Unrolls 2 kk iterations in one loop body: all 48 loads (kk+0 and kk+1), then 256 FMAs.
Result: 255 regs, 64.5% SM, 41.4 TFLOPS. Same cliff as compiler unroll.

**p1 (register-prefetch pipeline)**: `_matmul_cuda_ext_s4st_tn16_p1.cu`
Prefetches kk=0 before the loop; each loop body issues next-kk loads then FMAs on
current kk. Register count: 193, just below the cliff.
Result: 193 regs, 79% SM, 41.1 TFLOPS. Below the cliff but underperforms u1 (81.5%)
due to prologue/epilogue overhead and the rename copies adding instruction overhead.

Neither variant beats tn16_u1. The conclusion: u1 with 194 registers is naturally in
the sweet spot — just below the cliff, with the loop back-edge incidentally preventing
ptxas A-load vectorization (which would force 255 regs).

---

## s4st_tn16_f2: Float2 B Loads with TN=16

### Design

Replace the 16 scalar B loads per kk with 8 float2 loads using 2-contiguous output
assignment (same as s4st2): thread ltx owns column pairs `{2*ltx, 2*ltx+1}`,
`{2*ltx+32, 2*ltx+33}`, ... Bank conflict analysis is identical to s4st2: zero conflicts.

Per-kk summary:
- A: 8 scalar loads (unchanged)
- B: 8 float2 loads (vs 16 scalar)
- FMAs: 128
- FMA/load instruction ratio: 128/16 = **8.0** (vs 5.33 for scalar tn16)

### Register Count Surprise

| Kernel | Regs |
|---|---|
| tn16_u1 (scalar, inline PTX) | 194 |
| tn16_f2_u1 (float2 B, C++ indexing) | **168** |
| tn16_f2_u2 | 255 |
| tn16_f2_u4 | 255 |
| tn16_f2_u8 | 255 |
| tn16_f2_u16 | 255 |

f2_u1 uses only 168 registers. With the `#pragma unroll`-ed j-loop (8 float2 steps),
the compiler allocates `_bv` as a 2-register slot reused per j-step, whereas the scalar
version may keep multiple `b_j` values live simultaneously to hide load latency.

Despite 168 registers and fewer instructions (271M vs 405M), f2_u1 performs *worse*
(39.5 vs 42.9 TFLOPS): the float2 loop structure interleaves each load with its
immediate FMAs (`load bv; FMA×16; load bv; FMA×16; ...`), creating tight load-use
dependency on every step — no latency hiding.

### Wavefronts vs Instructions

| Kernel | smem_ld_wf | smem_ld_inst |
|---|---|---|
| tn16_u1 (scalar) | 402M | 405M |
| tn16_f2_u1 (float2 B) | 402M | 271M |
| tn16_f2_u2 | 335M | 204M |
| tn16_f2_u4/u8 | 335M | 170M |

**Float2 B reduces instructions but NOT wavefronts.** Each `ld.shared.v2.f32` with
16 unique addresses (all 32 banks used by 16 threads × 2 elements) requires 2 memory
cycles — one for .x, one for .y. Thus 8 float2 instructions = 16 wavefronts = same
as 16 scalar instructions.

By contrast, ptxas-vectorized A loads (enabled at u2+ where two consecutive kk A values
are visible in the same code region) DO reduce wavefronts: A_shared[row][kk] and
A_shared[row][kk+1] are adjacent 4-byte elements; with heavy broadcast (16 threads
→ 2 unique addresses), the v2 load needs only 1 wavefront. 8 scalar A loads → 4 float2
A loads = 4 wavefronts (down from 8). This explains the 402M→335M drop at u2+.

### Benchmark Results @ 4096³

| Kernel | Regs | SM% | L1% | TFLOPS |
|---|---|---|---|---|
| tn16_u1 (baseline) | 194 | 81.5 | 81.5 | 42.9 |
| tn16_f2_u1 | 168 | 59.4 | 50.1 | 39.5 |
| tn16_f2_u2 | 255 | 61.4 | 41.6 | 42.5 |
| tn16_f2_u4 | 255 | 67.4 | 39.8 | 46.0 |
| **tn16_f2_u8** | **255** | **~69** | **~39** | **47.4** |
| tn16_f2_u16 | 255 | — | — | 44.9 |

f2_u8 is the new best CUDA kernel: **47.4 TFLOPS at 4096³ (+11% over s4st_tn16_u1)**.

### Why f2_u8 Wins Despite the Register Cliff

f2_u8 has 255 regs and only 67–69% SM — worse than tn16_u1's 81.5%. Yet it's faster.
The key: L1/TEX% drops from 81.5% (tn16_u1) to ~39% (f2_u8). The smem pipe is no
longer the bottleneck. With far fewer load instructions (170M vs 405M), each active
cycle has more FMAs relative to loads, so the FMA pipeline runs at higher effective
density even though peak SM% is lower.

The optimal unroll is u8 = BK/2. At u16 (full unroll, BK/16 = 1 outer loop iteration),
performance drops, likely because the instruction scheduler has trouble finding good
orderings across the enormous inlined body and/or I-cache pressure increases.

### Why Higher Unroll Always Helps (in this regime)

The standard latency-hiding argument ("128 FMAs >> 20-cycle smem latency, u1 is enough")
fails here because it assumes other warps hide the latency. With 4 warps per SM (low
occupancy) and high register pressure, warp-level latency hiding is nearly disabled.
The only mechanism is intra-warp ILP via unrolling: with u2+, ptxas can schedule kk+1
loads while executing kk FMAs (since they're inlined into the same code region and are
independent). With u1, each kk iteration starts with a short stall (~4 cycles) while
the last few B loads return.

---

## Final Benchmark Comparison @ 4096³

| Kernel | TFLOPS | Notes |
|---|---|---|
| cuBLAS (no TF32) | 55.7 | tensor cores |
| Triton autotuned (BN=256 added) | **47.8** | BM=128,BN=128,BK=32,w8,s3 wins |
| **s4st_tn16_f2_u8** | **47.4** | new best CUDA SIMT |
| s4st2_u16 | 45.5 | prev best CUDA SIMT |
| s4st_tn16_u1 | 42.9 | |

s4st_tn16_f2_u8 closes the gap to Triton to within 1%. The remaining gap is Triton's
larger BK=32 (more arithmetic intensity per tile) and 3-stage pipeline.

s4st2 remains the better kernel for small matrices (BN=128 → more blocks → better SM
utilization at N≤2048). The two kernels are complementary by matrix size.
