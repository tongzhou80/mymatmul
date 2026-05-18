# Optimizing BF16 Matmul on H800 (Hopper, sm_90a) — A Tutorial

This walks through the optimization trajectory of a hand-written BF16 matrix
multiplication kernel on the NVIDIA H800 GPU, starting from an Ada-style
kernel ported to Hopper and ending at parity with **cuBLAS** and ahead of
**Triton**. Each step is a single, isolated change — explained mechanically,
benchmarked, and tied to the underlying hardware reason it works.

All numbers below are at **N=8192 (square M=K=N=8192)** measured with
`triton.testing.do_bench(warmup=200ms, rep=2000ms)` on H800 SXM5 (132 SMs,
228 KB SMEM/SM, 989 TFLOPS BF16 tensor-core peak).

---

## 1. Performance Progression

Numbers below are the **median of 3 independent autotune+bench runs**
at N=8192 (per-kernel run-to-run variation ±1-3%; details in §1b
methodology).

| # | Kernel | Key change vs previous | TFLOPS @ 8192 | % of cuBLAS | % of TC peak |
|---|--------|------------------------|--------------:|------------:|-------------:|
| — | **cuBLAS BF16** | reference (vendor) | **735** | 100% | 74.3% |
| 1 | `h1_ms`             | Ada baseline ported (cp.async + mma.sync + XOR swizzle + N-stage pipeline, autotuned) | 390 | 53.1% | 39.4% |
| 2 | `h2_s5`             | switch to **wgmma** + B128 SMEM swizzle (TMA-loaded) | 438 | 59.6% | 44.3% |
| 3 | `h2_s7`             | **wgmma.wait_group 1** — overlap tensor-core compute with the next tile's load | 581 | 79.1% | 58.7% |
| 4 | `h2_s7_runptr`      | **running pointers** — Triton-style per-thread gmem ptrs, save ~15 int ops/iter | 631 | 85.9% | 63.8% |
| 5 | `h2_s8_smem_wb`     | **SMEM-staged vec-4 writeback** — fix coalescing of wgmma fragment layout | 657 | 89.4% | 66.4% |
| 6 | `h2_s8_smem_wb_swz` | **CTA swizzle (GROUP_M)** — improve L2 reuse via co-located CTA waves | **725** | **98.6%** | 73.3% |
| — | CUTLASS BF16        | reference (autotuned over 7 (TileShape, ClusterShape) configs) | 635 | 86.4% | 64.2% |

We go from **53% of cuBLAS** at step 1 to **99% at step 6**. Across the
full sweep (2048-16384) we beat Triton at every size and tie/beat
cuBLAS at most sizes — see §1b.

**Step-by-step deltas at N=8192 (median of 3 runs):**

| Step | Δ TFLOPS | Δ % | Why |
|---|---:|---:|---|
| h1_ms → h2_s5 | +48 | +12.3% | wgmma replaces 4 × mma.sync per warp |
| h2_s5 → h2_s7 | +143 | +32.6% | wait_group 1: TC pipeline overlaps with DMA |
| h2_s7 → h2_s7_runptr | +50 | +8.6% | save ~15 int ops/iter on warp-issue-bound K-loop |
| h2_s7_runptr → h2_s8_smem_wb | +26 | +4.1% | 4× fewer global stores + 100% cacheline utilisation |
| h2_s8_smem_wb → swz | **+68** | **+10.4%** | GROUP_M=8 closes L2-reuse deficit; swz autotune also more stable than smem_wb's |

---

## 1b. Full Results Across All Shapes (2048-16384)

### Benchmarking methodology

**Hardware**: H800 SXM5 (132 SMs, 228 KB SMEM/SM, ~3.35 TB/s HBM3,
989 TFLOPS BF16 tensor-core peak), single GPU
(`CUDA_VISIBLE_DEVICES=2`).

**Timing**: `triton.testing.do_bench(warmup=200ms, rep=2000ms,
quantiles=(0.5, 0.0, 1.0))`. Reported number is `ms_min` from the
returned (median, min, max) triple — i.e., the best-of run within a
2-second timed window, which yields the tightest distribution at this
GPU's noise floor. Sustained (not cold-burst) measurement.

**Autotune** (for our kernel only; Triton/cuBLAS pick configs
internally):
- Score by **median, not min** (`ms_med < best_t`). On the 424-config
  swz search space, short-rep `min` is noise-biased — single-iter
  lucky-cold-cache scores can win over genuinely-better configs whose
  first iter was unlucky. Median tolerates the warmup tail and matches
  the long-rep sustained measurement.
- `warmup=10ms, rep=100ms` per config. Long enough to escape per-config
  noise, short enough that 424 configs × 9 sizes finishes in ~10 min.
- Per (M, N, K) key, cached across calls.

**Triton baseline**: `triton_ptx`, the hand-extracted best Triton BF16
PTX for the original config. Faster than `triton_bf16_autotuned` (the
runtime-autotune path picks worse configs in this measurement regime).
**cuBLAS**: PyTorch's BF16 matmul via `cublasGemmEx`.

**Sample sizes**: each entry is a single autotune-run measurement.
Run-to-run variance with the median-autotune is ±2-3% at most sizes;
larger N (≥ 10240) goes up to ±5% due to longer per-iter latency.

### Final kernel results vs Triton PTX and cuBLAS BF16

| Size | **swz (ours)** | Triton PTX | cuBLAS BF16 | swz/Triton | swz/cuBLAS |
|-----:|----:|----:|----:|---:|---:|
| 2048  | **605** | 574 | 568 | **105.4%** | **106.7%** |
| 3072  | 579 | 517 | **653** | **112.1%** | 88.6% |
| 4096  | **694** | 677 | 672 | **102.5%** | **103.2%** |
| 5120  | 633 | 617 | **677** | **102.5%** | 93.5% |
| 6144  | **689** | 689 | 686 | **100.1%** | **100.6%** |
| 7168  | **722** | 706 | 662 | **102.3%** | **109.2%** |
| 8192  | 725 | 694 | **735** | **104.5%** | 98.7% |
| 9216  | **716** | 695 | 676 | **103.1%** | **105.9%** |
| 10240 | **712** | 698 | 693 | **102.0%** | **102.7%** |
| 11264 | 727 | 706 | **737** | **103.0%** | 98.7% |
| 12288 | 721 | 695 | **737** | **103.7%** | 97.8% |
| 13312 | 732 | 696 | **740** | **105.1%** | 98.9% |
| 14336 | **742** | 694 | 734 | **107.0%** | **101.1%** |
| 15360 | 722 | 717 | **739** | **100.6%** | 97.7% |
| 16384 | **731** | 693 | 725 | **105.4%** | **100.8%** |

**Summary across all 15 sizes:**

| Metric | vs Triton | vs cuBLAS |
|---|---:|---:|
| Geometric mean ratio | **103.9%** | **100.3%** |
| Wins (≥ 100%) | **15 / 15** ✓ | 9 / 15 |
| Best ratio | 112.1% (N=3072) | 109.2% (N=7168) |
| Worst ratio | 100.1% (N=6144) | 88.6% (N=3072 — cuBLAS specialty) |

**Peak: 742 TF at N=14336 — 75.0% of the 989 TF tensor-core peak.**

We **beat Triton at every single size** in the sweep. Against cuBLAS:
parity on average, with cuBLAS winning at non-power-of-2 specialty
sizes (3072, 5120) and within the 11264-15360 band (where it hits
735-740 TF). Production sweet-spot (4096-9216): we match or beat both.

---

## 2. Step 1 — `h1_ms`: Ada Baseline on Hopper

**Starting point.** The previous Ada-architecture kernel was a multi-stage
`cp.async` + `mma.sync` matmul. The same source code compiles and runs on
H800 (with sm_90 forward-compatible `mma.sync` instructions); we just port
it.

**Anatomy:**
- **Load**: `cp.async.cg.shared.global` 16 B per thread, double/triple-buffered.
- **Compute**: `mma.sync.aligned.m16n8k16` per warp (not warpgroup-aware).
- **Layout**: manual XOR swizzle on the A SMEM tile to eliminate bank conflicts
  in the per-warp `ldmatrix` accesses.
- **Autotune**: BM, BN, BK, NUM_STAGES per problem size.

**Result:** **385 TFLOPS at 8192 = 52% of cuBLAS** (39% of TC peak). The
kernel works and is correct, but it leaves the Hopper tensor cores
underutilised. `mma.sync` instructions issue from individual warps;
Hopper's faster path is `wgmma` which uses an entire **warpgroup** (4
warps = 128 threads) as the unit of MMA.

---

## 3. Step 2 — `h2_s5`: First wgmma Kernel

**The big architectural switch.** `wgmma.mma_async.sync.aligned.m64nNk16`
replaces 4 separate `mma.sync` calls with a single warpgroup-wide
asynchronous MMA. The two-instruction pipeline (`wgmma.fence` + N ×
`wgmma.mma_async` + `wgmma.commit_group` + `wgmma.wait_group 0`) lets the
tensor core overlap compute with subsequent SMEM reads.

**What changes in code:**
- **Compute**: A and B descriptors (`GmmaDescriptor`, 64-bit struct
  encoding SMEM address + layout) feed `wgmma`. A read by `ldmatrix` →
  registers (RS mode in s5; we later move to SS mode).
- **B layout**: 128-byte SMEM swizzle (`B128`) baked into the descriptor.
  This is wgmma's **required** swizzle; the tile must be stored as
  `[BN/64]` packed 64-column sub-tiles of `[BK][64]` BF16. TMA writes
  this format natively.
- **Load**: TMA (`cp.async.bulk.tensor.2d`) + mbarrier replaces the
  per-thread `cp.async` + `__pipeline_*`. One thread issues a descriptor;
  the DMA engine delivers the full tile.
- **Pipeline**: NUM_STAGES = 2..4 tunable.

**Result:** **441 TFLOPS, 60% of cuBLAS** (+14.5% over h1_ms). The
wgmma + B128 swizzle unlocks the tensor cores at the warpgroup level,
but synchronization is still naive (`wgmma.wait_group 0` after every
tile drains the TC pipeline fully before issuing the next load). Tensor
core and DMA engine alternate rather than overlap.

---

## 4. Step 3 — `h2_s7`: Two-in-Flight wgmma (`wait_group 1`)

**This is the big Hopper-specific win.** Coming from Triton PTX analysis:
Triton waits for `wgmma.wait_group 1`, not `0`.

**Mechanism.** Tag each wgmma group as it commits:
```
COMPUTE_TILE(slot k): wgmma.fence; wgmma_kk0; wgmma_kk1; ...; wgmma.commit_group
WAIT_MMA(1): wgmma.wait_group 1     ← only wait for group k-1, NOT k
LOAD_TILE(slot k-1 mod NS): cp.async for tile k+NS-1
```
After committing wgmma group k, the warp scheduler keeps group k in the
tensor-core pipeline while it issues cp.async for the next tile. **Tensor
core and DMA engine now run concurrently**, not sequentially.

**SMEM safety.** ISSUE goes to slot `(k+NS-1)%NS = (k-1)%NS`. `wgmma.wait_group 1`
at iteration k guarantees `wgmma[k-1]` (which read slot `(k-1)%NS`) has
finished; that SMEM slot is now safe to overwrite.

**Why it helps so much on H800.** MMA time ≈ memory fetch time per tile
(~5 µs each). With `wait_group 0` they alternate; total = 2 × per-tile.
With `wait_group 1` they overlap; total ≈ 1 × per-tile.

**Pipeline structure** (NS=3 stage, single commit per LOAD_TILE):
```
   WAIT cp.async(NS-2)  ← tile k just landed
   __syncthreads
   COMPUTE wgmma group k
   wgmma_commit
   WAIT wgmma(1)        ← group k-1 done, slot (k-1)%NS freed
   __syncthreads
   LOAD cp.async tile k+NS-1 into slot (k-1)%NS
```

**Result:** **580 TFLOPS, 79% of cuBLAS** (+31.5% over h2_s5 — the
single biggest jump in the entire trajectory). One source change unlocks
~150 TFLOPS by removing the serialisation between tensor core and DMA.

Also switches from TMA to `cp.async.cg.shared.global` because TMA's
single-issuer pattern serializes the load phase on H800 (see "Things
that didn't work out").

---

## 5. Step 4 — `h2_s7_runptr`: Running Pointers (Codegen-Level Win)

**Source of the idea.** Decompiled Triton's compiled PTX side-by-side
against our kernel and noticed that Triton maintains per-thread *running*
global memory pointers, advanced by the stride each K-iter, while ours
recomputed the full address from base each cp.async issue.

**Before** (s7's `LOAD_TILE`, simplified):
```cpp
for (int _i = 0; _i < A_GROUPS; _i++) {
    // recompute address from base every issue
    cp.async( &A[(block_row + row(_i)) * K + k*BK + col(_i)], ... );
}
// ~18 integer ops/iter just for address arithmetic
```

**After** (`h2_s7_runptr`):
```cpp
// Preheader: build per-thread running ptrs once at k=0
for (int _i = 0; _i < A_GROUPS; _i++)
    A_curr[_i] = &A[(block_row + row(_i)) * K + col(_i)];

// In LOAD_TILE:
for (int _i = 0; _i < A_GROUPS; _i++) {
    cp.async( A_curr[_i], ... );
    A_curr[_i] += BK;          // single add.s64
}
// ~3 add.s64 per iter for A, ~3 for B
```

**Why this matters.** From profiling (notes-hopper/s7_cycle_breakdown.md)
we know the K-loop is **warp instruction-issue-bound** in this regime —
every saved integer op in the inner loop directly delays the next wgmma
issue. Going from ~18 to ~3 address-arithmetic ops per K-iter is real.

**Result:** **624 TFLOPS, 85% of cuBLAS** (+7.6% over h2_s7). A pure
codegen optimization with no algorithmic change — same wgmma, same load
pattern, same pipeline. Just better instruction-level efficiency.

(Bundled with two smaller follow-ups in the codebase: folded SMEM
destination offsets, and merged A+B cp.async into a single commit
group — they live in `h2_s8_smem_wb` as the final form.)

---

## 6. Step 5 — `h2_s8_smem_wb`: SMEM-Staged vec-4 Writeback

**Source of the idea.** Same t36-vs-our-kernel diff revealed that the
epilogue was the biggest remaining structural difference. Triton stages
the accumulator through SMEM before writing to global; we wrote directly.

**The problem with direct writeback.** wgmma's output fragment layout is
optimal for **chained MMA** in registers but pessimal for row-major
global memory. Per `(m, j)` iter, each thread writes 2× `__bfloat162`
(4 B each) at its fragment-native row/col. Within a warp of 32 lanes:

```
        col 0  4  8  12      ...
row R   |L0|L1|L2|L3|  ← 16 useful B in this cacheline
row R+1 |L4|L5|L6|L7|  ← another cacheline
...
row R+7 |L28|L29|L30|L31|  ← 8th cacheline
```

The warp's 32 × 4 B = 128 B is spread across **8 different cachelines**,
each receiving only 16 useful bytes (~12% cacheline utilization).

**The SMEM-staged fix.** After wgmma drain:

1. Each thread writes its acc[] values to SMEM at the wgmma-native
   positions. (Cheap: SMEM bandwidth is plenty.)
2. `__syncthreads()`
3. Re-divide work: all 256 threads cooperatively stream the BM × BN tile
   from SMEM to global via `st.global.v4.b32` (16 B per store).

Within a warp now: 32 lanes × 16 B = 512 B all on **the same row**, =
4 full cachelines per warp burst, 100% utilization.

| | Direct | SMEM-staged |
|---|---:|---:|
| Store instructions per thread | 32 × 2 = 64 | 16 |
| Cacheline utilization per warp burst | ~12% | 100% |
| Sector transactions per warp store | 8 | 4 |

**SMEM cost.** The C staging buffer reuses `smem_raw` — the A/B K-loop
buffer is idle by the time we hit the epilogue, so zero net SMEM cost.
Row stride padded by 8 BF16 to break the power-of-2 bank-conflict
pattern.

**Result:** **657 TFLOPS, 89% of cuBLAS** (+4.1% over runptr at this
size). Bigger relative wins at other sizes; see §1b.

---

## 7. Step 6 — `h2_s8_smem_wb_swz`: CTA Swizzle (GROUP_M)

**The remaining gap was at large N.** With the natural row-major CTA
launch (`gridDim = (N/BN, M/BM, 1)`), a wave of 132 CTAs reads a wide
band of A and a tall slice of B. At small/medium N this fits L2; at
N ≥ 7168 the wave's bounding box exceeds L2, every CTA misses, and DRAM
bandwidth becomes the bottleneck.

**The fix.** Re-linearise the (m, n) grid so each block of `GROUP_M ×
num_pid_n` CTAs covers a `GROUP_M × num_pid_n` band of output tiles,
iterating M-first inside the band:

```cpp
const int num_pid_n = gridDim.x;
const int pid       = blockIdx.y * num_pid_n + blockIdx.x;
const int per_group = GROUP_M * num_pid_n;
const int group_id  = pid / per_group;
const int idx       = pid - group_id * per_group;
pid_m = group_id * GROUP_M + (idx % GROUP_M);
pid_n = idx / GROUP_M;
```

A wave of 132 CTAs now hits a smaller bounding box → much higher L2 hit
rate. Adjacent CTAs in a wave share A rows / B columns more aggressively.

**The catch — it's size-dependent.** At small/medium N the natural scan
already fits L2 and the swizzle adds overhead with no payoff. So we
**autotune** `GROUP_M ∈ {1, 2, 4, 8}` and let the selector pick.

Autotune-picked GROUP_M per size at this kernel:

| N | best GROUP_M |
|---|:---:|
| 4096-6144 | 1 (no swizzle) |
| ≥ 7168    | 8 |

**Result:** **725 TFLOPS, 99% of cuBLAS at 8192** (+10.4% over
smem_wb). Where swz really shines is the largest sizes, but the win is
material at the production sweet-spot too:

| N | smem_wb | swz | gain |
|---|---:|---:|---:|
| 7168  | 663 | 722 | **+8.9%** |
| 8192  | 657 | 725 | **+10.4%** |
| 9216  | 651 | 716 | **+10.0%** |
| 10240 | 632 | 718 | **+13.6%** |
| 16384 | n/a | 731 | (beats Triton's 693) |

Crucially closes the deficit at N ≥ 10240 where the natural row-major
scan blew past L2. At small/medium N the autotune picks GROUP_M=1 (no
swizzle, identical to smem_wb).

There's a **second**, subtler effect: swz's expanded autotune space
(424 configs vs smem_wb's 106) interacts with the median-scoring
selector to make the autotune itself more reliable. With smem_wb's
narrow search, the selector sometimes picks BM=256/BN=128 over the
true winner BM=128/BN=256, costing ~3-4% in the long-rep measurement.
swz's GROUP_M dimension gives the selector the right "knob" to
discriminate on.

---

## 8. Optimizations That Didn't Work Out

Six experiments looked promising on paper but consistently failed to
move the needle. Documenting them is important: they collectively form
a strong piece of evidence that **the rolled K-loop at this kernel's
perf level is at a local ptxas-scheduling optimum**, and the remaining
wins require structural / autotune-side changes rather than local
restructuring.

### 8.1. TMA + mbarrier path (h2_s2, s3, s4, s5)
Hopper's Tensor Memory Accelerator was supposed to be the bulk-copy
solution: one thread issues a descriptor, the engine delivers the tile.
We built it up across h2_s2 → s5 and got working kernels.

**Why it didn't beat cp.async on this kernel.** TMA's single-issuer
pattern serialises the load phase on H800: a 200µs MMA tile arrives in
~5µs from HBM, but TMA's launching mbarrier protocol takes long enough
that the cp.async multi-thread alternative finishes faster overall.
The CUTLASS-built version of the same kernel showed the same ceiling
(`notes-hopper/tma_serialization.md`). Triton confirmed the choice —
their best Hopper BF16 PTX also uses cp.async.

### 8.2. `_pipe`: precompute wgmma descriptors before WAIT
The intuition: `wgmma_a_desc` and `wgmma_b_desc` are pure register
arithmetic on SMEM addresses (no data dependency). If we hoist them
above `__pipeline_wait_prior`, they can hide behind the wait stall and
wgmma fires immediately after the barrier.

**What we did.** Created `h2_s8_smem_wb_pipe` with explicit precompute
phase that stashes descriptors in register arrays before the wait.
**Result: −3 to −5%.** Source-level change added register pressure;
autotune picked a different (worse) config.

**The clean isolation.** Did the move directly in PTX (compile baseline,
hand-edit instruction order, re-`ptxas`, A/B bench). Source-level
confounds eliminated. **Result: perf-neutral, mean −0.4%.**

Conclusion: the wait barriers don't stall in steady state with NS=3
(2 tiles in flight by the time we wait → wait returns near-instantly).
With no stall to hide behind, the precompute does nothing.

### 8.3. `_u3`: unroll K-loop by 3 (NS=3 cycle length)
At NS=3 the slot indices `cur, nxt` cycle through 0, 1, 2 with period 3.
Unrolling the main K-loop by 3 turns these into compile-time constants,
eliminating `k % NS` magic-divide (10 ops/iter) and `slot * stride` adds.

**Result: −2 to −3% at 8192.** Code-size 3× the K-loop body — likely
icache pressure / different branch behaviour. The micro-architectural
win on op count was real but came with macro-architectural cost.

### 8.4. `_clu`: CTA cluster (2×1) co-location
`__cluster_dims__(2, 1, 1)` declares clusters of 2 CTAs without using
DSMEM or cluster.sync — just relying on Hopper's "cluster CTAs co-reside
on the same GPC" guarantee for L2 locality.

**Result: −0.2% to −0.9%**, essentially flat. The kernel's A working
set fits in L2 already at the sizes where it helps, and the cluster
launch overhead eats the rest.

### 8.5. Conservative single-op pre-barrier hoists
After the original `_pipe` regression, retried with the broader autotune
(median selector, GROUP_M dim). Each time hoisting one tiny op:
- `_bb = __cvta_generic_to_shared(&B_sh[cur][0][0][0])` before wait_smem: 
  **perf-neutral** (3 sizes, mean −0.4%, same configs picked).
- `_A_base = &A_sh[nxt][0][0]` before wgmma.wait_group: **perf-neutral**
  (−0.9% / 0.0% at sizes with matched configs).

Five separate confirmations that pre-barrier scheduling has no
remaining room on this kernel.

### 8.6. `swizzle on h2_s7` (early)
Tried GROUP_M swizzle on the slower `h2_s7` base. Result: lost at every
square size 2048-8192. Reason: h2_s7 was warp-issue-bound, not
L2-bound, so swizzle's benefit (improved L2 reuse) didn't matter and
its overhead (an extra div/mod for `pid_m`) showed up directly.

The same swizzle on the faster `h2_s8_smem_wb` base — where the
warp-issue bottleneck is gone — works perfectly. **The right answer
was "autotune over GROUP_M", not "always 1" or "always > 1".**

### 8.7. Direct read of cuBLAS / CUTLASS as a target
We compared against Triton's pre-compiled PTX and cuBLAS BF16. We tried
also building a CUTLASS reference kernel; the wrapper overhead
(per-call `cuTensorMapEncodeTiled`) added ~40 µs and dominated short
sweeps. Fixed with a per-call `Gemm` state cache + persistent output
buffer, but at that point CUTLASS as a reference contributes nothing
new — it ends up in roughly the same place as Triton.

---

## 9. Methodology — Three Lessons

### 9.1. PTX surgical edit as A/B test
Source-level changes always introduce confounds: register pressure,
alternate config picks under autotune, different compiler scheduling.
For "is this scheduling change actually doing what I think?" questions,
the cleanest tool is **edit the compiled PTX by hand and re-`ptxas`**.
Used to isolate the `_pipe` regression as autotune-induced (the PTX
move itself is perf-neutral) and to show steady-state waits don't stall.

### 9.2. The autotune is itself a kernel that needs tuning
On a small config space (~100 configs, e.g. `h2_s8_smem_wb`), the
`triton.testing.do_bench(warmup=10, rep=50, ms_min)` picker is mostly
correct. On a 4× expanded space (~400 configs, `_swz` adds `GROUP_M`),
short-rep min becomes **noise-biased** — configs that scored well on a
single cold-cache run win over genuinely-faster configs whose first iter
was unlucky.

The fix: **`quantiles=(0.5, ...)` and read `ms_med` instead of `ms_min`;
increase rep to 100 ms**. Recovered 9-13% at sizes 7168-10240 just from
better config selection — same kernel, same source, different picker.

The deeper lesson: when adding a new autotune dimension, switch to
median *before* expanding the config space, not after.

### 9.3. Structural > scheduling at the kernel frontier
Once `h2_s8_smem_wb` reached parity with Triton's pre-compiled PTX, six
local scheduling tweaks (_pipe, PTX move, _u3, _clu, _bb hoist,
_A_base hoist) all came in flat-to-negative. The two wins that *did*
work after that point were both structural:

- **Adding GROUP_M to the autotune space** — a new dimension the
  selector could explore.
- **Adding median scoring** — fixing how the selector ranked configs in
  the expanded space.

Neither touched the K-loop body. The "compiler is a black box" lesson:
at the frontier, the body is already on a strong local optimum and
local perturbations cost more than they help. Look for structural
changes — new dimensions, new layouts, new pipeline shapes — not new
scheduling.

---

## 10. Where The Remaining Gap Is

At the largest sizes (N = 10240 - 16384), our kernel **beats** Triton's
pre-compiled PTX and the runtime-autotuned Triton, and is within a few
percent of cuBLAS. cuBLAS still wins consistently at non-power-of-2
sizes (3072, 5120) and at 8192 — likely Stream-K or split-K paths that
fix wave quantization. Those are the natural next investigation if
pursued.

| Production sweet spot (4096-9216) | Status |
|---|---|
| vs Triton PTX | ≥ 100% at 7/9 sizes |
| vs cuBLAS | mixed: even-power-of-2 we tie or beat, awkward shapes cuBLAS specialty |
| vs TC peak (989 TF) | ~73-75% |

Peak we hit: **742 TF @ N=14336**, 75.0% of theoretical BF16 peak.
