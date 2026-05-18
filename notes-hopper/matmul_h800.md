# GEMM Optimization on H800 (Hopper, sm_90a)

## Hardware

| Property | H800 SXM5 | RTX 4090 (reference) |
|----------|-----------|----------------------|
| Architecture | Hopper (sm_90a) | Ada Lovelace (sm_89) |
| SMs | 132 | 128 |
| SMEM / SM | 228 KB | 128 KB |
| Registers / SM | 65 536 × 32-bit | 65 536 × 32-bit |
| Peak BF16 TC | ~989 TFLOPS | ~165 TFLOPS |
| Memory | 80 GB HBM3 | 24 GB GDDR6X |
| Bandwidth | 3.35 TB/s | 1.01 TB/s |
| L2 | 50 MB | 72 MB |

---

## Key New Hopper Instructions

### wgmma (Warp-Group MMA)
`wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16`

- Operates on an entire **warpgroup** (4 warps = 128 threads) as a unit.
- M is fixed at 64; N ∈ {8, 16, …, 256} in multiples of 8; K = 16.
- A operand: **registers** (loaded via `ldmatrix.x4` across the warpgroup).
- B operand: directly from **SMEM** via a 64-bit GmmaDescriptor — hardware reads it,
  threads never touch B registers.
- Result: F32 registers (BN/2 per thread) distributed across the 128-thread warpgroup.
- Fencing: `wgmma.fence` before first call; `wgmma.commit_group` +
  `wgmma.wait_group 0` after the kk loop.

### TMA (Tensor Memory Accelerator)
`cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes`

- One thread issues a hardware descriptor; DMA engine delivers the full 2D tile.
- Descriptor (`CUtensorMap`) encodes tensor shape, stride, swizzle — built CPU-side
  via `cuTensorMapEncodeTiled` and passed as `__grid_constant__` kernel parameter.
- Synchronised via `mbarrier` (arrive_expect_tx + test_wait) instead of
  `__pipeline_*`.
- 128B swizzle (`boxDim[0]=64 BF16`) written by TMA matches the B128 GmmaDescriptor
  that wgmma expects — they are designed as a pair.

### GmmaDescriptor (B tile for wgmma)
64-bit struct encoding the B SMEM tile layout:

```
[13:0]  start_address_ = smem_addr >> 4
[29:16] leading_byte_offset_ (LBO): 8×BK for BN>64, 0 for BN=64
[45:32] stride_byte_offset_ (SBO): 64
[63:62] layout_type_: 1 = B128 (128B swizzle)
```

Note: `layout_type_=0` (INTERLEAVE) causes illegal memory access with standard
lbo/sbo values on this hardware — B128 is the working mode even for pitch-linear-
written B tiles.

---

## Kernel Implementations

All kernels are in `mymatmul/gpu/hopper/`.

### tc5_regpruned on H800 (baseline)
Ada kernel (`mma.sync` + `cp.async`) ported to sm_90. Pre-compiled sm_90 cubins
work as-is. Establishes the baseline — shows `mma.sync` reaches ~54% of cuBLAS on
H800, motivating Hopper-specific optimizations.

### h1_ms — multi-stage tc5 (cp.async, mma.sync, NS stages)
`matmul_h1_ms.py` / `_matmul_h1_ms.cu`

- tc5_lb with `NUM_STAGES ∈ {2,3,4,5}` as a tunable template parameter.
- NS=2 is identical to tc5_lb. Larger NS hides more cp.async latency.
- Uses H800's 200 KB SMEM budget (vs 100 KB on 4090) — more configs eligible.
- Compiled per LB value (1–4); Python register-estimate pruning filters infeasible configs.

### h2_s1 — TMA + mbarrier, mma.sync, no swizzle
`matmul_h2_s1.py` / `_matmul_h2_s1.cu`

- Replaces `cp.async` + `__pipeline_*` with TMA + `mbarrier`.
- Compute unchanged: `ldmatrix` + `mma.sync`.
- **Neither A nor B has XOR swizzle** — both SMEM layouts are linear row-major.
  Intentional: isolates TMA plumbing from swizzle correctness.
- Bank conflicts on both A and B `ldmatrix` calls limit performance; showed that
  TMA alone (without swizzle) doesn't help vs cp.async.
- Uses `cuda.bindings.driver` (cuda-python) for host side; TMA descriptors passed
  as `__grid_constant__` by value.

### h2_s2 — TMA + mbarrier, wgmma, BM=64 fixed
`matmul_h2_s2.py` / `_matmul_h2_s2.cu`

First working wgmma kernel. Kept deliberately simple to validate the wgmma + swizzle combination.

- **BM=64 hardcoded** — one warpgroup (128 threads) covers exactly one 64-row output tile.
  No M-loop, no multi-warpgroup, no template parameter for BM.
- Replaces `ldmatrix-B + mma.sync` with `wgmma` reading B directly from SMEM descriptor.
- **B**: TMA SWIZZLE_128B (boxDim=64 BF16); stored as BN/64 packed sub-tiles of [BK][64].
- **A**: XOR swizzle for BK=64 only; no swizzle for BK=16/32 (64B/32B XOR breaks
  ldmatrix's 16-byte chunk alignment — BK<64 has A bank conflicts).

### h2_s3 — TMA + wgmma + multi-warpgroup (canonical Hopper kernel)
`matmul_h2_s3.py` / `_matmul_h2_s3.cu`

Generalises h2_s2 from 1 fixed warpgroup to multiple.

- **`NUM_WG ∈ {1, 2}`** as a template parameter; BM = NUM_WG × 64.
  With NUM_WG=2: 2 warpgroups share the same B tile, each owns 64 rows of A/C.
- BM is now a free parameter (64 or 128), vs BM=64 hardcoded in h2_s2.
- Everything else unchanged from h2_s2: same TMA loading, same B128 swizzle,
  same wgmma RS mode, same 2-stage pipeline.
- Single cubin with LB=1; cuda-python (`cuda.bindings.driver`) for all host ops.

### h2_s4 — TMA + wgmma + M-loop (BM up to 256)
`matmul_h2_s4.py` / `_matmul_h2_s4.cu`

- Extends h2_s3: `BM` is now a free parameter, not locked to `NUM_WG × 64`.
- `M_ITERS = BM / (NUM_WG × 64)` — each warpgroup issues M_ITERS wgmma calls
  per kk step, each covering 64 rows. With BM=256, WG=2: M_ITERS=2.
- Accumulator: `float acc[M_ITERS][BN/2]` per thread.
- Doubles SMEM usage for A vs h2_s3, raising arithmetic intensity.
- 2-stage pipeline (NS hardcoded).

### h2_s5 — TMA + wgmma + M-loop + NUM_STAGES
`matmul_h2_s5.py` / `_matmul_h2_s5.cu`

- Extends h2_s4: `NUM_STAGES ∈ {2, 3, 4}` is now a tunable template parameter.
- Deeper pipelines keep more TMA transfers in flight, hiding mbarrier::wait latency.
- SMEM: `A[NS][BM][BK]`, `B[NS][BK][BN]`, `mbar[NS]`.
- Prologue issues `NS-1` tiles; main loop issues one tile per iteration;
  drain (unrolled) processes the last `NS-1` tiles.
- Profiling: 26% membar stalls (mbarrier::wait), 186 regs/thread.

### h2_s6 — cp.async + wgmma SS mode
`matmul_h2_s6.py` / `_matmul_h2_s6.cu`

Inspired by Triton PTX analysis showing Triton uses cp.async (not TMA) + wgmma SS mode.

- **SS mode**: both A and B read from SMEM via `GmmaDescriptor` — no `ldmatrix` for A.
  A descriptor: `layout_type=B32/B64/B128` (for BK=16/32/64), `LBO=0`, `SBO=BK`.
  Extra `transA=0` parameter in wgmma PTX vs RS mode.
- **A swizzle** now works for all BK: XOR formula `(col/8) ^ ((row/(64/BK)) % (BK/8))`,
  same as h1_ms. In RS mode only BK=64 worked (B64/B32 XOR broke ldmatrix 16B alignment).
- **B sub-tile format**: `[NS][BN/64][BK][64]` with `row%8` XOR per sub-tile (same as h2_s5).
- Replaces TMA+mbarrier with `cp.async` + `__pipeline_wait_prior` — simpler synchronisation.
- Result: **slower than h2_s5** (383–407 vs 427–441 TFLOPS). More membar stalls (40% vs 26%)
  from `__pipeline_wait_prior`, and SS mode uses more registers (202 vs 186) due to
  descriptor intermediates being kept live across unrolled kk iterations.

### h2_s7 — cp.async + wgmma SS + wgmma.wait_group 1
`matmul_h2_s7.py` / `_matmul_h2_s7.cu`

Key insight from Triton PTX analysis: `wgmma.wait_group 1` instead of 0.

- After committing wgmma group k, only wait for group k-1. Group k stays in the tensor
  core pipeline while cp.async loads the next tile. Both run concurrently.
- `wgmma.fence` at the start of each tile acts as the acc[] ordering barrier — the
  hardware serialises group k+1's reads of acc after group k's writes, without the CPU
  having to wait for group k to finish.
- **ISSUE placed after `wait_group 1`** (not before compute): `wait_group 1` at iter k
  proves wgmma[k-1] has released its SMEM slot `(k-1)%NS`, making it safe to overwrite.
- Loop restructured: `wait_prior(NS-2)` instead of `(NS-1)` to guarantee the current
  tile is ready even though ISSUE moved to the end of the iteration.
- **`wgmma.wait_group 0` only at the epilogue** to drain the final group.
- Best config: `BM=128, BN=256, BK=64, WG=2, NS=3` (M_ITERS=1).

### triton_ptx — pre-compiled Triton BF16 kernel
`matmul_triton_ptx.py`

Triton's best Hopper BF16 kernel loaded directly from PTX: BM=128, BN=256, BK=32, NS=4, NW=8.
Uses `ptxas` to compile to cubin; launched via `cuLaunchKernel`. Beats cuBLAS at large N.
Used as a reference target for reverse-engineering.

PTX analysis (see `notes-hopper/triton_ptx_analysis.md`):
- **No advanced Hopper features**: no TMA, no CTA clusters, no distributed SMEM.
- `cp.async.cg` (cache-global, bypasses L1) + `cp.async.wait_group` for tile loading.
- `wgmma SS mode` (both A and B from SMEM descriptors) + `wgmma.wait_group 1`.
- `ldmatrix` appears only in the epilogue to rearrange acc data in SMEM before the
  global C store — not used for A/B loading.
- The 87% SM SoL vs our 68% comes from BK=32 (smaller wgmma groups drain faster,
  wait stalls 12% vs 24%), not from any hardware feature we lack.

---

## Benchmark Results (H800 GPU2, BF16, square M=K=N)

Measurement: `triton.testing.do_bench` (warmup 10 reps, timed 50 reps), median time.
All TFLOPS = 2·M·N·K / time.

### Performance (TFLOPS)

| Size | h2_s5 | h2_s6 | **h2_s7** | Triton PTX | cuBLAS BF16 | h2_s7/cuBLAS |
|------|:-----:|:-----:|:---------:|:----------:|:-----------:|:------------:|
| 4096 | 427 | 407 | **574** | 677 | 672 | 85% |
| 6144 | 436 | 386 | **560** | 686 | 681 | 82% |
| 8192 | 440 | 397 | **586** | 697 | 694 | 84% |
| 10240 | 439 | 383 | **585** | 712 | 687 | 85% |

Triton PTX = pre-compiled Triton kernel (BM=128, BN=256, BK=32, NS=4, NW=8).
h2_s6 is slower than h2_s5 despite SS mode — see observations below.

### Best Configs Selected by Autotuner

**h2_s7**: always `BM=128, BN=256, BK=64, WG=2, NS=3` (M_ITERS=1).

**h2_s5**: always `BM=256, BN=128, BK=64, WG=2, M_ITERS=2`:

| Size | NS |
|------|----|
| 4096–6144 | 3 |
| 8192–10240 | 4 |

### Progression of optimizations

| Step | Kernel | Key addition | 8192 TFLOPS | vs cuBLAS |
|------|--------|-------------|:-----------:|:---------:|
| 1 | tc5_regpruned | Ada baseline on H800 | 362 | 52% |
| 2 | h1_ms | multi-stage cp.async | 368 | 53% |
| 3 | h2_s3 | TMA + wgmma + 2 warpgroups | 418 | 60% |
| 4 | h2_s4 | larger M tile (BM=256, M_ITERS=2) | 434 | 63% |
| 5 | h2_s5 | deeper pipeline (NS=3/4) | 441 | 64% |
| 6 | h2_s6 | cp.async + wgmma SS mode | 397 | 57% |
| 7 | **h2_s7** | **wgmma.wait_group 1** | **586** | **84%** |
| — | Triton PTX | reference (BK=32, wait_group 1) | 697 | 100% |
| — | cuBLAS | full Hopper optimization | 694 | 100% |

---

## Key Observations

### What each stage contributes

| Kernel | Key addition | 4096 TFLOPS | 8192 TFLOPS |
|--------|-------------|:-----------:|:-----------:|
| tc5_regpruned | Ada baseline on H800 | 378 | 362 |
| h1_ms | multi-stage cp.async pipeline | 382 | 368 |
| h2_s1 | TMA (no swizzle, mma.sync) | ~188 | ~203 |
| h2_s2 | TMA + wgmma, BM=64 | ~314 | ~290 |
| h2_s3 | + 2 warpgroups (BM=128) | 361 | 415 |
| h2_s4 | + M-loop: BM=256, M_ITERS=2 | 389 | 434 |
| h2_s5 | + NUM_STAGES=3/4 pipeline | 427 | 441 |
| h2_s6 | cp.async + wgmma SS mode | 407 | 397 |
| **h2_s7** | **+ wgmma.wait_group 1** | **574** | **586** |
| Triton PTX | reference target | 677 | 697 |
| cuBLAS | full Hopper optimization | 672 | 694 |

h2_s1 regresses vs tc5 because removing the B XOR swizzle (for correct linear SMEM)
causes ldmatrix bank conflicts that outweigh TMA's load-instruction savings.

### Why h2_s6 is slower than h2_s5

h2_s6 (cp.async + wgmma SS mode) was expected to improve over h2_s5 by:
- Eliminating ldmatrix for A → fewer registers
- Enabling BK=32 with correct swizzle → smaller SMEM → more CTAs/SM

In practice neither benefit materialised:
- Registers: 202/thread (h2_s6) vs 186 (h2_s5). SS mode descriptors keep more
  intermediates live across the unrolled kk loop.
- BK=32 configs didn't win autotuning — BK=64 still optimal.
- Membar stalls rose from 26% (h2_s5) to 40% (h2_s6): `__pipeline_wait_prior`
  generates more stalls than TMA's mbarrier protocol.

### Why h2_s7 is +34% over h2_s5 (the big win)

The entire gain comes from `wgmma.wait_group 1` instead of `0`.

With `wait_group 0` (h2_s6/h2_s5): after committing wgmma[k], the warp scheduler
stalls until the tensor core finishes. The tensor core and DMA engine do not overlap.

With `wait_group 1` (h2_s7): after committing wgmma[k], only wait for wgmma[k-1].
Group k stays in the tensor core pipeline while the warp scheduler issues cp.async for
the next tile. Tensor core and DMA run concurrently.

This works because on H800, MMA time >> memory fetch time (compute-bound regime):
each tile takes ~5 μs on tensor cores; HBM fetch also ~5 μs. With NS=3, the fetch
for tile k+2 has 2 full compute iterations (~10 μs) to complete before it's needed.
`wait_group 1` eliminates the idle period between tensor core operations.

SMEM safety: ISSUE goes to slot `(k+NS-1)%NS = (k-1)%NS`. `wait_group 1` at iter k
guarantees wgmma[k-1] (which read slot `(k-1)%NS`) is fully done. No race.

### Three synchronisation primitives

| Primitive | Scope | Used for |
|-----------|-------|----------|
| `__pipeline_wait_prior(N)` | per-thread | ensure this thread's own cp.async data is in SMEM |
| `wgmma.wait_group N` | per-warpgroup | ensure this warpgroup's wgmma acc writes are done |
| `__syncthreads()` | CTA-wide | bridge per-thread/per-warpgroup guarantees to all 256 threads |

`__syncthreads()` is needed after both waits: `wait_prior` only knows about the
current thread's data; `wait_group` only knows about the current warpgroup's wgmma.
Without the CTA barrier, one warpgroup could race ahead and read incomplete SMEM.

### h2_s7 vs Triton PTX profiling comparison

From NCU profiling at N=4096:

| Metric | h2_s7 | Triton PTX |
|--------|-------|-----------|
| SM SoL | 67.6% | 86.7% |
| Registers/thread | 202 | 168 |
| SMEM | 144 KB | 96 KB |
| Occupancy | 12.5% | 12.5% |
| wait stalls (wgmma) | 24% | 12% |
| long_sb stalls | 5% | 32% |
| barrier stalls | 5% | 28% |
| LD bank conflicts | 0 | 12 420 |

Despite fewer registers and less SMEM, Triton achieves the same 12.5% occupancy —
both are register-limited to 1 CTA/SM. Triton's bank conflicts (12 420) are higher
than ours (0). The gap in SM SoL is not fully explained.

---

## Implementation Notes

### TMA descriptor passing
Descriptors are built CPU-side via `cuTensorMapEncodeTiled` (cuda-python), passed
to kernels as `__grid_constant__ TmaDesc` by value — no GPU allocation needed.
The runtime copies 128 bytes into per-CTA constant memory at launch.

### B SMEM layout for wgmma
B is stored as `BN/64` packed sub-tiles of `[BK][64]` BF16, back-to-back:
- TMA writes with hardware 128B swizzle (boxDim[0]=64 enforced by hardware constraint)
- wgmma descriptor LBO = `8×BK` (encodes stride between sub-tiles)
- kk-step advancement: `base + kk × 2048 bytes` (16 rows × 64 BF16-wide × 2 bytes)

### A SMEM swizzle limitation
A uses TMA 128B swizzle only when BK=64 (128-byte rows → `A_SWZ_PERIOD=8`).
For BK=16/32: 32B/64B TMA swizzle modes operate at 8-byte granularity, breaking
ldmatrix's 16-byte chunk alignment → data corruption. These use SWIZZLE_NONE with
bank conflicts. Full A swizzle across all BK requires reformatting A to always be
a multiple of 128 bytes wide (CUTLASS minimum atom size approach).

### GmmaDescriptor layout_type encoding
Empirically verified: `layout_type_=0` (bits 63:62 = 00) causes illegal memory
access; `layout_type_=1` (bit 62 set) works. The hardware encoding is opposite to
what the CUTLASS LayoutType enum name "INTERLEAVE" suggests — bit 62 = 1 is the
working pitch-linear/B128 mode.

### cuda-python vs PyCUDA
H2+ kernels use `cuda.bindings.driver` throughout:
- `cuTensorMapEncodeTiled` → returns `CUtensorMap` with `.opaque` field (16×uint64)
- `cuLaunchKernel` → `kernelParams` must be `np.array([ctypes.addressof(...)], dtype=np.intp)`
- `cuuint32_t` / `cuuint64_t` wrappers required for descriptor fields

---

## Post-s7 Kernel Additions

After h2_s7, work focused on closing the remaining gap to Triton via PTX-level
side-by-side comparison of the K-loop body and the epilogue. Each step is one
isolated change on top of the previous one.

### h2_s7_runptr — per-thread running pointers (+9% over s7)
`_matmul_h2_s7_runptr.cu`

From t36-vs-s7 decompile diff: Triton maintains per-thread running gmem
pointers `A_curr[i] / B_curr[i]` and advances them by stride each K-iter.
s7 instead recomputes `&A[(block_row+r)*K + k*BK + c]` per cp.async (~18
integer ops/iter for address compute). Replacing with running pointers
saved ~15 ops/iter on a warp-issue-bound kernel.

Sustained (BM=128 BN=256 BK=64 NS=3): **565 → 615 TF at 4096 (+8.9%)**.

### h2_s8 — folded SMEM destination offsets (+1.8%)
`_matmul_h2_s8.cu`

PTX inspection of runptr showed each cp.async's SMEM destination was built
as a chain of separate `add.s32`:
```
A:  add(A_stage_base, _r*BK*2);    add(prev, _sc*2)               (2 adds)
B:  add(B_stage_base, _st*BK*64*2); add(prev, _kr*64*2);
    add(prev, _sc*2)                                                (3 adds)
```
Although `_r,_sc,_st,_kr` are loop-invariant per (tid, _i), ptxas does NOT
CSE them. Pre-folding into single `A_sh_off[A_GROUPS] / B_sh_off[B_GROUPS]`
ints in the preheader collapses each chain to one add per cp.async issue.
K-loop add.s32 count: 45 → 26 per iter (−19 ops). +1.8%.

### h2_s8_smem_wb — SMEM-staged vec-4 epilogue (+7-9%)
`_matmul_h2_s8_smem_wb.cu` (was `_smem_epi`; renamed for brevity)

t36 epilogue uses SMEM round-trip: write acc → SMEM at wgmma-fragment
positions, syncthreads, all 256 threads cooperatively stream BM*BN to
global via `st.global.v4.b32` (16 B/store, ideal coalescing). Old direct
epilogue: each thread writes 2× `__bfloat162` per j-stripe (32 stores/
thread). New: 16 stores/thread, half the issue count, much better
coalescing of fragment-native layouts.

Bigger than expected: +9% at 4096 (621 → 677 TF), bringing us to Triton
parity at 4096-8192 for the first time. Also merged A+B cp.async into
a single commit group (halves commit count, perf-neutral, simpler).

Reuses smem_raw (idle after WAIT_MMA(0)); row stride padded by 8 BF16
to break power-of-2 bank-conflict pattern.

### Local-restructuring experiments (all flat or negative)

After landing smem_wb, several scheduling/microarchitecture tweaks were
tried — all **failed** to move the needle. Documented as negative results.

| Variant | Idea | Result |
|---|---|---|
| `_pipe` | Split COMPUTE_TILE into MMA_PRECOMPUTE (descriptors) + MMA_ASYNC; precompute hidden behind WAIT_SMEM stall | −3 to −5% at source level; perf-neutral when verified at PTX level (surgical instruction relocation). cp.async.wait_group does not stall in steady state with NS=3 |
| `_u3` | Unroll K-loop by 3 (NS=3 cycle length) so slot indices 0/1/2 become compile-time constants, eliminating `k%3` magic-div and `slot*stride` adds | Consistent −2 to −3% at 8192. Code-size 3× perhaps eats icache headroom |
| `_clu` | Add `__cluster_dims__(2, 1, 1)` to launcher for L2 co-location | Flat (−0.2 to −0.9% across sizes). Cluster co-location heuristic doesn't help when working set already fits L2 |

The pattern: at h2_s8_smem_wb's perf level, the rolled K-loop is already
at a ptxas-scheduling local optimum. Local scheduling perturbations
(precompute slots, manual unroll, instruction reorder) cost ~2-5% by
disturbing register pressure or warp issue patterns the compiler had
optimized around. **PTX surgical edit + ptxas re-compile** is the cheapest
A/B tool for separating "is the stall real?" from "did the compiler do
something else under the source change?"

### h2_s8_smem_wb_swz — Triton-style GROUP_M block remap (the final win)
`_matmul_h2_s8_smem_wb_swz.cu`

Source body byte-identical to h2_s8_smem_wb; only the block_row/block_col
derivation changes:
```cpp
template<int BM, int BN, int BK, int NUM_WG, int NUM_STAGES, int GROUP_M>
...
if constexpr (GROUP_M <= 1) {
    pid_m = blockIdx.y; pid_n = blockIdx.x;
} else {
    const int num_pid_n = gridDim.x;
    const int pid       = blockIdx.y * num_pid_n + blockIdx.x;
    const int per_group = GROUP_M * num_pid_n;
    const int group_id  = pid / per_group;
    const int idx       = pid - group_id * per_group;
    pid_m = group_id * GROUP_M + (idx % GROUP_M);
    pid_n = idx / GROUP_M;
}
```
Each block of `GROUP_M × num_pid_n` CTAs covers a `GROUP_M × num_pid_n`
band of output tiles, iterating M-first. A wave of 132 CTAs then hits a
smaller (M_tiles × N_tiles) bounding box → much higher L2 hit rate at
sizes where the natural row-major scan blows past L2 between waves.

Autotunes over GROUP_M ∈ {1, 2, 4, 8}. The selector reliably picks **GM=1**
at N=4096-6144 (no swizzle, identical to wb) and **GM=8** at N=2048,
3072, and N≥7168 (boundary regimes where L2 reuse matters).

Closes the 10240 deficit entirely (91% → 99% of Triton) and unlocks
+9-13% at N=7168-9216.

### Autotune methodology: median, not min

The min-of-do_bench selector was found to be **noise-biased** on the
expanded swz config space (424 vs the original 106 configs). Switching
to **median + 100ms rep budget** recovered 9-13% at N=7168-9216 by
finding configs that min had systematically mispicked:

```
                    min picked              median picked       sustained Δ
N=7168    BM=128/BN=256 NS=3 GM=1   BM=128/BN=256 NS=4 GM=8     +5.0%
N=8192    BM=256/BN=128 NS=3 GM=1   BM=128/BN=256 NS=3 GM=8    +10.5%
N=9216    BM=256/BN=128 NS=3 GM=1   BM=128/BN=256 NS=3 GM=8    +11.0%
```

Why min is biased: GROUP_M-swizzled configs change the spatial cache
footprint and need a few warmup launches; min(50ms rep) reports the
single luckiest cold-cache launch and rewards near-tied configs with
"unlucky" first iters that happen to be fast. Median tolerates the
warmup tail and reports steady-state behavior, which matches the
production (long-rep) launcher.

The effect scales with autotune space size: small spaces (~100 cfgs,
e.g. wb) are mostly OK under min; large spaces (~400+ cfgs, e.g. swz)
systematically mispick. **Switch to median *before* expanding the
config space**, not after.

---

## Final Results (h2_s8_smem_wb_swz vs Triton vs cuBLAS)

Sustained BF16 matmul on H800, square M=K=N, `triton.testing.do_bench`
warmup=200ms rep=2000ms.

| Size | **swz** | Triton PTX | cuBLAS BF16 | swz/Triton | swz/cuBLAS |
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

**Summary across all 15 sizes (2048-16384):**

| Metric | vs Triton | vs cuBLAS |
|---|---:|---:|
| Geometric mean ratio | **103.9%** | **100.3%** |
| Wins (≥100%) | **15 / 15** ✓ | 9 / 15 |
| Best ratio | 112.1% (N=3072) | 109.2% (N=7168) |
| Worst ratio | 100.1% (N=6144) | 88.6% (N=3072 — cuBLAS specialty) |

**Peak: 742 TF at N=14336 — 75% of the 989 TF tensor-core peak.**

We beat Triton at every single size in the sweep. cuBLAS wins at its
non-power-of-2 specialty sizes (3072, 5120) and in the 11264-15360 band
where it hits 735-740 TF; we win or tie elsewhere.

---

## Final Progression Table

| Step | Kernel | Key change | 4096 TF | 8192 TF |
|------|--------|-----------|:-------:|:-------:|
| 1  | tc5_regpruned | Ada baseline on H800 | 378 | 362 |
| 2  | h1_ms | multi-stage cp.async | 382 | 368 |
| 3  | h2_s3 | TMA + wgmma + 2 warpgroups | 361 | 415 |
| 4  | h2_s4 | larger M tile (BM=256) | 389 | 434 |
| 5  | h2_s5 | deeper pipeline (NS=3/4) | 427 | 441 |
| 6  | h2_s7 | wgmma.wait_group 1 (overlap MMA + load) | 574 | 586 |
| 7  | h2_s7_runptr | per-thread running gmem ptrs | 615 | 605 |
| 8  | h2_s8 | folded SMEM dest offsets | 622 | 639 |
| 9  | h2_s8_smem_wb | SMEM-staged vec-4 epilogue | 677 | 681 |
| 10 | **h2_s8_smem_wb_swz** | **+ Triton-style GROUP_M autotune dim** | **694** | **725** |
| —  | Triton PTX | reference target | 677 | 694 |
| —  | cuBLAS BF16 | NVIDIA's own | 672 | 735 |

Total improvement: tc5_regpruned → swz = **378 → 694 TF at 4096** (+84%).

---

## Methodology Notes (added during post-s7 work)

### Compiler is a black box

Five separate experiments after smem_wb (`_pipe` precompute, PTX surgical
move, `_u3` unroll-by-3, `_clu` CTA cluster, B-side base+offset) all came
in flat-to-negative even when the source-level intent ("reduce ops",
"hide compute behind stall", "improve L2 locality") was clearly correct.

The pattern: at h2_s8_smem_wb's perf level, ptxas's default schedule on
the rolled NS=3 K-loop is a strong local optimum, and *local* source
perturbations sit on a flat-or-rising plateau around it. Things that
worked were **structural**: adding a missing autotune dimension
(GROUP_M), or making the autotune *reliable* (median).

### PTX surgical edit as A/B test

For "is this stall real?" questions, the cheapest, cleanest tool is to
edit the compiled .ptx file by hand (move N instructions), re-compile
with `ptxas -O3`, and bench. This eliminates every compiler-side
confound (register pressure, alternate config selection, schedule
disturbance) and isolates the pure effect of the reorder. Used at:
verifying that pre-barrier descriptor compute is perf-neutral (the
post-barrier wait does not stall in steady state with NS=3).

### Autotune is the second kernel

Realised late: **the autotune is itself a kernel that needs tuning**.
For a 400+ config space, the difference between min and median selectors
is 10% in delivered perf. For a 100-config space it's ~0. The autotune
budget (warmup/rep) and the score statistic (min/median/quantile) are
first-class hyperparameters.

A more principled fix than median-alone would be a two-stage autotune:
short pass for top-K, long re-test among those to pick the winner. Not
implemented yet — median + 100ms rep recovered most of the gap.
