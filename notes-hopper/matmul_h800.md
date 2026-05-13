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

### h2_s1 — TMA + mbarrier, mma.sync, no B swizzle
`matmul_h2_s1.py` / `_matmul_h2_s1.cu`

- Replaces `cp.async` + `__pipeline_*` with TMA + `mbarrier`.
- Compute unchanged: `ldmatrix` + `mma.sync`.
- B SMEM is linear (no XOR swizzle) — intentional to isolate TMA plumbing.
- Bank conflicts on B `ldmatrix` limit performance; showed TMA alone doesn't help.
- Uses `cuda.bindings.driver` (cuda-python) for host side; TMA descriptors passed
  as `__grid_constant__` by value.

### h2_s2 — TMA + mbarrier, wgmma, BM=64 fixed
`matmul_h2_s2.py` / `_matmul_h2_s2.cu`

- Adds B128 TMA swizzle and wgmma (replaces ldmatrix-B + mma.sync).
- Single warpgroup (BM=64), NUM_WG=1 only.
- B stored as BN/64 packed sub-tiles (each 64-BF16-wide × BK-tall), back-to-back.
- A: XOR swizzle matching TMA 128B for BK=64; no swizzle for BK=16/32 (64B/32B
  swizzle breaks ldmatrix 16-byte chunk alignment).

### h2_s3 — TMA + wgmma + multi-warpgroup (canonical Hopper kernel)
`matmul_h2_s3.py` / `_matmul_h2_s3.cu`

- Extends h2_s2 with `NUM_WG ∈ {1, 2}` (BM = NUM_WG × 64).
- Both warpgroups share the same B tile; each owns 64 rows of A/C.
- **This is the canonical Hopper kernel**: TMA + wgmma + 2 warpgroups.
- Single cubin with LB=1; cuda-python (`cuda.bindings.driver`) for all host ops.

### h3 — cp.async + wgmma (hybrid, educational)
`matmul_h3.py` / `_matmul_h3.cu`

- Uses h1_ms's cp.async pipeline but replaces mma.sync with wgmma.
- B written with `row%8` XOR via cp.async (same physical layout as TMA 128B) so
  B128 GmmaDescriptor applies.
- Confirms wgmma works with cp.async but TMA+wgmma is the correct Hopper pair.

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

Key insight from Triton PTX reverse-engineering: `wgmma.wait_group 1` instead of 0.

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

### Remaining gap to Triton (~15%)

h2_s7 reaches 85% of Triton at N=4096–10240. Triton's additional techniques:

1. **BK=32 instead of BK=64**: smaller K-step = more K-tiles = finer-grained
   pipeline overlap. Triton's SMEM is 96KB vs our 144KB → fits 2 CTAs/SM in principle.
2. **More complex B swizzle**: Triton uses a warp-level XOR formula combining warp_id
   bits (not just row%8). May reduce bank conflicts further.
3. **2 cp.async commits per tile** (A and B separately): allows `wait_group(NS)` with
   finer granularity than our single-commit approach.
4. **Grouped CTA swizzle** for better L2 reuse across CTA block IDs.

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
