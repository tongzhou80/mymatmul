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

---

## Benchmark Results (H800, BF16, square M=K=N)

Measurement: `triton.testing.do_bench` (warmup 100ms, timed 500ms), best (min) time.
All TFLOPS = 2·M·N·K / time.

### Performance (TFLOPS)

| Size | tc5_regpruned | h1_ms | **h2_s3** | cuBLAS | h2_s3/cuBLAS |
|------|:---:|:---:|:---:|:---:|:---:|
| 1024 | 145 | **160** | 100 | 157 | 64% |
| 2048 | 326 | 331 | **334** | 570 | 59% |
| 3072 | 331 | **338** | 315 | 655 | 48% |
| 4096 | 378 | **382** | 361 | 673 | 54% |
| 5120 | 363 | **361** | 353 | 669 | 53% |
| 6144 | 374 | 381 | **405** | 669 | 61% |
| 7168 | 373 | 372 | **407** | 657 | 62% |
| 8192 | 362 | 368 | **415** | 695 | 60% |
| 9216 | 346 | 362 | **413** | 682 | 61% |
| 10240 | 375 | 365 | **399** | 667 | 60% |

Bold = best of our kernels at that size.

### Best Configs Selected by Autotuner

**h2_s3:**

| Size | Config |
|------|--------|
| 1024 | WG=2, BN=64, BK=64 |
| 2048–10240 | WG=2, BN=256, BK=64 |

**h1_ms** varies with size (BM=64–256, BN=64–128, BK=32–64, NS=2–4).

---

## Key Observations

### What each stage contributes

| Kernel | Key addition | 4096 TFLOPS | 8192 TFLOPS |
|--------|-------------|:-----------:|:-----------:|
| tc5_regpruned | Ada baseline on H800 | 378 | 362 |
| h1_ms | multi-stage cp.async pipeline | 382 | 368 |
| h2_s1 | TMA (no swizzle, mma.sync) | ~188 | ~203 |
| h2_s2 | TMA + wgmma, BM=64 | ~314 | ~290 |
| **h2_s3** | **+ 2 warpgroups (BM=128)** | **361** | **415** |
| cuBLAS | full Hopper optimization | 673 | 695 |

h2_s1 regresses vs tc5 because removing the B XOR swizzle (for correct linear SMEM)
causes ldmatrix bank conflicts that outweigh TMA's load-instruction savings.

### Wave quantization in h2_s3

h2_s3's best config is always `WG=2, BN=256` (128×256 output tile per CTA). This
creates few CTAs at medium sizes, causing poor SM utilization:

| Size | h2_s3 CTAs | Waves | Notes |
|------|-----------|-------|-------|
| 3072 | 288 | 2.2 | **worst** — last wave only 24 CTAs |
| 5120 | 800 | 6.1 | slight dip |
| 6144+ | ≥1152 | ≥8.7 | smooth, h2_s3 consistently wins |

h1_ms uses smaller tiles (128×128), giving 3–4× more CTAs and better quantization
across all sizes.

### Practical recommendation

- **N ≥ 6144**: use h2_s3 (TMA + wgmma wins by 8–13%)
- **N ≤ 5120**: use h1_ms (better wave utilization, lower wgmma overhead)
- A simple dispatcher switching at N=6144 covers the full range optimally.

### Why the remaining gap to cuBLAS (~40%) persists

Our h2_s3 hits ~55–62% of cuBLAS at large sizes. The remaining gap requires:
1. **Deeper pipeline** (3–4 TMA stages) to fully hide HBM3 latency
2. **Persistent kernels** (stream-K) for better load balancing
3. **Thread Block Clusters** for B-tile broadcast across CTAs

Each of these is a meaningful additional implementation effort, and cuBLAS almost
certainly uses all three simultaneously.

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
