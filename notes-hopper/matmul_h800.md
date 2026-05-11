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

## Key New Hopper Instructions

### wgmma (Warp-Group MMA)
`wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16`

- Operates on an entire **warpgroup** (4 warps = 128 threads) as a unit.
- M is fixed at 64; N ∈ {8, 16, …, 256} in multiples of 8; K = 16.
- A operand comes from **registers** (loaded via `ldmatrix` across the warpgroup).
- B operand comes directly from **SMEM** (hardware reads it; threads don't touch it).
- Result in F32 registers distributed across the 128 threads of the warpgroup.
- Requires `wgmma.fence.sync.aligned` before first use and
  `wgmma.commit_group.sync.aligned` / `wgmma.wait_group.sync.aligned` to drain.

vs `mma.sync` on Ada (sm_89):
- `mma.sync` is per-warp (32 threads), tile = m16n8k16.
- `wgmma` tile = m64n256k16 — **32× more FLOPs per instruction** at full N.

### TMA (Tensor Memory Accelerator)
`cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx_bytes`

- Hardware-managed DMA; one thread issues the descriptor, all benefit.
- Supports multidimensional layouts, automatic swizzle/format conversion.
- Works with `mbarrier` (async transaction barrier) instead of `__pipeline_*`.
- Dramatically reduces instruction count for tile loads vs per-thread `cp.async`.

### Thread Block Clusters
- Groups of up to 8 CTAs that share an L1/SMEM region (Distributed Shared Memory).
- Allows B tiles to be broadcast from one CTA to all cluster members.
- Enables persistent-kernel patterns at scale.

---

## Implementation Plan

### Stage H1 — tc5_lb Baseline on H800
Port the tc5_lb kernel (best 4090 BF16 kernel) to H800.
- The kernel uses `mma.sync.m16n8k16` and `cp.async` — both valid on sm_90a.
- Pre-compiled sm_90 cubins already exist in `tensor_core/`.
- Goal: establish a baseline and verify the autotuner + benchmark harness work on H800.
- Expected: ~150–200 TFLOPS at N=4096 (limited by `mma.sync` throughput).

### Stage H2 — wgmma Kernel (no TMA)
Implement `wgmma.mma_async` with existing `cp.async` data movement.

Design:
- Thread block = 1 warpgroup = 128 threads (4 warps).
- CTA tile: BM = 64 (one warpgroup row), BN ∈ {128, 192, 256}, BK = 16.
- A tile: loaded by threads via `ldmatrix.x4` into registers, then used by wgmma.
- B tile: loaded into SMEM by `cp.async`; wgmma reads directly from SMEM.
- Pipeline: 2-stage double buffer (same `__pipeline_*` as tc5).
- Accumulator: F32, stored as `float acc[WN_TILES * 2]` per thread (2 floats per
  m16n8 wgmma output per thread in a warpgroup).

Key challenge: wgmma accumulator layout across 128 threads is different from
`mma.sync` layout across 32 threads. The epilogue (write C) needs matching.

### Stage H3 — TMA + wgmma (Persistent Kernel)
Add TMA descriptors for A and B tiles.

- Eliminates per-thread address computation; one thread issues `cp.async.bulk.tensor`.
- `mbarrier` replaces `__pipeline_memcpy_async` / `pipeline_wait_prior`.
- Enables deeper pipeline (3–4 stages with 228 KB SMEM).
- Persistent kernel: grid stays resident; blocks loop over multiple output tiles.

### Stage H4 — Thread Block Clusters
- Cluster size 2 or 4 along N dimension.
- B tiles shared via Distributed Shared Memory — each CTA loads one BN/cluster
  slice, making the effective BN tile per SM = cluster_size × BN.
- Reduces global-memory B traffic by cluster_size×.

---

## Notes

- Compute capability for wgmma: requires sm_90a (not sm_90). Compile with `-arch=sm_90a`.
- TMA requires CUDA 12.0+. Current install: CUDA 12.8 (driver 580.82.07). ✓
- The `__pipeline_*` API (used by tc5) still works on sm_90a for stages H1/H2.
- For H2+, use `cuda/pipeline.h` or raw PTX `wgmma.*` / `mbarrier.*` intrinsics.
- Useful reference: CUTLASS 3.x `cute` + Hopper kernels (sm90_gemm_*.hpp).
