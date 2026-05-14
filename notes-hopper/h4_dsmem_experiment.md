# h4_dsmem — Cluster (2-CTA) experiment: didn't pay off

**Status:** dead-end. Kept in tree as a functional reference, but slower than s7.

## Motivation (the AI argument)

s7 per-CTA arithmetic intensity for BM=128, BN=256:

```
AI = BM × BN / (BM + BN) = 128 × 256 / 384 = 85.3 FLOP/byte
```

At measured H800 L2 bandwidth (~7.5 TB/s literature, ~3 TB/s for write+RBW workloads), this puts a ceiling of 85.3 × 7.5 ≈ 640 TFLOPS. We observe s7 at ~600 TFLOPS — confirming the kernel is L2-bandwidth bound, not compute bound.

To raise AI, share B across CTAs:
- 2-CTA cluster along M direction → each B byte loaded from DRAM once, used by both CTAs' wgmma
- Effective AI: 2 × BM × BN / (2 × BM + BN) = 65536 / 512 = **128 FLOP/byte** (right at L2 ridge)
- Theoretical ceiling: 128 × 7.5 ≈ 960 TFLOPS

## What we tried

Two cluster designs, both based on s7's pipeline:

### Design A: K-split + DSMEM copy
- Each CTA loads its own A strip + half of B's K-rows from DRAM (cp.async)
- After loading, each CTA reads peer's B half via `ld.shared::cluster.v4.b32` and stores into its own full-B SMEM
- wgmma reads local SMEM (identical to s7 from wgmma's perspective)

### Design B: TMA multicast for B
- A still loaded per-CTA via cp.async
- B loaded via `cp.async.bulk.tensor.*.multicast::cluster` — TMA fetches B once from DRAM, hardware delivers to both CTAs' local SMEM
- mbarrier per slot tracks B's TMA delivery
- wgmma reads local SMEM (identical to s7)

Pipeline was reorganized to overlap the cluster barrier with wgmma's async execution window:

```
WAIT_SMEM(NS-2, cur)
fence_proxy_async; __syncthreads
COMPUTE_TILE(cur)          # async wgmma starts
LOAD_ARM(nxt)              # per-CTA mbarrier init+arm (no sync)
cluster.sync()             # overlaps with wgmma in flight
WAIT_MMA(1); __syncthreads
LOAD_ISSUE(nxt, ...)       # cp.async A + (rank 0) TMA-multicast B
```

## What we measured (BF16, N=4096, BM=128, BN=256, BK=64, WG=2, NS=3)

| Variant | TFLOPS | ms | speedup vs s7 |
|---------|--------|----|----|
| h2_s7 (baseline) | 604 | 0.227 | 1.00× |
| h4_dsmem — Design A (DSMEM copy) | 305 | 0.451 | 0.50× |
| h4_dsmem — Design B (TMA multicast, overlap pipeline) | 318 | 0.432 | 0.53× |

Both cluster designs are roughly half the speed of s7. Correctness checks pass at all sizes.

## Why it didn't work

Both designs need a **cross-CTA barrier every iteration** (`cluster.sync()` or its equivalent). That barrier dominates the iter time:

- s7 per-iter cost: ~7100 cycles (measured)
- h4_dsmem per-iter cost: ~13500 cycles
- Extra cost: ~6400 cycles/iter

The cluster.sync on Hopper appears to be on the order of 1000+ cycles, not the 50–100 a back-of-envelope suggests. Multiplied by ~64 K-iters, it overwhelms the AI gain.

The overlap trick (issue wgmma async, then cluster.sync during its in-flight window) didn't help because wgmma execution time is ~the same order as cluster.sync, leaving little to hide behind.

## Why CUTLASS gets away with clusters

CUTLASS Hopper cluster kernels use **warp specialization**:
- Producer warpgroup handles all loads + mbarrier signaling
- Consumer warpgroups run wgmma only
- Cross-CTA sync happens on the producer warp; consumers never stall

Our s7-style design has all warps as consumers, so the cluster barrier becomes a global pipeline stall.

## Decision

- s7 at ~600 TFLOPS is already close to the L2-bandwidth ceiling for its 85.3 F/B intensity.
- To raise the ceiling further, we'd need warp specialization or TMA + persistent kernels — a substantially larger rewrite.
- The complicated optimization that only conditionally helps did not help here. Pattern matches earlier "advanced feature" experiments: the only optimizations that consistently pay off are the simple, must-win ones.

Files kept in tree: `mymatmul/gpu/hopper/_matmul_h4_dsmem.cu`, `matmul_h2c.py`. Will not be used by default.
