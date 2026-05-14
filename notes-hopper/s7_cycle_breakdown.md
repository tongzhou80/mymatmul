# h2_s7 per-phase cycle breakdown

Instrumented the main K-loop of `h2_s7` (best config: BM=128, BN=256, BK=64,
WG=2, NS=3) with `clock64()` reads between each phase, accumulated cycles
across all main-loop iterations, and reported per-iter averages over all CTAs.

Measured at N=4096³ on H800. Instrumented version: `_matmul_h2_s7_timed.cu`.
Analysis script: `benchmarks/profile_s7_phases.py`.

## Per-iter cycle breakdown (mean over 512 CTAs, 62 main-loop iters/CTA)

```
Phase                cycles/iter   share
─────────────────────────────────────────
WAIT_SMEM                  19.1    1.5%   ← cp.async drain
sync_pre_cmp                2.3    0.2%   ← __syncthreads
COMPUTE_TILE              282.9   23%     ← wgmma issues + scoreboard stalls
WAIT_MMA(1)                 5.5    0.4%   ← wgmma drain
sync_pre_load               2.0    0.2%   ← __syncthreads
LOAD_TILE                 924.4   75%     ← cp.async issues + queue backpressure
─────────────────────────────────────────
TOTAL (instrumented)     1236.3
```

Actual per-CTA per-iter time (from total kernel time / iters per CTA, accounting
for ~4 waves over 132 SMs): **~1800 cycles**. The ~600 cycle gap vs the
instrumented total is clock64 overhead + multi-CTA L2 contention + wave effects.

## What this proves and disproves

### Asynchronous ops are completely hidden
- `WAIT_SMEM` is **19 cycles** — cp.async is *always* done by the time we reach
  the wait. The cp.async pipeline is fully drained behind the warp's other work.
- `WAIT_MMA(1)` is **5.5 cycles** — wgmma drain is also free. The previous group's
  wgmma has long since written back by the time we wait for it.
- **No need to deepen the cp.async pipeline (NS) or change wait_group depths.**

### Barriers are essentially free
- Both `__syncthreads` cost ~2 cycles. Warpgroups are well-aligned; no
  meaningful sync cost between them.

### The warp's time is dominated by ISSUE, not data movement
The warp spends 75% of each iter just *issuing* `cp.async` instructions in
LOAD_TILE. The actual data transfers happen asynchronously in the background
and are fully complete by the time the warp checks. The bottleneck is the
warp-issue-side throughput, not memory bandwidth.

## Decomposing COMPUTE_TILE (283 cycles)

Expected breakdown:
- Initial descriptor compute (k%NS + B desc + A desc): ~18 ops × 1 cyc = 18 cycles
- `wgmma.fence` + setup: ~10 cycles
- 4× `wgmma.mma_async` issue at peak rate (~32 cyc each): 128 cycles
- Per-wgmma descriptor advance between issues (~8 int ops between each): 24 cycles
- `wgmma.commit_group`: ~5 cycles
- Counter wraparound for next iter's `cur`: ~5 cycles
- **Predicted: ~190 cycles. Measured: 283. Gap: ~90 cycles.**

The ~90 missing cycles are **per-wgmma accumulator-scoreboard stalls**. Each
wgmma N+1 reads `acc[]` (output of wgmma N). The hardware scoreboard enforces
serialized accumulator writes, so wgmma N+1's issue may briefly stall until
wgmma N's writeback is acknowledged. Over 4 wgmma, this accumulates to ~90 cycles.

## Decomposing LOAD_TILE (924 cycles)

Expected breakdown:
- Address compute for 12 loads (~6-8 int ops each, with some reuse): ~80 cycles
- 12× `cp.async.cg.shared.global` issue at peak (~16 cyc each): 192 cycles
- `cp.async.commit_group`: ~5 cycles
- **Predicted: ~280 cycles. Measured: 924. Gap: ~640 cycles.**

The ~640 missing cycles are **cp.async issue-side backpressure**. The DMA
engine has a finite queue per SM/warp. Issuing 12 cp.async in quick succession
without intermediate commits saturates the queue; subsequent issues stall until
the queue drains enough to accept new entries.

This is *not* the same as cp.async completion latency — it's a throughput
limit on the warp's ability to push instructions into the DMA queue.

## Implications for the tensor pipe utilization gap

ncu reports tensor-pipe utilization at **71%** (29% idle). This breakdown explains why:

- Tensor pipe is fed by 4 wgmma per COMPUTE_TILE per warpgroup
- COMPUTE_TILE only takes 283 cycles of warp time
- After COMPUTE_TILE, the warp moves on to LOAD_TILE (924 cycles)
- During LOAD_TILE, the tensor pipe drains all 4 wgmma in well under 924 cycles
- → tensor pipe sits idle for the remainder of LOAD_TILE before the next COMPUTE_TILE

To raise tensor pipe utilization, we want **more wgmma per COMPUTE_TILE** so
the wgmma queue stays non-empty for longer. Two concrete levers:

1. **M_ITERS = 2** (BM=256, WG=2): each kk step issues 2 wgmma instead of 1.
   COMPUTE_TILE has 8 wgmma per warpgroup. Doubles the queue depth.

2. **Smaller BK + more wgmma** (already tried via autotune): keeps per-iter
   compute work small but adds more iters → not a net win for our kernel.

## What doesn't help (verified)

- **Wider cp.async per thread**: already at max (16B per cp.async issue).
- **TMA bulk transfer** (`cp.async.bulk.tensor`): replaces 12 cp.async with
  ~5 TMA issues, but mbarrier infrastructure (init + arm + wait) eats the
  saving. Net result: 75% of s7's speed (see `h2_s7_tma`).
- **Deeper NS (more cp.async stages)**: cp.async is already not blocking,
  so adding stages just wastes SMEM and registers without helping.
- **Counter-based slot tracking** (h2_s7_counter): cleaner PTX but no perf
  gain — the int ops saved were not on the critical path.
- **Descriptor advance** (h2_s7_desc): same story as counter.

## Key takeaway

The bottleneck for s7 is *warp instruction issue rate*, not memory or compute
async-engine throughput. The way to push past 600 TFLOPS is to keep the
**warp's issue pipeline biased toward wgmma rather than cp.async** — and that
means more wgmma per iter (M_ITERS), not micro-tuning around the existing
issue mix.
