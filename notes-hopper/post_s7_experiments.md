# Post-s7 experiments — why nothing beats s7 for this design

s7 reaches ~600 TFLOPS BF16 on H800 with BM=128, BN=256, BK=64, WG=2, NS=3.
After s7 we tried five distinct ideas to close the ~15% gap to Triton and to
cuBLAS. **None won meaningfully.** This file collects the findings so we don't
re-run the same experiments.

## Summary table (N=4096³, vs s7's ~597 TFLOPS)

| Variant | Idea | Result | Files |
|---|---|---|---|
| h4 | `__cluster_dims__(1,2,1)` only | ~1% (within noise) | `_matmul_h4.cu` |
| h4_s2 | Sweep cluster shapes | (1,1)/(1,2) ≈ s7; ≥(2,2) is −5–10% | `_matmul_h4_s2.cu` |
| h5 | Descriptor advance (no rebuild) | −1–2% (slightly worse) | `_matmul_h5.cu` |
| h6 | SMEM-staged 16-byte global stores | −10% (after fixing bank conflicts) | `_matmul_h6.cu` |
| h7 | Split cp.async into 2 commits (A+B) | +3% at BK=32, −4% at BK=64 | `_matmul_h7.cu` |
| h8 | Counter-based slot tracking (no `k%NS`) | −4% (slightly worse) | `_matmul_h8.cu` |
| h2c | 2-CTA cluster + DSMEM/TMA-multicast | −50% (h2c_cluster_experiment.md) | `_matmul_h2c.cu` |

### Same-config head-to-head with Triton (the real puzzle)

The most informative experiment was just running s7's autotune at BM=128, BN=256, WG=2 across all BK/NS combinations:

```
Config (BM=128 BN=256 WG=2)         TFLOPS   vs s7 best
BK=16 NS=5                            464    0.78x
BK=32 NS=4                            545    0.91x
BK=64 NS=3 (s7 autotune-best)         592    1.00x
BK=64 NS=4                            569    0.96x
```

So BK=64 is the right choice **for our pipeline structure**, beating BK=32 by 8%.

But **Triton uses BK=32, NS=4 and gets ~709 TFLOPS**. That means **at the same BK=32 config, Triton is 30% faster than s7**. The BK choice only explains ~8% — the rest is structural to how Triton emits the loop.

h7 (split commits, BK=32 NS=4) gets 555 TFLOPS — closes ~3% of that gap. The remaining ~22% at the same config is from things we can't pinpoint from PTX inspection: instruction scheduling, micro-pipelining, possibly the `// wait for regs:` hint Triton emits before `wgmma.wait_group`.

### Instruction-count audit and h8

We instrumented the count of integer instructions between `wgmma.fence` and the first `wgmma.mma_async`:

```
s7:     18 ops    (5 for `k mod 3` via mul-by-reciprocal + 13 for descriptors)
h8:     13 ops    (counter-based slot tracking; mod is gone)
Triton:  3 ops    (counter already updated above wait_group)
```

h8 replaces `cur = k % NS` (5 instructions of mul-by-reciprocal for non-power-of-2 NS=3) with a loop-carried counter that wraps via `(c+1==NS) ? 0 : c+1` (3 instructions). Successfully cut the critical-path int ops from 18 → 13.

**Result: h8 is 4% slower than s7 at the same config.**

This is the same lesson as h5: micro-optimizing instruction counts around wgmma doesn't translate to performance. The wgmma issue rate isn't actually the bottleneck. The kernel's wait stalls come from something deeper than instruction issue rate — hardware behavior at the tensor pipe / accumulator scoreboard level.

### Where the remaining gap actually lives

After h4, h4_s2, h5, h6, h7, h8, the structural rules we've derived:
- Almost any PTX-visible micro-optimization is a wash or slight regression
- Larger structural changes (clusters, SMEM-staged epilogue, descriptor schemes) are net-negative
- Triton's ~22% advantage at same config is in things the PTX surface does not show

The remaining lead would require either:
- **SASS-level analysis** of the actual machine code (nvdisasm the cubins, find where they differ)
- Or **warp specialization** (structural rewrite), which is the only optimization-class that has consistently won for matmul on Hopper

We have stopped pursuing PTX-level optimizations. s7 at ~600 TFLOPS is the design's ceiling.

## What we learned about the actual bottleneck

ncu profile of s7 best config (BM=128 BN=256 BK=64 WG=2 NS=3):

```
SM%             67.8%
DRAM%           21.7%      ← far from saturated
L1%             47.6%
L2%             58.3%      ← also not saturated
tensor pipe%    71.1%      ← 29% idle
wait stall%     23.6%      ← single biggest stall category
```

The kernel is **neither DRAM-bound nor L2-bound**. The per-tile AI of 85.3 F/B
(below the L2 ridge of 132 F/B) is misleading because:
1. Inter-CTA L2 reuse drops effective L2 traffic well below per-tile estimates
2. Global problem AI = N/3 ≈ 1365 F/B at N=4096, far above the DRAM ridge
3. wgmma SS reads from SMEM, not L2 directly — bandwidth at L2 is not on the
   wgmma critical path

The true bottleneck is the **wgmma micro-pipeline**: 71% tensor pipe utilization
means 29% of cycles the tensor core is idle. The 23.6% wait stall correlates
with this — warps are waiting on the wgmma engine to complete previous work.

This is *structural to single-warp-type-everywhere designs* like ours. Closing
it requires warp specialization (producer warp loads + consumer warps do
nothing but wgmma), which is a substantially different kernel design.

## What didn't work and why

### h4 — `__cluster_dims__(1,2,1)` is essentially free
Adding the cluster attribute with no DSMEM, no cluster.sync, no anything else
gave ~1% performance change. **Launching CTAs in clusters by itself has no
overhead.** This proves the slowdown in h2c came from the *sync + DSMEM
operations*, not from cluster scheduling.

### h4_s2 — Cluster shape doesn't matter (in the right direction)
Swept (1,1), (1,2), (2,1), (2,2), (1,4), (4,1), (2,4), (4,2):

```
size     (1,1)   (1,2)   (2,1)   (2,2)   (1,4)   (4,1)   (2,4)   (4,2)
4096     596.5   581.9   574.5   511.5   514.2   503.8   508.2   507.0
7168     600.8   594.2   570.3   534.2   528.9   513.2   541.6   525.6
8192     589.6   577.0   564.4   555.5   576.6   543.2   555.4   543.0
```

(1,1) and (1,2) tied at the top, anything bigger hurts ≥5%. Likely a side
effect of SM-placement constraints when clusters span GPCs unfavourably.

### h5 — Descriptor advance optimization is irrelevant
s7's `COMPUTE_TILE` rebuilds the wgmma descriptor on each kk step (8 int ops
between wgmma issues: `shr → and → cvt → or`). Triton's PTX uses `add.s64` to
advance the descriptor (2 int ops). We applied the same pattern.

```
PTX section between wgmma calls:
  s7: 12 lines (rebuild descriptor)
  h5: 6  lines (just add.s64 to advance)
```

End-to-end performance: **identical (within noise)**. ncu actually shows
slightly *worse* metrics for h5:

```
              s7       h5     delta
SM%           67.7%    65.7%  -2.0
tensor pipe%  71.1%    69.2%  -1.9
wait stall%   23.6%    27.6%  +4.0
```

The int ops between wgmma issues were **filler cycles** the warp scheduler
could use to hide other latencies. Removing them just exposed the underlying
wgmma serialization. The descriptor rebuild wasn't actually costing us
anything.

### h6 — SMEM-staged 16-byte stores
s7 emits 4-byte global stores direct from wgmma accumulator. h6 stages
through SMEM, then emits 16-byte coalesced `st.global.v4.b32` (mirroring
Triton's epilogue). The idea: 4× fewer global store transactions.

**Result with naive `[BM][BN]` SMEM layout**: −12% (slower than s7). 1.8M
store-side bank conflicts (`ST-cf=1835008`) because BN=256 is a power of 2,
making every row of `C_sh` hit the same banks → 4-way conflict.

**After padding to `[BM][BN+8]` to break the stride**: ST-cf=0, but still −10%.

Why padding didn't fix it:
- h6 writes one `bf16x2` (4 bytes) to SMEM at a time → low SMEM-store
  bandwidth utilization on the *staging* side
- Triton uses 16-byte SMEM stores by carefully packing 4 non-adjacent
  `bf16x2` values per `st.shared.v4.b32` (looks like `{%r229, %r231, %r233,
  %r235}` in their PTX, picking specific scattered registers from the wgmma
  output layout)
- Without the careful packing, the SMEM-side cost exceeds the global-side
  saving

Replicating Triton's full pattern requires reasoning about the wgmma output
layout (which thread holds which 4 fp32 values for which (row, col) tuple)
and assembling `v4.b32` stores from those. That's substantial work — same
scope as warp specialization but with less clear payoff (epilogue is ~5–10%
of kernel time at best).

Also: the H800 LSU has **store-combining logic** that recovers a lot of the
"scattered 4-byte" cost in s7. The naive cost model ("4-byte stores are 4×
slower than 16-byte stores") overestimates the upside.

### h2c — 2-CTA cluster + DSMEM / TMA-multicast (see h2c_cluster_experiment.md)
The headline experiment of this series. Both DSMEM-copy and TMA-multicast
designs ran at ~50% of s7's speed, because each iter needs a `cluster.sync`
that drifts as much as ~3000+ cycles per iter due to inter-SM scheduling
jitter. The 770-cycle cluster.sync microbenchmark cost was the floor, not the
realised cost in a memory-bound steady state.

## Pattern across all these experiments

Every "advanced" optimization we tried either:
- Did nothing (h4, h5)
- Made things slightly worse (h5 stall metrics, h6 perf)
- Made things much worse (h2c, h4_s2 large shapes)

The s7 design is already **structurally at its ceiling**. The remaining
~15% gap to Triton/cuBLAS lives in design choices we can't bolt on:
- Warp specialization (producer + consumer split)
- Persistent grid + smart tile scheduling
- TMA with multi-stage prefetch tied to warpgroup ownership

These are CUTLASS-class rewrites. None of the "swap one thing" optimizations
we tried can deliver more than a few percent on this baseline.

## What is worth keeping

- `_matmul_h4.cu` / `matmul_h4.py` — useful as the "cluster skeleton" baseline
  for any future cluster-based experiment
- `bench_cluster_sync.py` / `_cluster_sync_bench.cu` — useful microbenchmark
- `_matmul_h2_s7_one.cu` / `_matmul_h5_one.cu` — single-launcher PTX
  inspection helpers
- `bench_cluster_shapes.py` — the cluster-shape sweep harness

h5 and h6 are kept as dead-end references but are not used by default.
