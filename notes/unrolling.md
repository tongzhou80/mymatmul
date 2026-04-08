# Why loop unrolling helps, and why the speedup exceeds 2x

## Setup

Hardware: AMD EPYC 7773X (Zen 3)
- 2 FMA units per core
- FMA throughput: 2 per cycle (one per unit) — superscalar: 2 FMA units dispatch simultaneously
- FMA latency: 4 cycles (result not ready until 4 cycles later)
- Peak FP32: 8 (AVX2) × 2 (FMA ops) × 2 (FMA units) × 3.5277 GHz = ~113 GFLOPS/core
- AVX2: 8 floats per vector (256-bit)

Implementation: `cpp_ikj_unroll` — SAXPY inner loop:
```cpp
for (int k ...) {
    float a_ik = A[i,k];
    for (int j ...) {
        C[i,j] += a_ik * B[k,j];   // FMA
    }
}
```

---

## 1. Why unrolling helps at all

Without unrolling, the inner loop has a single accumulator `C[i,j]`.
Each FMA reads the previous result of that same accumulator — a **loop-carried dependency**:

```
cycle 1:  FMA issued  (C[j] = C[j] + a * B[j])
cycle 2:  waiting for result...
cycle 3:  waiting...
cycle 4:  waiting...
cycle 5:  result ready → next FMA can issue
```

FMA latency is 4 cycles, so we issue **1 FMA every 4 cycles** instead of 1 every 0.5 cycles.
We're running at 1/8 of peak throughput.

With 4 independent accumulators (unroll by 4), the chains don't depend on each other:

```
cycle 1:  FMA for C[j+0]   FMA for C[j+8]   ← both issued
cycle 2:  FMA for C[j+16]  FMA for C[j+24]
cycle 3:  FMA for C[j+0]   (next iteration, result now ready after 4 cycles... almost)
...
```

The CPU issues new FMAs into the idle slots while waiting for earlier results.
**Latency is hidden** — the pipeline stays full.

---

## 2. Why the speedup is much more than 2x

It is tempting to think: "there are 2 FMA units, so the max speedup from unrolling is 2x."
That reasoning would be correct **if the baseline were already compute-bound**.

But the baseline (single accumulator, no unrolling) is **latency-bound**, not compute-bound.
The actual bottleneck breakdown:

| Situation | FMAs per cycle | Fraction of peak |
|---|---|---|
| Single accumulator (baseline) | 1/4 | 1/8 |
| 4 accumulators (latency hidden, 1 FMA unit) | ~1 | 1/2 |
| 8 accumulators (both FMA units saturated) | ~2 | 1 (peak) |

Going from 1 to 4 chains: **4x gain** (latency hiding).
Going from 4 to 8 chains: **2x gain** (second FMA unit).

In practice we observed ~3.3 → ~31 GFLOPS, roughly **9x**, which is close to the
combined 4 * 2 = 8x theoretical. The extra gain comes from removing the loop overhead
and index recomputation that the compiler could not optimize in the single-chain version.

---

## 3. Why 4 chains and 8 chains give the same result here

Once latency is hidden with 4 chains, the next limit is memory bandwidth.
At large matrix sizes the working set does not fit in L1/L2, so the cache cannot
supply data fast enough to keep both FMA units busy regardless of how many
accumulator chains we have.

**4 chains = latency hidden, bandwidth bound.**
**8 chains = latency hidden, still bandwidth bound. No further gain.**

To extract the second 2x from the second FMA unit we would need to also solve the
bandwidth problem — i.e., tiling so the working set fits in L1 and data is reused
before being evicted.

---

## Summary

```
baseline (1 chain):   latency-bound    →  1/8 of peak
4 chains unrolled:    bandwidth-bound  →  ~1/2 of single-core peak
tiled + unrolled:     compute-bound    →  closer to full single-core peak
```

The lesson: **unrolling fixes latency stalls; tiling fixes bandwidth pressure.
You need both to approach the hardware peak.**
