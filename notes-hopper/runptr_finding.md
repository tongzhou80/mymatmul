# Running pointers — +9% over s7_split, closes 40% of the Triton gap

## TL;DR

`h2_s7_runptr` replaces s7_split's recompute-from-base address pattern with
per-thread running pointers (Triton's pattern). Result: **+50 TF average
(+8.8%) sustained, closes ~40% of the remaining Triton gap**.

## Background — how the finding came from decompilation

After exhausting structural variants (h4 clusters, h5 SMEM-staged epi, h6,
h7 split-commits, h8 counter, h9 wait-hints, h2_s7_tma, h2_s7_predesc), the
remaining 15-20% gap to Triton's pre-compiled PTX at the SAME (BK, NS, NW)
config seemed to live in "things the PTX surface does not show" — i.e.
SASS-level scheduling.

To make Triton's specific instruction choices legible we built a 36-stage
decompilation pipeline (`mymatmul/gpu/hopper/make_t*.py` → `_matmul_t*.cu`).
By t36 the kernel reads as algorithmic C++ with named `__device__
__forceinline__` helpers wrapping each PTX op, while staying bit-equal to
`triton_ptx` and within 2-3 TF cold-burst.

Side-by-side comparing t36 against `_matmul_h2_s7_split.cu` revealed one
concrete difference in the LOAD_TILE address pattern:

  - **Triton**: maintains per-thread running pointers, advanced by stride
    once per K-iter (3 `add.s64` total per iter).
  - **s7_split**: recomputes the global address from base in each cp.async:
        `&A[(block_row + _r) * K + (k0_) + _c]`
    ≈ 3 ops/cp.async × 6 cp.async = ≈ 18 integer ops/iter.

Per `s7_cycle_breakdown.md`, the kernel is warp-instruction-issue-bound:
≈15 extra integer ops/iter directly delays the next wgmma issue.

## Implementation

`_matmul_h2_s7_runptr.cu` adds per-thread arrays:

```cpp
const __nv_bfloat16* A_curr[A_GROUPS];
const __nv_bfloat16* B_curr[B_GROUPS];
```

initialized once at k0 = 0 in the preheader, then advanced inside
`LOAD_TILE`:

```cpp
#define LOAD_TILE(slot_)                                                    \
    do {                                                                    \
        _Pragma("unroll")                                                   \
        for (int _i = 0; _i < A_GROUPS; _i++) {                             \
            ...                                                             \
            __pipeline_memcpy_async(&A_sh[slot_][_r][_sc],                  \
                                    A_curr[_i],                             \
                                    A_ELEM * sizeof(__nv_bfloat16));        \
            A_curr[_i] += BK;                                               \
        }                                                                   \
        __pipeline_commit();                                                \
        _Pragma("unroll")                                                   \
        for (int _i = 0; _i < B_GROUPS; _i++) {                             \
            ...                                                             \
            __pipeline_memcpy_async(&B_sh[slot_][_st][_kr][_sc],            \
                                    B_curr[_i],                             \
                                    B_ELEM * sizeof(__nv_bfloat16));        \
            B_curr[_i] += BK * N;                                           \
        }                                                                   \
        __pipeline_commit();                                                \
    } while (0)
```

The `k0_` parameter is removed — running pointers carry the K position.

## Perf

`bench_gpu2.py` sustained measurement on H800, BM=128 BN=256 BK=64 WG=2 NS=3:

| Size | h2_s7 | h2_s7_split | **h2_s7_runptr** | triton_ptx | Δ vs split | runptr / triton |
|-----:|------:|------------:|-----------------:|-----------:|-----------:|----------------:|
| 4096 | 569.2 | 564.7 | **615.2** | 666.1 | **+50.5 (+8.9%)** | 92.4% |
| 5120 | 531.9 | 528.1 | **578.2** | 614.3 | **+50.1 (+9.5%)** | 94.1% |
| 6144 | 567.1 | 555.8 | **624.7** | 671.3 | **+68.9 (+12.4%)** | 93.0% |
| 7168 | 566.9 | 591.7 | **618.9** | 689.4 | +27.3 (+4.6%) | 89.8% |
| 8192 | 557.1 | 581.5 | **631.7** | 695.1 | +50.2 (+8.6%) | 90.9% |

Cold-burst at 7168 (warmup=5ms, rep=20ms, 2s cooldown):

| | s7_split | s7_runptr | triton |
|---|---:|---:|---:|
| BK=64 NS=3 | 637 | **669** | 759 |
| BK=32 NS=4 | 621 | 627 | 759 |

Bit-equal output to s7_split and triton_ptx across 1024, 2048, 4096, 7168.

## What's left

After running pointers, s7_runptr sits at ~91% of Triton's sustained perf.
The remaining ~9% is likely:

- B-side pointer encoding: Triton uses `B_gmem_base + B_offset_i` (1 advance +
  4 constant offsets) where we have 4 advancing pointers. Same total
  `add.s64` count per iter but different register-pressure tradeoff.
- SASS-level instruction scheduling that requires disassembly comparison.

Or it requires structural rewrites (warp specialization, TMA, DSMEM) that
are large enough to warrant their own design phase.
