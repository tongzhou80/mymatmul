# H2-S6 Design: cp.async + wgmma SS Mode

## Motivation

From profiling Triton's BF16 kernel (BM=128, BN=256, BK=32, NS=4, NW=8):

| Metric | Our h2_s5 | Triton |
|--------|-----------|--------|
| SM SoL | 50.9% | **91.8%** |
| Registers/thread | 186 | **168** |
| Dynamic SMEM | 192 KB | **96 KB** |
| membar stalls | 28.4% | **0%** |
| LD bank conflicts | 0 | 0 |

Triton achieves near-2× our SM utilisation using **cp.async** (not TMA) with
**wgmma SS mode** (both A and B from SMEM via descriptors, no ldmatrix for A).

Key insight: TMA doesn't reduce DRAM latency vs cp.async — it only reduces
instruction overhead. A well-tuned cp.async pipeline hides the same latency with
simpler synchronisation and smaller SMEM, enabling more CTAs/SM.

---

## What Changes from h2_s5 / h1_ms

| Component | h2_s5 | h1_ms | **h2_s6 (new)** |
|-----------|-------|-------|-----------------|
| Data movement | TMA + mbarrier | cp.async + pipeline | **cp.async + pipeline** |
| A compute path | ldmatrix → registers → wgmma RS | ldmatrix → regs → mma.sync | **SMEM → wgmma SS (no ldmatrix)** |
| B compute path | SMEM → wgmma SS | ldmatrix → regs → mma.sync | **SMEM → wgmma SS** |
| A swizzle (cp.async write) | BK=64 only | BK=64 only | **BK/8 period (all BK)** |
| B swizzle (cp.async write) | row%8 sub-tile | row%(BN/8) | **row%8 sub-tile** |
| SMEM per stage | NS×(BM×BK + BK×BN)×2 | same | **same formula** |
| Registers | ~186 (128 acc + ldmatrix frags) | ~240 | **~168 (128 acc, no frags)** |

The A ldmatrix bank-conflict limitation no longer applies in SS mode, so
**BK=16/32 can now use their matching swizzle** (B32/B64), not just BK=64.

---

## SMEM Layout and Swizzle

### A tile: K-major layout

A is stored as `[BM][BK]` (M rows, K cols), K is the inner/contiguous dimension.

**cp.async write XOR pattern:**
```
A_SWZ   = BK / 8         (number of distinct XOR values)
A_SHIFT = 64 / BK        (rows per XOR group)

physical_col_group = logical_col_group XOR ((row / A_SHIFT) % A_SWZ)
physical_col_8bf16 = physical_col_group * 8 + logical_col_8bf16 % 8

BK=16 → A_SWZ=2, A_SHIFT=4 → B32 swizzle (rows group in 4s: 0-3 same XOR)
BK=32 → A_SWZ=4, A_SHIFT=2 → B64 swizzle (rows group in 2s: 0-1 same XOR)
BK=64 → A_SWZ=8, A_SHIFT=1 → B128 swizzle (each row unique XOR)
```

This is the **same XOR formula as h1_ms** — verified from Triton PTX (BK=32):
rows 0,1 use XOR=0; rows 2,3 use XOR=1; rows 4,5 use XOR=2; etc.
(The design note initially said `row % (BK/8)` which was wrong — rows pair up.)

**A GmmaDescriptor fields** (verified from Triton PTX for BK=32):
```
layout_type  : BK=16→B32(3), BK=32→B64(2), BK=64→B128(1)
LBO          : 0   (no leading-dim jump needed for K-major)
SBO          : BK  (= 8 rows × BK BF16 × 2 bytes / 16)

SBO meaning: jump between consecutive 8-M-row blocks in SMEM.
             For BK=32: SBO=32 → actual stride = 512 bytes = 8×32×2 ✓
```

**Per-kk advancement:**
```
kk_advance = kk * 2  (in start_address_ units = ×16 bytes)
           = kk * 32 bytes = kk * 16-K-elements × 2 bytes/elem
```

Construction:
```cpp
template<int BK>
uint64_t make_wgmma_a_desc(uint32_t smem_addr, int kk) {
    constexpr int layout = (BK==64) ? 1 : (BK==32) ? 2 : 3;  // B128/B64/B32
    constexpr uint64_t LAYOUT_BITS = (uint64_t)layout << 62;
    constexpr uint64_t sbo = BK;
    uint64_t start = ((uint64_t)(smem_addr >> 4) & 0x3FFF) + kk * 2;
    return start | (sbo << 32) | LAYOUT_BITS;   // LBO=0
}
```

---

### B tile: MN-major sub-tile layout (identical to h2_s5/h3)

B is stored as `BN/64` packed sub-tiles of `[BK][64]`, back-to-back in SMEM.

```
Why sub-tiles?
  B's full row is BN BF16 = BN×2 bytes.
  128B swizzle requires boxDim[0] ≤ 64 BF16 (= 128 bytes).
  So B is split into BN/64 sub-tiles of 64 columns each.

  For BN=256: 4 sub-tiles of [BK][64], total BK×256×2 bytes.
  Sub-tile 0: B[0..BK-1][0..63]    at SMEM offset 0
  Sub-tile 1: B[0..BK-1][64..127]  at SMEM offset BK×128
  Sub-tile 2: B[0..BK-1][128..191] at SMEM offset BK×256
  Sub-tile 3: B[0..BK-1][192..255] at SMEM offset BK×384
```

**cp.async write XOR pattern:**
```
For each sub-tile st (0..BN/64-1):
  physical_col_8bf16 = logical_col_8bf16 XOR (row % 8)   ← B128, always period=8
```

**B GmmaDescriptor fields** (same as h2_s5, verified from Triton PTX):
```
layout_type  : B128 (1)
LBO          : 8×BK  (for BN>64) → jump between 64-col sub-tiles
SBO          : 64    (always)    → jump between 8-K-row blocks within a sub-tile

LBO meaning: jump from sub-tile 0 to sub-tile 1.
             LBO=8×BK → actual = 8×BK×16 = BK×128 bytes = BK×64×2 ✓

SBO meaning: jump between 8-K-row blocks inside one sub-tile.
             SBO=64 → actual = 1024 bytes = 8 rows × 64 BF16 × 2 bytes ✓
```

**Per-kk advancement:**
```
kk_advance = kk * 128  (in start_address_ units = ×16 bytes)
           = kk * 2048 bytes = K_STEP_BYTES

Why 2048 and not BN×16×2 (=8192 for BN=256)?
  B is NOT stored row-major [BK][BN]. It's sub-tile format.
  Within sub-tile 0 ([BK][64]), K-row 16 is 16×64×2=2048 bytes away.
  The hardware uses LBO to simultaneously access all sub-tiles for the full BN.
```

Construction (unchanged from h2_s5):
```cpp
template<int BN, int BK>
uint64_t make_wgmma_b_desc(uint32_t smem_addr) {
    // Same formula as h2_s5
    constexpr int n_atoms = BN / 64;
    constexpr uint64_t lbo = (n_atoms<=1) ? 0ULL : (uint64_t)(8*BK);
    constexpr uint64_t sbo = 64;
    uint64_t start = ((uint64_t)(smem_addr>>4)) & 0x3FFF;
    return start | (lbo<<16) | (sbo<<32) | (1ULL<<62);  // B128
}
// Per-kk: B_base + kk * 2048
```

---

## wgmma SS vs RS PTX

**RS mode (h2_s5 — A from registers):**
```ptx
wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16
    {d0..d127},        ← accumulator (in/out)
    {a0,a1,a2,a3},     ← A fragment from ldmatrix (registers)
    desc_B,            ← B descriptor
    1, 1, 1, 1;        ← scaleD, scaleA, scaleB, transB
```

**SS mode (h2_s6 — both A and B from SMEM):**
```ptx
wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16
    {d0..d127},        ← accumulator (in/out)
    desc_A,            ← A descriptor (no registers!)
    desc_B,            ← B descriptor
    1, 1, 1, 0, 1;     ← scaleD, scaleA, scaleB, transA, transB
```

Note: SS mode has an extra `transA=0` parameter (K-major = not transposed).
Verified from Triton PTX: `..., %p5, 1, 1, 0, 1`

---

## SBO and LBO: the hardware's navigation map

The wgmma hardware reads tiles in **8-row × 64-column atoms** (the smallest unit
it processes at a time). SBO and LBO tell it how to hop between atoms.

```
For A [64 M-rows][BK K-cols]:          For B sub-tile [BK K-rows][64 N-cols]:

M-rows  0.. 7  ←── block 0 ─┐          K-rows  0.. 7  ←── block 0 ─┐
M-rows  8..15  ←── block 1  │ SBO      K-rows  8..15  ←── block 1  │ SBO
M-rows 16..23  ←── block 2  │          K-rows 16..23  ←── block 2  │
M-rows 24..31  ←── block 3  │          K-rows 24..31  ←── block 3  │
...                          │          ...                          │
M-rows 56..63  ←── block 7 ─┘          K-rows 56..63  ←── block 7 ─┘

SBO (A) = 8 × BK × 2 / 16 = BK       SBO (B) = 8 × 64 × 2 / 16 = 64

LBO (A) = 0  (K is inner, no jump)    LBO (B) = BK × 64 × 2 / 16 = 8×BK
                                                  ↑
                                                  jump from sub-tile 0 to sub-tile 1
```

---

## Pipeline Structure

Identical to h1_ms (cp.async + `__pipeline_wait_prior`), adapted from h2_s5's
multi-stage structure with M_ITERS:

```
Template: <BM, BN, BK, NUM_WG, NUM_STAGES>
  M_ITERS = BM / (NUM_WG * 64)
  NS = NUM_STAGES ∈ {2, 3, 4, 5}

SMEM: A[NS][BM][BK]  (K-major, A_SWZ_PERIOD=BK/8 XOR)
      B[NS][BK][BN]  (sub-tile format, row%8 XOR)

Prologue : issue NS-1 tiles
Main loop: for each K-tile k:
             issue next tile
             __pipeline_wait_prior(NS-1)   ← replaces doorbell_wait
             __syncthreads()
             MULTIPLY_TILE(cur)
             __syncthreads()
Drain    : last NS-1 tiles (unrolled)
```

MULTIPLY_TILE (SS mode):
```
wgmma_begin()           ← fence.proxy.async + wgmma.fence

for kk in 0..BK/16:
    B_desc = make_wgmma_b_desc(B_smem_base + kk * 2048)
    for m in 0..M_ITERS:
        A_row = wg_id * M_PER_WG + m * 64
        A_desc = make_wgmma_a_desc(&A_sh[buf][A_row][0], kk)  ← no ldmatrix!
        wgmma_SS_call(acc[m], A_desc, B_desc)   ← acc += A × B

wgmma_commit()
wgmma_drain()
```

---

## Expected Improvements vs h2_s5

1. **Smaller SMEM**: same formula but BK=32 now viable (was effectively BK=64-only
   for good A swizzle). BM=128, BN=256, BK=32, NS=4: 96 KB vs 192 KB → enables
   2 CTAs/SM instead of 1 → potential 2× occupancy.

2. **No ldmatrix bank conflicts**: SS mode eliminates all A ldmatrix calls.

3. **Fewer registers**: no A fragments in registers (~168 vs ~186), potentially
   enabling larger configs within register budget.

4. **0% membar stalls**: cp.async + `__pipeline_wait_prior` vs mbarrier.

Reference: Triton PTX at
`triton_ptx/triton_bf16_bm128_bn256_bk32_ns4_nw8_n8192.ptx`
