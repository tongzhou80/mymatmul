# Triton PTX Analysis: BM=128, BN=256, BK=32, NS=4, NW=8

## Key Differences vs h2_s6

| Feature | h2_s6 | Triton PTX |
|---------|--------|------------|
| `wgmma.wait_group` | 0 (full drain) | **1** (keep 1 in flight) |
| `fence.proxy.async` | yes | **no** (only wgmma.fence) |
| SMEM layout | A first, B after | **B first, A after** |
| Loop: issue position | before compute | **after wait_group 1** |
| Drain loop | yes | **no (unified loop)** |
| `cp.async.wait_group` | wait_prior(NS-1) | `wait_group NS` (= 4) |

---

## `wgmma.wait_group 1` — the critical insight

In the main loop (line 614):
```ptx
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;   ← NOT 0
```

This keeps **1 wgmma group in flight** while loading the next tile.
- After iter k: group k committed, group k-1 guaranteed done.
- Group k still potentially running in tensor core.
- Hardware serializes acc writes automatically — wgmma[k+1] reads
  correct acc values because wgmma.fence at start of k+1 creates
  the dependency barrier.
- `wait_group 0` only at the final epilogue drain.

---

## Main Loop Structure

```
$L__BB0_2:                             // k = 0..num_tiles-1
  advance cur_stage ring counter (0→3)

  cp.async.wait_group 4;               // keep NS=4 commits in flight
  bar.sync 0;                          // sync all threads

  // build A and B descriptors for cur_stage
  wgmma.fence.sync.aligned;           // NO fence.proxy.async
  wgmma m64n256k16 (kk=0);            // acc += A[cur][0..15] × B[cur]
  wgmma m64n256k16 (kk=1);            // acc += A[cur][16..31] × B[cur]
  wgmma.commit_group.sync.aligned;

  wgmma.wait_group.sync.aligned 1;    // wait for k-1, keep k in flight

  advance nxt_stage ring counter
  bar.sync 0;                          // sync before cp.async issue

  cp.async A[nxt] (guarded by k < num_tiles-NS)
  cp.async.commit_group;
  cp.async B[nxt] (guarded by k < num_tiles-NS)
  cp.async.commit_group;

  k++
  bra $L__BB0_2

$L__BB0_3:
  wgmma.wait_group.sync.aligned 0;    // drain final group
  cp.async.wait_group 0;              // drain final cp.async
  bar.sync 0;
  // epilogue: write C
```

Key: **ISSUE comes AFTER wait_group 1**, not before. This is required
for correctness — wait_group 1 at iter k proves wgmma[k-1] has
released its SMEM slot, making it safe to overwrite.

---

## SMEM Layout

B first, A second (opposite of h2_s5/h2_s6):
```
smem[0      .. 65535]  = B stages 0..3  (4 × BN×BK×2 = 4×16384 = 64KB)
smem[65536  .. 98303]  = A stages 0..3  (4 × BM×BK×2 = 4×8192  = 32KB)
```

B stage s: `smem + s * 16384`
A stage s: `smem + 65536 + s * 8192`

Per-warpgroup A offset (from shfl.sync.idx to read warp_id):
  WG0: A_desc_start = A_stage_base >> 4
  WG1: A_desc_start = (A_stage_base + BM/2*BK*2) >> 4 = A_desc + 256

---

## A Tile Loading

Each thread loads 2 groups of 16 bytes each:
- Group 1: `A[block_m + tid/4][k0 + (tid%4)*8 .. +7]`  (rows 0..63)
- Group 2: same col, row+64                              (rows 64..127)

A SMEM swizzle offset `r16`:
```
r14 = tid * 16
r85 = r14 & 0xFF0                  // column byte offset
r86 = (tid & 24) * 2               // row-group XOR  (groups of 2 rows)
r16 = r85 XOR r86                  // = (col_group XOR (row/2)%4) * 16
```
This matches h2_s6's formula: `(col/8 ^ (row/A_SHIFT) % A_SWZ) * 8`.

A section base: `smem + 65536`. Stage s: `+s*8192`.

---

## B Tile Loading (2 commits per tile vs h2_s6's 1)

Each tile: 2 separate `cp.async.commit_group` calls:
1. Commit A (2 cp.async × 16 bytes = 32 bytes/thread)
2. Commit B (4 cp.async × 16 bytes = 64 bytes/thread)

→ `cp.async.wait_group 4` = NS=4 commits = 2 stages in flight.

B SMEM offset `r18` (complex warp-level swizzle):
```
r89 = (tid & 0x18) << 9            // row-within-8 → high bits
r17 = (tid*16) & 0x70              // col bits
r90 = (tid & 0xE0) << 2            // warp_id contribution
r91 = (tid & 0xE0) >> 1
r92 = r89 | r17
r93 = r90 | r91
r18 = r92 XOR r93
```

B section: `smem + stage*16384 + r18`. Sub-tile advances: +1024, +2048, +3072.

---

## GmmaDescriptor OR Masks (verified)

A (B64, BK=32):
  mask = 0x8000002000000000
  = layout(2=B64) << 62 | SBO(32) << 32 | LBO(0) << 16

B (B128):
  mask = 0x4000004000000000
  = layout(1=B128) << 62 | SBO(64) << 32 | LBO(8*BK=256) << 16

kk=0 → kk=1 advances:
  A: +2 to start_address (= +32 bytes = 16 K-elements)
  B: +128 to start_address (= +2048 bytes = 16 K-rows × 64 BF16)

---

## cp.async.wait_group N semantics

`cp.async.wait_group N` = keep at most N outstanding commits.
Triton uses 2 commits/tile and `wait_group 4` = 2 tiles in flight.
h2_s6 uses 1 commit/tile and `wait_prior(NS-1)` = NS-1 commits.
Equivalent: both keep NS-1 tile loads in flight. No change needed.

---

## Occupancy

regs=168/thread (from NCU), smem=96KB:
- 2 WGs × 128 threads × 168 regs = 43008 regs / CTA
- 65536 regs / SM → 1 CTA/SM (register limited)
- smem: 96KB/SM → 2 CTAs would fit, but regs prevent it

Our h2_s6 has 178-218 regs → also 1 CTA/SM. The register difference
(168 vs 178+) comes from Triton's M_ITERS=1 (no M loop unrolling)
and fewer compiler intermediates. SM SoL 91.8% achieved primarily
through wgmma.wait_group 1 eliminating wgmma drain stalls.
