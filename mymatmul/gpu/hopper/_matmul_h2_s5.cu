#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

/*
 * H2 Stage 5: TMA + wgmma multi-stage pipeline with tunable depth (NS ∈ {2..5}).
 *
 * ── Three hardware engines, three coordination problems ──────────────────────
 *
 *   ┌────────────┐  doorbell (mbarrier)  ┌──────────────────┐
 *   │ TMA engine │ ──── rings when ────► │  Warp execution  │
 *   │ (DMA unit) │      data arrives     │  (ldmatrix, ctrl)│
 *   └────────────┘                       └────────┬─────────┘
 *        │  writes SMEM                           │ issues wgmma calls
 *        │                    fence.proxy.async   ▼
 *        └──────────────────────────────► ┌──────────────┐
 *              makes writes visible to    │ Tensor core  │
 *              wgmma's async SMEM view    │  (wgmma unit)│
 *                                         └──────────────┘
 *
 * ── Pipeline: NS slots, each holding one A+B tile ────────────────────────────
 *
 *   Prologue  : fill NS-1 slots with TMA loads (ring doorbells when ready)
 *   Main loop : for each k-tile: load next slot, wait current doorbell, compute
 *   Drain     : drain last NS-1 slots (no new loads, just wait+compute)
 *
 *   Deeper NS → more tiles in flight → more time for TMA to finish → fewer stalls.
 *   Limit: SMEM = NS × (BM×BK + BK×BN) × 2 bytes must fit in 200 KB.
 *
 * Compiled with -arch=sm_90a, single cubin LB=1.
 */

#ifndef LB_MIN_BLOCKS
#define LB_MIN_BLOCKS 1
#endif

struct alignas(64) TmaDesc { uint64_t opaque[16]; };

// ── Doorbell (mbarrier): hardware signal between TMA engine and warps ─────────
//
// Each pipeline slot has one doorbell in SMEM. The protocol per slot:
//   1. doorbell_reset()  — arm it: "not ready, expecting one TMA delivery"
//   2. doorbell_arm_tma()— tell TMA engine: "ring this doorbell when N bytes arrive"
//      + issue the actual TMA load (tma_load_tile)
//   3. doorbell_wait()   — block all warps until TMA rings (tile is in SMEM)

__device__ __forceinline__ void doorbell_reset(uint64_t* db) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                 :: "r"((uint32_t)__cvta_generic_to_shared(db)) : "memory");
}

__device__ __forceinline__ void doorbell_arm_tma(uint64_t* db, uint32_t bytes_expected) {
    // "arrive" (software side of count=1) + declare how many bytes TMA will deliver
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(db);
    uint64_t token;
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;\n"
                 : "=l"(token) : "r"(addr), "r"(bytes_expected) : "memory");
}

__device__ __forceinline__ void doorbell_wait(uint64_t* db) {
    // Spin until TMA rings: tile is safe to read
    uint32_t done = 0, addr = (uint32_t)__cvta_generic_to_shared(db);
    while (!done) {
        asm volatile("{\n.reg .pred P;\n"
                     "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P, [%1], 0;\n"
                     "selp.u32 %0, 1, 0, P;\n}\n"
                     : "=r"(done) : "r"(addr) : "memory");
    }
}

// ── TMA tile load: DMA engine pulls one A+B tile from HBM into SMEM ──────────
// Thread 0 issues this; hardware notifies the doorbell when delivery completes.

__device__ __forceinline__ void tma_load_tile(
    const TmaDesc* desc, void* smem_ptr, uint64_t* db, int32_t coord0, int32_t coord1
) {
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t db_addr = (uint32_t)__cvta_generic_to_shared(db);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
        :: "r"(dst), "l"((unsigned long long)desc), "r"(db_addr),
           "r"(coord0), "r"(coord1) : "memory");
}

// ── wgmma coordination: three fences that bridge the three hardware worlds ────
//
//   wgmma_begin():  two fences before issuing wgmma calls —
//     fence.proxy.async  : TMA's SMEM writes → visible to wgmma's async-proxy view
//     wgmma.fence        : ldmatrix's register writes → visible to wgmma's engine
//   wgmma_commit():  tell tensor core "here is a batch of multiply work"
//   wgmma_drain():   block thread until accumulator registers are ready to read

__device__ __forceinline__ void wgmma_begin() {
    asm volatile("fence.proxy.async;\n"          ::: "memory");
    asm volatile("wgmma.fence.sync.aligned;\n"   ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_drain() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// ── ldmatrix: load A fragment from SMEM into registers (one per wgmma call) ──

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_ptr
) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(smem_ptr));
}

// ── wgmma B SMEM descriptor ───────────────────────────────────────────────────
//
// B is stored [BK][BN] row-major in SMEM, written by TMA with 128B swizzle.
// wgmma reads it as MN-major (N-contiguous), transB=1.
//
// GmmaDescriptor bit layout (from CUTLASS mma_sm90_desc.hpp):
//   [13:0]  start_address_ : smem_addr[17:4]  (>> 4, 14 bits)
//   [29:16] leading_byte_offset_ : LBO >> 4
//   [45:32] stride_byte_offset_  : SBO >> 4
//   [63:62] layout_type_ : 1 = B128
//
// LBO, SBO from tilelang MN-major formula (b_swizzle = 128B):
//   n_atoms = BN / 64  (64 bf16 per 128-byte swizzle atom)
//   if n_atoms <= 1: LBO = 0,           SBO = BN
//   else:            LBO = 8 * BK,      SBO = 64

template<int BN, int BK>
__device__ __forceinline__ uint64_t make_wgmma_b_desc(uint32_t smem_addr) {
    constexpr uint64_t LAYOUT_B128 = 1ULL << 62;
    constexpr int n_atoms = BN / 64;
    constexpr uint64_t lbo = (n_atoms <= 1) ? 0ULL        : (uint64_t)(8 * BK);
    constexpr uint64_t sbo = (n_atoms <= 1) ? (uint64_t)BN : 64ULL;
    uint64_t start = (uint64_t)(smem_addr >> 4) & 0x3FFF;
    return start | (lbo << 16) | (sbo << 32) | LAYOUT_B128;
}

// ── wgmma.mma_async wrappers ──────────────────────────────────────────────────
//
// Always accumulate (scaleD=1); caller zero-inits acc[].
// transB=1 = N-major B (our [BK][BN] row-major layout).

__device__ __forceinline__
void wgmma_m64n64k16(float d[32], uint32_t a[4], uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "{%32,%33,%34,%35},%36,1,1,1,1;\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
         "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
         "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(b));
}

__device__ __forceinline__
void wgmma_m64n128k16(float d[64], uint32_t a[4], uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "{%64,%65,%66,%67},%68,1,1,1,1;\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
         "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
         "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
         "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
         "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
         "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
         "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(b));
}

__device__ __forceinline__
void wgmma_m64n256k16(float d[128], uint32_t a[4], uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,"
        "%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,"
        "%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,"
        "%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,"
        "%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127},"
        "{%128,%129,%130,%131},%132,1,1,1,1;\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
         "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
         "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
         "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
         "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
         "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
         "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),
         "+f"(d[64]),"+f"(d[65]),"+f"(d[66]),"+f"(d[67]),"+f"(d[68]),"+f"(d[69]),"+f"(d[70]),"+f"(d[71]),
         "+f"(d[72]),"+f"(d[73]),"+f"(d[74]),"+f"(d[75]),"+f"(d[76]),"+f"(d[77]),"+f"(d[78]),"+f"(d[79]),
         "+f"(d[80]),"+f"(d[81]),"+f"(d[82]),"+f"(d[83]),"+f"(d[84]),"+f"(d[85]),"+f"(d[86]),"+f"(d[87]),
         "+f"(d[88]),"+f"(d[89]),"+f"(d[90]),"+f"(d[91]),"+f"(d[92]),"+f"(d[93]),"+f"(d[94]),"+f"(d[95]),
         "+f"(d[96]),"+f"(d[97]),"+f"(d[98]),"+f"(d[99]),"+f"(d[100]),"+f"(d[101]),"+f"(d[102]),"+f"(d[103]),
         "+f"(d[104]),"+f"(d[105]),"+f"(d[106]),"+f"(d[107]),"+f"(d[108]),"+f"(d[109]),"+f"(d[110]),"+f"(d[111]),
         "+f"(d[112]),"+f"(d[113]),"+f"(d[114]),"+f"(d[115]),"+f"(d[116]),"+f"(d[117]),"+f"(d[118]),"+f"(d[119]),
         "+f"(d[120]),"+f"(d[121]),"+f"(d[122]),"+f"(d[123]),"+f"(d[124]),"+f"(d[125]),"+f"(d[126]),"+f"(d[127])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(b));
}

template<int BN>
__device__ __forceinline__
void wgmma_call(float* acc, uint32_t a[4], uint64_t desc_b) {
    if constexpr      (BN == 64)  wgmma_m64n64k16 (acc, a, desc_b);
    else if constexpr (BN == 128) wgmma_m64n128k16(acc, a, desc_b);
    else if constexpr (BN == 256) wgmma_m64n256k16(acc, a, desc_b);
}

// ── Kernel implementation ─────────────────────────────────────────────────────

template<int BM, int BN, int BK, int NUM_WG, int NUM_STAGES>
__device__ __forceinline__ void h2s5_impl(
    const TmaDesc& tma_A,
    const TmaDesc& tma_B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    // M_ITERS: how many 64-row wgmma calls each warpgroup issues per kk step.
    // M_ITERS=1 replicates h2_s3; M_ITERS=2 with BM=256,NUM_WG=2 is the new case.
    static_assert(BM % (NUM_WG * 64) == 0, "BM must be multiple of NUM_WG*64");
    constexpr int M_ITERS  = BM / (NUM_WG * 64);
    constexpr int M_PER_WG = BM / NUM_WG;  // = M_ITERS * 64
    constexpr int D        = BN / 2;

    // 1D thread block: NUM_WG * 128 threads
    const int tid        = threadIdx.x;
    const int wg_id      = tid / 128;         // which warpgroup (0..NUM_WG-1)
    const int local_warp = (tid % 128) / 32;  // warp within warpgroup (0..3)
    const int lane       = tid % 32;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    constexpr int NS = NUM_STAGES;

    // ── SMEM layout ───────────────────────────────────────────────────────────
    // A[NS][BM][BK], B[NS][BK][BN], mbar[NS]
    extern __shared__ char smem_raw[];
    constexpr int A_BYTES  = NS * BM * BK * 2;
    constexpr int B_BYTES  = NS * BK * BN * 2;
    constexpr int MBAR_OFF = (A_BYTES + B_BYTES + 7) & ~7;

    auto A_sh = reinterpret_cast<__nv_bfloat16 (*)[BM][BK]>(smem_raw);
    auto B_sh = reinterpret_cast<__nv_bfloat16 (*)[BK][BN]>(smem_raw + A_BYTES);
    auto mbar = reinterpret_cast<uint64_t*>(smem_raw + MBAR_OFF);

    // ── Accumulators: M_ITERS separate D-element arrays (zero-initialised) ──
    // acc[m] holds the wgmma outputs for the m-th 64-row M-slice of this warpgroup.
    float acc[M_ITERS][D] = {};

    // A swizzle: only BK=64 uses TMA 128B swizzle (period=8 rows, 16B-chunk-aligned XOR).
    // BK=16/32 use SWIZZLE_NONE (64B/32B swizzle XORs at 8B granularity which
    // breaks ldmatrix 16B chunk boundaries and corrupts data).
    constexpr int A_SWZ_PERIOD = (BK == 64) ? 8 : 0;

    const int num_tiles = K / BK;

    // ── B SMEM sub-tile helpers ───────────────────────────────────────────────
    // For 128B swizzle, boxDim[0] must be exactly 64 BF16 (= 128 bytes).
    // A BN-wide B tile is loaded as BN/64 sub-tiles of 64 columns each, packed
    // back-to-back: sub-tile i starts at B_sh[buf][0][0] + i * BK * 64 * 2 bytes.
    // The wgmma descriptor's LBO encodes the stride between consecutive sub-tiles.
    //
    // Within a pipeline stage, advancing past kk*16 K-rows inside sub-tile 0:
    //   byte offset = kk * 16 * 64 * 2  (16 K-rows × 64 BF16-wide × 2 bytes)
    // This is the formula used for the wgmma descriptor start_address per kk step.
    constexpr int SUBTILE_COL = 64;                           // BF16 per sub-tile column
    constexpr int K_STEP_BYTES = 16 * SUBTILE_COL * 2;       // = 2048: bytes per kk advance
    // (For BN==64, sub-tile stride = BK*128 = BK*K_STEP_BYTES/16 = ... but only one sub-tile)

    // ── LOAD_TILE: thread 0 arms the doorbell then tells TMA to deliver the tile ─
    // A: one load of BM×BK BF16 rows.
    // B: BN/64 loads of 64-column sub-tiles (128B-swizzle constraint forces boxDim=64).
#define LOAD_TILE(slot_, k_start_)                                                  \
    do {                                                                            \
        doorbell_arm_tma(&mbar[(slot_)], (BM * BK + BK * BN) * 2);                \
        tma_load_tile(&tma_A, &A_sh[(slot_)][0][0], &mbar[(slot_)],               \
                      (k_start_), block_row);                                       \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < BN / SUBTILE_COL; _i++) {                            \
            void* _b = (char*)(&B_sh[(slot_)][0][0]) + _i * BK * SUBTILE_COL * 2; \
            tma_load_tile(&tma_B, _b, &mbar[(slot_)],                              \
                          block_col + _i * SUBTILE_COL, (k_start_));               \
        }                                                                           \
    } while (0)

    // ── Prologue: install NS doorbells, prefill pipeline with NS-1 tiles ─────
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NS; s++) doorbell_reset(&mbar[s]);
        #pragma unroll
        for (int s = 0; s < NS - 1; s++) LOAD_TILE(s, s * BK);
    }
    __syncthreads();

    // ── MULTIPLY_TILE: wgmma compute on one pipeline slot ────────────────────
    // See wgmma_begin/commit/drain above for the 3-step protocol explanation.
#define MULTIPLY_TILE(slot_)                                                        \
    do {                                                                            \
        wgmma_begin();                   /* bridge TMA→async-proxy, regs→wgmma */  \
        _Pragma("unroll")                                                           \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                 \
            uint32_t _bb = (uint32_t)__cvta_generic_to_shared(&B_sh[(slot_)][0][0]);\
            uint64_t _db = make_wgmma_b_desc<BN,BK>(_bb + _kk * K_STEP_BYTES);    \
            _Pragma("unroll")                                                       \
            for (int _m = 0; _m < M_ITERS; _m++) {                                 \
                uint32_t _a[4];                                                     \
                const int _ar  = wg_id*M_PER_WG + _m*64 + local_warp*16+(lane%16);\
                const int _alg = _kk*2 + (lane/16);                                \
                const int _aph = (A_SWZ_PERIOD>0) ? (_alg^(_ar%A_SWZ_PERIOD)):_alg;\
                ldmatrix_x4(_a[0],_a[1],_a[2],_a[3],                              \
                    (uint32_t)__cvta_generic_to_shared(&A_sh[(slot_)][_ar][_aph*8]));\
                wgmma_call<BN>((float*)acc[_m], _a, _db);  /* acc += A × B */      \
            }                                                                       \
        }                                                                           \
        wgmma_commit();                  /* submit batch to tensor core engine */   \
        wgmma_drain();                   /* acc[] is ready for next iteration  */   \
    } while (0)

    // ── Main K loop: tiles 0..num_tiles-NS  (always issues a new tile) ───────
    for (int k = 0; k < num_tiles - (NS - 1); k++) {
        const int cur = k % NS;
        const int nxt = (k + NS - 1) % NS;

        if (tid == 0) {
            // Reset and arm the nxt doorbell, then kick off TMA for the next tile.
            // Safe: nxt slot was consumed NS-1 iterations ago (behind __syncthreads).
            doorbell_reset(&mbar[nxt]);
            LOAD_TILE(nxt, (k + NS - 1) * BK);
        }

        doorbell_wait(&mbar[cur]);    // wait: cur tile is in SMEM
        MULTIPLY_TILE(cur);           // compute: acc += A[cur] × B[cur]
        __syncthreads();              // all warps done before next iter reuses cur
    }

    // ── Drain: last NS-1 tiles that were issued but not yet waited on ────────
    #pragma unroll
    for (int d = NS - 2; d >= 0; d--) {
        const int slot = (num_tiles - d - 1) % NS;
        doorbell_wait(&mbar[slot]);
        MULTIPLY_TILE(slot);
        __syncthreads();
    }

#undef LOAD_TILE
#undef MULTIPLY_TILE

    // ── Write accumulators to C ───────────────────────────────────────────────
    // Same wgmma output layout as h2_s3, now wrapped in an m-iteration loop.
    // m=0: rows [wg_id*M_PER_WG .. wg_id*M_PER_WG+63]
    // m=1: rows [wg_id*M_PER_WG+64 .. wg_id*M_PER_WG+127]  (if M_ITERS >= 2)
    // etc.
    const int base_col = (lane % 4) * 2;
    const int base_row = lane / 4;
    #pragma unroll
    for (int m = 0; m < M_ITERS; m++) {
        #pragma unroll
        for (int j = 0; j < BN / 8; j++) {
            const int gc  = block_col + j * 8 + base_col;
            const int gr0 = block_row + wg_id * M_PER_WG + m * 64
                            + local_warp * 16 + base_row;
            const int gr8 = gr0 + 8;
            if (gr0 < M && gc < N)
                *reinterpret_cast<__nv_bfloat162*>(&C[gr0 * N + gc]) =
                    __floats2bfloat162_rn(acc[m][j*4+0], acc[m][j*4+1]);
            if (gr8 < M && gc < N)
                *reinterpret_cast<__nv_bfloat162*>(&C[gr8 * N + gc]) =
                    __floats2bfloat162_rn(acc[m][j*4+2], acc[m][j*4+3]);
        }
    }
}

// ── Kernel entry points ───────────────────────────────────────────────────────

#define MAKE_LAUNCHER(BM_, BN_, BK_, NG_, NS_)                                   \
extern "C" __global__ __launch_bounds__(NG_ * 128, LB_MIN_BLOCKS)               \
void matmul_h2s5_bm##BM_##_bn##BN_##_bk##BK_##_wg##NG_##_ns##NS_(              \
    const __grid_constant__ TmaDesc tma_A,                                       \
    const __grid_constant__ TmaDesc tma_B,                                       \
    __nv_bfloat16* __restrict__ C, int M, int K, int N)                          \
{                                                                                 \
    h2s5_impl<BM_, BN_, BK_, NG_, NS_>(tma_A, tma_B, C, M, K, N);              \
}

// NS=2,3,4 for key configs. Exclude BM=256,NG=1,BN=256 (acc regs = 512 = limit).
// For each (BM,BN,BK,NG) combo, instantiate NS=2,3,4.

#define MAKE3(BM_, BN_, BK_, NG_) \
    MAKE_LAUNCHER(BM_, BN_, BK_, NG_, 2) \
    MAKE_LAUNCHER(BM_, BN_, BK_, NG_, 3) \
    MAKE_LAUNCHER(BM_, BN_, BK_, NG_, 4) \
    MAKE_LAUNCHER(BM_, BN_, BK_, NG_, 5)

// NW=1 warpgroup
MAKE3( 64,  64, 16, 1) MAKE3( 64,  64, 32, 1) MAKE3( 64,  64, 64, 1)
MAKE3( 64, 128, 16, 1) MAKE3( 64, 128, 32, 1) MAKE3( 64, 128, 64, 1)
MAKE3( 64, 256, 16, 1) MAKE3( 64, 256, 32, 1) MAKE3( 64, 256, 64, 1)
MAKE3(128,  64, 16, 1) MAKE3(128,  64, 32, 1) MAKE3(128,  64, 64, 1)
MAKE3(128, 128, 16, 1) MAKE3(128, 128, 32, 1) MAKE3(128, 128, 64, 1)
MAKE3(128, 256, 16, 1) MAKE3(128, 256, 32, 1) MAKE3(128, 256, 64, 1)
MAKE3(256,  64, 16, 1) MAKE3(256,  64, 32, 1) MAKE3(256,  64, 64, 1)
MAKE3(256, 128, 16, 1) MAKE3(256, 128, 32, 1) MAKE3(256, 128, 64, 1)

// NW=2 warpgroups
MAKE3(128,  64, 16, 2) MAKE3(128,  64, 32, 2) MAKE3(128,  64, 64, 2)
MAKE3(128, 128, 16, 2) MAKE3(128, 128, 32, 2) MAKE3(128, 128, 64, 2)
MAKE3(128, 256, 16, 2) MAKE3(128, 256, 32, 2) MAKE3(128, 256, 64, 2)
MAKE3(256,  64, 16, 2) MAKE3(256,  64, 32, 2) MAKE3(256,  64, 64, 2)
MAKE3(256, 128, 16, 2) MAKE3(256, 128, 32, 2) MAKE3(256, 128, 64, 2)
MAKE3(256, 256, 16, 2) MAKE3(256, 256, 32, 2) MAKE3(256, 256, 64, 2)

#undef MAKE3
#undef MAKE_LAUNCHER
