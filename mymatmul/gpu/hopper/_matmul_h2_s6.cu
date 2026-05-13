#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>

/*
 * H2 Stage 6: cp.async + wgmma SS mode (both A and B from SMEM descriptors).
 *
 * Key differences from h2_s5 (TMA + wgmma RS):
 *   1. cp.async replaces TMA: all threads load tiles via __pipeline_memcpy_async.
 *      __pipeline_wait_prior replaces mbarrier.
 *   2. wgmma SS mode: A read from SMEM via GmmaDescriptor — no ldmatrix for A.
 *      Extra transA=0 parameter in PTX vs RS mode.
 *   3. A swizzle period = BK/8 (B32/B64/B128 for BK=16/32/64).
 *      In RS mode only BK=64 worked; ldmatrix's 16B alignment broke B64/B32 XOR.
 *      SS mode has no such constraint — all BK values work.
 *   4. Smaller SMEM: BM=128, BN=256, BK=32, NS=4 → 96 KB vs 192 KB → 2 CTAs/SM.
 *
 * B sub-tile layout in SMEM:
 *   B is stored as (BN/64) sub-tiles of [BK][64] packed back-to-back per stage.
 *   Within each sub-tile: 128B swizzle (physical_col_group = logical XOR (row%8)).
 *   Required because 128B swizzle needs boxDim ≤ 64 BF16 — the GmmaDescriptor LBO
 *   field bridges between sub-tiles.
 *   cp.async reads from global row-major B[K][N] and writes to SMEM sub-tile format.
 *
 * GmmaDescriptor bit layout (CUTLASS mma_sm90_desc.hpp):
 *   [13: 0] start_address  = smem_addr >> 4
 *   [29:16] LBO            = leading_byte_offset >> 4
 *   [45:32] SBO            = stride_byte_offset  >> 4
 *   [63:62] layout_type    = 1:B128, 2:B64, 3:B32
 */

#ifndef LB_MIN_BLOCKS
#define LB_MIN_BLOCKS 1
#endif

// ── wgmma coordination (same semantics as h2_s5) ─────────────────────────────
__device__ __forceinline__ void wgmma_begin() {
    asm volatile("fence.proxy.async;\n"        ::: "memory");
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_drain() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// ── A GmmaDescriptor (K-major [BM][BK]) ──────────────────────────────────────
//   layout: BK=64→B128(1), BK=32→B64(2), BK=16→B32(3)
//   LBO = 0     (K is inner/contiguous, no sub-tile jump)
//   SBO = BK    (8 M-rows × BK BF16 × 2 bytes / 16 = BK)
//   kk advance: +2 per kk (= 32 bytes = 16 K-elements)

template<int BK>
__device__ __forceinline__ uint64_t make_wgmma_a_desc(uint32_t smem_addr, int kk) {
    constexpr uint64_t layout = (BK == 64) ? 1ULL : (BK == 32) ? 2ULL : 3ULL;
    constexpr uint64_t sbo = (uint64_t)BK;
    uint64_t start = ((uint64_t)(smem_addr >> 4) & 0x3FFFULL) + (uint64_t)(kk * 2);
    return start | (sbo << 32) | (layout << 62);
}

// ── B GmmaDescriptor (sub-tile [BK][64], N_SUBTILES sub-tiles) ───────────────
//   layout: B128 always
//   LBO = 8*BK  (sub-tile 0→1 jump: BK×64×2 bytes / 16 = 8×BK)
//   SBO = 64    (8 K-rows × 64 BF16 × 2 bytes / 16 = 64)
//   kk advance: caller adds kk*2048 to smem_addr before calling

template<int BN, int BK>
__device__ __forceinline__ uint64_t make_wgmma_b_desc(uint32_t smem_addr) {
    constexpr uint64_t LAYOUT_B128 = 1ULL << 62;
    constexpr int n_atoms = BN / 64;
    constexpr uint64_t lbo = (n_atoms <= 1) ? 0ULL : (uint64_t)(8 * BK);
    constexpr uint64_t sbo = 64ULL;
    uint64_t start = (uint64_t)(smem_addr >> 4) & 0x3FFFULL;
    return start | (lbo << 16) | (sbo << 32) | LAYOUT_B128;
}

// ── wgmma SS wrappers: both A and B from SMEM descriptors ────────────────────
// scaleD=1 (accumulate), transA=0 (K-major A), transB=1 (N-major B).

__device__ __forceinline__
void wgmma_ss_m64n64k16(float d[32], uint64_t a, uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32,%33,1,1,1,0,1;\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
         "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
         "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
        :"l"(a),"l"(b));
}

__device__ __forceinline__
void wgmma_ss_m64n128k16(float d[64], uint64_t a, uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "%64,%65,1,1,1,0,1;\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
         "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
         "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
         "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
         "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
         "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
         "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
        :"l"(a),"l"(b));
}

__device__ __forceinline__
void wgmma_ss_m64n256k16(float d[128], uint64_t a, uint64_t b) {
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
        "%128,%129,1,1,1,0,1;\n"
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
        :"l"(a),"l"(b));
}

template<int BN>
__device__ __forceinline__
void wgmma_ss_call(float* acc, uint64_t desc_a, uint64_t desc_b) {
    if constexpr      (BN ==  64) wgmma_ss_m64n64k16 (acc, desc_a, desc_b);
    else if constexpr (BN == 128) wgmma_ss_m64n128k16(acc, desc_a, desc_b);
    else if constexpr (BN == 256) wgmma_ss_m64n256k16(acc, desc_a, desc_b);
}

// ── Kernel implementation ─────────────────────────────────────────────────────

template<int BM, int BN, int BK, int NUM_WG, int NUM_STAGES>
__device__ __forceinline__ void h2s6_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    static_assert(BM % (NUM_WG * 64) == 0, "BM must be multiple of NUM_WG*64");
    static_assert(BN % 64 == 0, "BN must be multiple of 64");

    constexpr int NS         = NUM_STAGES;
    constexpr int M_ITERS    = BM / (NUM_WG * 64);
    constexpr int M_PER_WG   = BM / NUM_WG;
    constexpr int D          = BN / 2;            // acc floats per m64nBNk16
    constexpr int N_SUBTILES = BN / 64;

    // A swizzle: same XOR formula as h1_ms, works for all BK in SS mode.
    // A_SWZ rows share the same XOR value; A_SHIFT controls how many rows per group.
    // Verified from Triton PTX (BK=32): rows 0,1 → XOR=0; rows 2,3 → XOR=1; etc.
    constexpr int A_SWZ   = BK / 8;     // number of distinct XOR values (period)
    constexpr int A_SHIFT = 64 / BK;    // rows per XOR group (=A_SWZ_PERIOD=2 for BK=32)

    // cp.async thread assignment
    constexpr int THREADS  = NUM_WG * 128;
    constexpr int A_ELEM   = (BM * BK / THREADS >= 8) ? 8 : 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = (BK * BN / THREADS >= 8) ? 8 : 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    // B kk advance: one 16-K-row step within a 64-col sub-tile
    constexpr int K_STEP_BYTES = 16 * 64 * 2;    // = 2048 bytes

    const int tid        = threadIdx.x;
    const int wg_id      = tid / 128;
    const int local_warp = (tid % 128) / 32;
    const int lane       = tid % 32;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // ── SMEM layout ───────────────────────────────────────────────────────────
    // A: [NS][BM][BK]                — K-major, A_SWZ_PERIOD XOR
    // B: [NS][N_SUBTILES][BK][64]    — sub-tile format, row%8 XOR per sub-tile
    extern __shared__ char smem_raw[];
    constexpr int A_BYTES = NS * BM * BK * 2;

    auto A_sh = reinterpret_cast<__nv_bfloat16 (*)[BM][BK]>(smem_raw);
    auto B_sh = reinterpret_cast<__nv_bfloat16 (*)[N_SUBTILES][BK][64]>(smem_raw + A_BYTES);

    float acc[M_ITERS][D] = {};

    const int num_tiles = K / BK;

    // ── ISSUE_TILE: cp.async load A (BM×BK) and B (BK×BN → sub-tile format) ──
    //
    // A: each thread loads A_ELEM consecutive K-elements in one row.
    //    Global → SMEM with XOR swizzle: physical_col = (col/8 ^ row%period)*8 + col%8
    //
    // B: each thread loads B_ELEM consecutive column-elements within one sub-tile row.
    //    Linear assignment across [N_SUBTILES][BK][64] (sub-tile first, then K-row).
    //    Global B is row-major [K][N]; SMEM destination is sub-tile format.
#define ISSUE_TILE(k0_, buf_)                                                       \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                    \
            const int _g  = tid + _i * THREADS;                                    \
            const int _r  = (_g * A_ELEM) / BK;                                    \
            const int _c  = (_g * A_ELEM) % BK;                                    \
            const int _sc = ((_c / 8) ^ ((_r / A_SHIFT) % A_SWZ)) * 8 + (_c % 8);\
            __pipeline_memcpy_async(                                                \
                &A_sh[(buf_)][_r][_sc],                                             \
                &A[(block_row + _r) * K + (k0_) + _c],                             \
                A_ELEM * (int)sizeof(__nv_bfloat16));                               \
        }                                                                           \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                    \
            const int _g    = tid + _i * THREADS;                                  \
            const int _flat = _g * B_ELEM;                                         \
            const int _st   = _flat / (BK * 64);                                   \
            const int _kr   = (_flat % (BK * 64)) / 64;                            \
            const int _sc0  = (_flat % (BK * 64)) % 64;                            \
            const int _sc   = ((_sc0 / 8) ^ (_kr % 8)) * 8 + (_sc0 % 8);         \
            __pipeline_memcpy_async(                                                \
                &B_sh[(buf_)][_st][_kr][_sc],                                      \
                &B[((k0_) + _kr) * N + block_col + _st * 64 + _sc0],              \
                B_ELEM * (int)sizeof(__nv_bfloat16));                               \
        }                                                                           \
        __pipeline_commit();                                                         \
    } while (0)

    // ── MULTIPLY_TILE: wgmma SS compute on one pipeline slot ─────────────────
#define MULTIPLY_TILE(slot_)                                                        \
    do {                                                                            \
        wgmma_begin();                                                              \
        _Pragma("unroll")                                                           \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                 \
            const uint32_t _bb = (uint32_t)__cvta_generic_to_shared(               \
                                     &B_sh[(slot_)][0][0][0]);                     \
            const uint64_t _db = make_wgmma_b_desc<BN, BK>(                       \
                                     (uint32_t)(_bb + _kk * K_STEP_BYTES));        \
            _Pragma("unroll")                                                       \
            for (int _m = 0; _m < M_ITERS; _m++) {                                 \
                const int _mrow = wg_id * M_PER_WG + _m * 64;                     \
                const uint32_t _aa = (uint32_t)__cvta_generic_to_shared(           \
                                         &A_sh[(slot_)][_mrow][0]);                \
                const uint64_t _da = make_wgmma_a_desc<BK>(_aa, _kk);             \
                wgmma_ss_call<BN>((float*)acc[_m], _da, _db);                      \
            }                                                                       \
        }                                                                           \
        wgmma_commit();                                                             \
        wgmma_drain();                                                              \
    } while (0)

    // ── Prologue: fill pipeline with first NS-1 tiles ─────────────────────────
    #pragma unroll
    for (int s = 0; s < NS - 1; s++) {
        ISSUE_TILE(s * BK, s);
    }

    // ── Main K loop ───────────────────────────────────────────────────────────
    for (int k = 0; k < num_tiles - (NS - 1); k++) {
        ISSUE_TILE((k + NS - 1) * BK, (k + NS - 1) % NS);
        __pipeline_wait_prior(NS - 1);   // compile-time constant
        __syncthreads();
        MULTIPLY_TILE(k % NS);
        __syncthreads();
    }

    // ── Drain: last NS-1 tiles ────────────────────────────────────────────────
    #pragma unroll
    for (int d = NS - 2; d >= 0; d--) {
        __pipeline_wait_prior(d);        // compile-time per unrolled iteration
        __syncthreads();
        MULTIPLY_TILE((num_tiles - d - 1) % NS);
        __syncthreads();
    }

#undef ISSUE_TILE
#undef MULTIPLY_TILE

    // ── Write accumulators to C (same layout as h2_s5) ───────────────────────
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
void matmul_h2s6_bm##BM_##_bn##BN_##_bk##BK_##_wg##NG_##_ns##NS_(              \
    const __nv_bfloat16* __restrict__ A,                                         \
    const __nv_bfloat16* __restrict__ B,                                         \
    __nv_bfloat16* __restrict__ C,                                               \
    int M, int K, int N)                                                         \
{                                                                                \
    h2s6_impl<BM_, BN_, BK_, NG_, NS_>(A, B, C, M, K, N);                      \
}

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
