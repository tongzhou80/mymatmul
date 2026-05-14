#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

/*
 * H2 Stage 8: h2_s7 with cp.async replaced by TMA tile fetch.
 *
 * The ONLY change from h2_s7 is the load mechanism:
 *   - LOAD_TILE: cp.async per-thread memcpy → TMA bulk tensor load (thread 0)
 *   - WAIT_SMEM: __pipeline_wait_prior → mbarrier wait
 *
 * Everything else is identical to h2_s7:
 *   - wgmma SS variant (both A and B read from SMEM via descriptors)
 *   - SMEM layout for A and B (same shapes, same swizzle patterns; TMA hardware
 *     applies the swizzle that matches the wgmma descriptor's layout field)
 *   - Pipeline structure (WAIT → fence → __sync → COMPUTE → WAIT_MMA(1) → LOAD)
 *   - Drain, epilogue, accumulator layout
 *
 * TMA-related details:
 *   - mbarrier per slot tracks tile delivery (init + arm + wait per iter).
 *   - fence.proxy.async at COMPUTE_TILE start ensures TMA's async-proxy writes
 *     are visible to wgmma's reads of the same SMEM.
 *   - A's TMA swizzle is selected by BK to match wgmma A descriptor's layout:
 *       BK=64 → 128B,  BK=32 → 64B,  BK=16 → 32B.
 *   - B uses 128B swizzle uniformly (matches s7's manual B128 cp.async swizzle).
 */

#ifndef LB_MIN_BLOCKS
#define LB_MIN_BLOCKS 1
#endif

struct alignas(64) TmaDesc { uint64_t opaque[16]; };

// ── wgmma helpers (identical to h2_s7) ────────────────────────────────────────
__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
// fence.proxy.async: TMA writes (async proxy) → wgmma reads of SMEM
__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

// ── mbarrier helpers (one per pipeline slot, tracks TMA delivery) ────────────
__device__ __forceinline__ void mbar_init(uint64_t* mb) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                 :: "r"((uint32_t)__cvta_generic_to_shared(mb)) : "memory");
}

__device__ __forceinline__ void mbar_arm_tma(uint64_t* mb, uint32_t bytes_expected) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(mb);
    uint64_t token;
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;\n"
                 : "=l"(token) : "r"(addr), "r"(bytes_expected) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* mb) {
    uint32_t done = 0, addr = (uint32_t)__cvta_generic_to_shared(mb);
    while (!done) {
        asm volatile("{\n.reg .pred P;\n"
                     "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P, [%1], 0;\n"
                     "selp.u32 %0, 1, 0, P;\n}\n"
                     : "=r"(done) : "r"(addr) : "memory");
    }
}

// ── TMA tile load ─────────────────────────────────────────────────────────────
__device__ __forceinline__ void tma_load_tile(
    const TmaDesc* desc, void* smem_ptr, uint64_t* mbar,
    int32_t coord0, int32_t coord1)
{
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];\n"
        :: "r"(dst), "l"((unsigned long long)desc), "r"(mbar_addr),
           "r"(coord0), "r"(coord1) : "memory");
}

// ── A descriptor (identical to h2_s7) ─────────────────────────────────────────
template<int BK>
__device__ __forceinline__ uint64_t make_wgmma_a_desc(uint32_t smem_addr, int kk) {
    constexpr uint64_t layout = (BK == 64) ? 1ULL : (BK == 32) ? 2ULL : 3ULL;
    constexpr uint64_t sbo = (uint64_t)BK;
    uint64_t start = ((uint64_t)(smem_addr >> 4) & 0x3FFFULL) + (uint64_t)(kk * 2);
    return start | (sbo << 32) | (layout << 62);
}

// ── B descriptor (identical to h2_s7) ─────────────────────────────────────────
template<int BN, int BK>
__device__ __forceinline__ uint64_t make_wgmma_b_desc(uint32_t smem_addr) {
    constexpr uint64_t LAYOUT_B128 = 1ULL << 62;
    constexpr int n_atoms = BN / 64;
    constexpr uint64_t lbo = (n_atoms <= 1) ? 0ULL : (uint64_t)(8 * BK);
    constexpr uint64_t sbo = 64ULL;
    uint64_t start = (uint64_t)(smem_addr >> 4) & 0x3FFFULL;
    return start | (lbo << 16) | (sbo << 32) | LAYOUT_B128;
}

// ── wgmma SS wrappers (identical to h2_s7) ────────────────────────────────────
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

// ── Kernel ────────────────────────────────────────────────────────────────────

template<int BM, int BN, int BK, int NUM_WG, int NUM_STAGES>
__device__ __forceinline__ void h2_s7_tma_impl(
    const TmaDesc&                tma_A,
    const TmaDesc&                tma_B,
    __nv_bfloat16* __restrict__   C,
    int M, int K, int N)
{
    static_assert(BM % (NUM_WG * 64) == 0, "BM must be multiple of NUM_WG*64");
    static_assert(BN % 64 == 0, "BN must be multiple of 64");

    constexpr int NS         = NUM_STAGES;
    constexpr int M_ITERS    = BM / (NUM_WG * 64);
    constexpr int M_PER_WG   = BM / NUM_WG;
    constexpr int D          = BN / 2;
    constexpr int N_SUBTILES = BN / 64;
    constexpr int K_STEP_BYTES = 16 * 64 * 2;     // wgmma kk advance: 2048 bytes
    constexpr uint32_t TILE_BYTES = (BM * BK + BK * BN) * 2;

    const int tid        = threadIdx.x;
    const int wg_id      = tid / 128;
    const int local_warp = (tid % 128) / 32;
    const int lane       = tid % 32;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // SMEM: A[NS][BM][BK], B[NS][N_SUBTILES][BK][64], mbar[NS]
    extern __shared__ char smem_raw[];
    constexpr int A_BYTES  = NS * BM * BK * 2;
    constexpr int B_BYTES  = NS * BK * BN * 2;
    constexpr int MBAR_OFF = (A_BYTES + B_BYTES + 7) & ~7;

    auto A_sh = reinterpret_cast<__nv_bfloat16 (*)[BM][BK]>(smem_raw);
    auto B_sh = reinterpret_cast<__nv_bfloat16 (*)[N_SUBTILES][BK][64]>(smem_raw + A_BYTES);
    auto mbar = reinterpret_cast<uint64_t*>(smem_raw + MBAR_OFF);

    float acc[M_ITERS][D] = {};
    const int num_tiles = K / BK;

    // ── LOAD_TILE: thread 0 inits + arms mbarrier, then issues TMA for A and B
#define LOAD_TILE(slot_, k0_)                                                       \
    do {                                                                            \
        if (tid == 0) {                                                             \
            mbar_init(&mbar[(slot_)]);                                              \
            mbar_arm_tma(&mbar[(slot_)], TILE_BYTES);                               \
            tma_load_tile(&tma_A, &A_sh[(slot_)][0][0], &mbar[(slot_)],            \
                          (k0_), block_row);                                        \
            _Pragma("unroll")                                                       \
            for (int _i = 0; _i < N_SUBTILES; _i++) {                              \
                void* _b = (char*)&B_sh[(slot_)][0][0][0] + _i * BK * 64 * 2;      \
                tma_load_tile(&tma_B, _b, &mbar[(slot_)],                          \
                              block_col + _i * 64, (k0_));                         \
            }                                                                       \
        }                                                                           \
    } while (0)

    // ── WAIT_SMEM(slot): per-thread mbarrier wait (parity 0, set by mbar_init)
#define WAIT_SMEM(slot_) mbar_wait(&mbar[(slot_)])

    // ── COMPUTE_TILE: identical to h2_s7 (wgmma SS over BK) ──────────────────
    //   fence_proxy_async makes TMA's async-proxy writes visible to wgmma.
#define COMPUTE_TILE(slot_)                                                         \
    do {                                                                            \
        fence_proxy_async();                                                        \
        wgmma_fence();                                                              \
        _Pragma("unroll")                                                           \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                 \
            const uint32_t _bb = (uint32_t)__cvta_generic_to_shared(               \
                                     &B_sh[(slot_)][0][0][0]);                     \
            const uint64_t _db = make_wgmma_b_desc<BN, BK>(                        \
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
    } while (0)

#define WAIT_MMA(n_) \
    asm volatile("wgmma.wait_group.sync.aligned " #n_ ";\n" ::: "memory")

    // ── Prologue: NS-1 LOAD_TILEs (thread 0 issues all TMA + arms mbarriers) ─
    #pragma unroll
    for (int s = 0; s < NS - 1; s++) {
        LOAD_TILE(s, s * BK);
    }

    // ── Main loop ─────────────────────────────────────────────────────────────
    //   Identical pipeline shape to h2_s7:
    //     WAIT_SMEM(cur)     - mbarrier-wait for slot cur's TMA to complete
    //     __syncthreads      - all threads see TMA-written SMEM
    //     COMPUTE_TILE(cur)  - wgmma issues
    //     WAIT_MMA(1)        - prev wgmma drained, slot nxt safe to overwrite
    //     LOAD_TILE(nxt, ..) - next TMA load
    for (int k = 0; k < num_tiles - (NS - 1); k++) {
        const int cur = k % NS;
        const int nxt = (k + NS - 1) % NS;

        WAIT_SMEM(cur);
        __syncthreads();
        COMPUTE_TILE(cur);
        WAIT_MMA(1);
        __syncthreads();
        LOAD_TILE(nxt, (k + NS - 1) * BK);
    }

    // ── Drain ─────────────────────────────────────────────────────────────────
    #pragma unroll
    for (int d = NS - 2; d >= 0; d--) {
        const int slot = (num_tiles - d - 1) % NS;
        WAIT_SMEM(slot);
        __syncthreads();
        COMPUTE_TILE(slot);
        WAIT_MMA(1);
    }

    WAIT_MMA(0);

#undef LOAD_TILE
#undef WAIT_SMEM
#undef COMPUTE_TILE
#undef WAIT_MMA

    // ── Epilogue (identical to h2_s7) ─────────────────────────────────────────
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
void matmul_h2_s7_tma_bm##BM_##_bn##BN_##_bk##BK_##_wg##NG_##_ns##NS_(              \
    const __grid_constant__ TmaDesc tma_A,                                       \
    const __grid_constant__ TmaDesc tma_B,                                       \
    __nv_bfloat16* __restrict__     C,                                           \
    int M, int K, int N)                                                         \
{                                                                                \
    h2_s7_tma_impl<BM_, BN_, BK_, NG_, NS_>(tma_A, tma_B, C, M, K, N);              \
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
