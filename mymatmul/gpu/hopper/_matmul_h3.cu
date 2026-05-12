#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>

/*
 * H3: h1_ms with mma.sync replaced by wgmma.
 *
 * Changes vs h1_ms (_matmul_h1_ms.cu):
 *   1. Thread block: (32, NW, 1) → (NW*32, 1, 1) for clean warpgroup layout.
 *      BM = (NUM_WARPS/4) * 64  (each warpgroup covers 64 rows).
 *   2. ISSUE_TILE B: XOR swizzle removed — B written linearly via cp.async.
 *      (The XOR was for ldmatrix.x2.trans which is gone; wgmma uses an SMEM
 *       descriptor and its own hardware access pattern.)
 *   3. ISSUE_TILE A: unchanged — same XOR swizzle, same cp.async.
 *   4. COMPUTE_TILE: ldmatrix_x2_trans + mma.sync → wgmma.mma_async.
 *      A stays in registers (ldmatrix_x4 with XOR, unchanged).
 *      B is consumed directly from SMEM via a GmmaDescriptor.
 *      Fencing: fence.proxy.async + wgmma.fence before; commit_group +
 *      wait_group after the kk loop.
 *   5. Accumulator: [WM][WN][4] per warp → float acc[BN/2] per thread.
 *   6. Epilogue: warpgroup-aware row/col mapping (same as h2_s3).
 *
 * Multi-stage pipeline (NUM_STAGES) is IDENTICAL to h1_ms.
 * Compiled with -DLB_MIN_BLOCKS=N (same LB tuning).
 */

#ifndef LB_MIN_BLOCKS
#define LB_MIN_BLOCKS 1
#endif

// ── PTX: ldmatrix for A (unchanged from h1_ms / tc5_lb) ──────────────────────

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_ptr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_ptr));
}

// ── wgmma B SMEM descriptor (B128 / 128B-swizzle, MN-major) ─────────────────
//
// B is written via cp.async with row%8 XOR (= same physical layout as TMA 128B).
// wgmma reads it via B128 descriptor (layout_type_=1, bits 63:62 = 01).
// This is empirically verified to work; plain INTERLEAVE (layout_type=0) crashes.
//
// B is organised as BN/64 packed sub-tiles of [BK][64] BF16, back-to-back.
// kk-step advancement: base + kk * 16 * 64 * 2 bytes (= 2048 per step).
// Descriptor formula identical to h2_s3 (same B layout, different write path).

template<int BN, int BK>
__device__ __forceinline__ uint64_t make_h3_b_desc(uint32_t smem_addr) {
    constexpr uint64_t LAYOUT_B128 = 1ULL << 62;
    constexpr int n_atoms = BN / 64;
    constexpr uint64_t lbo = (n_atoms <= 1) ? 0ULL : (uint64_t)(8 * BK);
    constexpr uint64_t sbo = 64;
    uint64_t start = (uint64_t)(smem_addr >> 4) & 0x3FFF;
    return start | (lbo << 16) | (sbo << 32) | LAYOUT_B128;
}

// ── wgmma.mma_async wrappers (transB=1, scaleD=1, accumulate) ────────────────

__device__ __forceinline__
void wgmma_m64n64k16(float d[32], uint32_t a[4], uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "{%32,%33,%34,%35},%36,1,1,1,1;\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),
         "+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),
         "+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),
         "+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),
         "+f"(d[30]),"+f"(d[31])
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
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),
         "+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),
         "+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),
         "+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),
         "+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),
         "+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),
         "+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
         "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),
         "+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),
         "+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
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
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),
         "+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),
         "+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),
         "+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),
         "+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),
         "+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),
         "+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
         "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),
         "+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),
         "+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),"+f"(d[64]),"+f"(d[65]),
         "+f"(d[66]),"+f"(d[67]),"+f"(d[68]),"+f"(d[69]),"+f"(d[70]),"+f"(d[71]),
         "+f"(d[72]),"+f"(d[73]),"+f"(d[74]),"+f"(d[75]),"+f"(d[76]),"+f"(d[77]),
         "+f"(d[78]),"+f"(d[79]),"+f"(d[80]),"+f"(d[81]),"+f"(d[82]),"+f"(d[83]),
         "+f"(d[84]),"+f"(d[85]),"+f"(d[86]),"+f"(d[87]),"+f"(d[88]),"+f"(d[89]),
         "+f"(d[90]),"+f"(d[91]),"+f"(d[92]),"+f"(d[93]),"+f"(d[94]),"+f"(d[95]),
         "+f"(d[96]),"+f"(d[97]),"+f"(d[98]),"+f"(d[99]),"+f"(d[100]),"+f"(d[101]),
         "+f"(d[102]),"+f"(d[103]),"+f"(d[104]),"+f"(d[105]),"+f"(d[106]),"+f"(d[107]),
         "+f"(d[108]),"+f"(d[109]),"+f"(d[110]),"+f"(d[111]),"+f"(d[112]),"+f"(d[113]),
         "+f"(d[114]),"+f"(d[115]),"+f"(d[116]),"+f"(d[117]),"+f"(d[118]),"+f"(d[119]),
         "+f"(d[120]),"+f"(d[121]),"+f"(d[122]),"+f"(d[123]),"+f"(d[124]),"+f"(d[125]),
         "+f"(d[126]),"+f"(d[127])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(b));
}

template<int BN>
__device__ __forceinline__
void wgmma_call(float* acc, uint32_t a[4], uint64_t desc_b) {
    if constexpr      (BN ==  64) wgmma_m64n64k16 (acc, a, desc_b);
    else if constexpr (BN == 128) wgmma_m64n128k16(acc, a, desc_b);
    else if constexpr (BN == 256) wgmma_m64n256k16(acc, a, desc_b);
}

// ── Kernel implementation ─────────────────────────────────────────────────────

template <int BN, int BK, int NUM_WARPS, int NUM_STAGES>
__device__ __forceinline__ void h3_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    constexpr int NS  = NUM_STAGES;
    constexpr int BM  = (NUM_WARPS / 4) * 64;  // each warpgroup covers 64 rows
    constexpr int D   = BN / 2;                 // f32 accumulators per thread

    constexpr int THREADS  = NUM_WARPS * 32;

    // A load parameters (same as h1_ms)
    constexpr int A_ELEM   = (BM * BK / THREADS >= 8) ? 8 : 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int A_SWZ    = BK / 8;
    constexpr int A_SHIFT  = 64 / BK;

    // B load parameters — no swizzle
    constexpr int B_ELEM   = (BK * BN / THREADS >= 8) ? 8 : 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    // 1D thread block: warpgroups are groups of 4 consecutive warps
    const int tid        = threadIdx.x;
    const int wg_id      = tid / 128;
    const int local_warp = (tid % 128) / 32;
    const int lane       = tid % 32;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // SMEM: A[NS][BM][BK] with XOR swizzle; B[NS][BK][BN] linear
    extern __shared__ __nv_bfloat16 smem[];
    auto A_shared = reinterpret_cast<__nv_bfloat16 (*)[BM][BK]>(smem);
    auto B_shared = reinterpret_cast<__nv_bfloat16 (*)[BK][BN]>(smem + NS * BM * BK);

    float acc[D] = {};

    // ── ISSUE_TILE ────────────────────────────────────────────────────────────
    // A: XOR swizzle unchanged from h1_ms.
    // B: linear write (XOR removed — wgmma reads via descriptor, not ldmatrix).

#define ISSUE_TILE(k0_, buf_)                                                       \
    do {                                                                            \
        /* A: XOR swizzle */                                                        \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                    \
            const int _g  = tid + _i * THREADS;                                    \
            const int _r  = (_g * A_ELEM) / BK;                                   \
            const int _c  = (_g * A_ELEM) % BK;                                   \
            const int _sc = ((_c/8) ^ ((_r/A_SHIFT) % A_SWZ)) * 8 + (_c%8);      \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_sc],                     \
                &A[(block_row+_r)*K+(k0_)+_c], A_ELEM*(int)sizeof(__nv_bfloat16)); \
        }                                                                           \
        /* B: row%8 XOR in sub-tile format (matches TMA 128B / B128 descriptor).   \
         * BN/64 sub-tiles of [BK][64] packed back-to-back.                       \
         * Sub-tile st, row _r, local col _c:                                      \
         *   physical col = (_c/8 XOR _r%8)*8 + _c%8                              \
         *   SMEM offset from B base = st*BK*64*2 + _r*64*2 + phys_c*2           \
         * For BN=64 (one sub-tile): identical to h1_ms B write with B_SWZ=8.    */ \
        {                                                                           \
        constexpr int STCOL = 64;         /* cols per sub-tile */                  \
        constexpr int ST_ELEMS = BK * STCOL;                                       \
        constexpr int ST_GROUPS = ST_ELEMS / B_ELEM / THREADS;                     \
        _Pragma("unroll")                                                           \
        for (int _st = 0; _st < BN / STCOL; _st++) {                               \
            _Pragma("unroll")                                                       \
            for (int _i = 0; _i < ST_GROUPS; _i++) {                               \
                const int _g  = tid + _i * THREADS;                                \
                const int _r  = (_g * B_ELEM) / STCOL;                            \
                const int _c  = (_g * B_ELEM) % STCOL;  /* col within sub-tile */ \
                const int _sc = ((_c/8) ^ (_r%8)) * 8 + (_c%8);  /* row%8 XOR */ \
                /* Destination: sub-tile _st at B_shared base + offset */          \
                __nv_bfloat16* _dst = reinterpret_cast<__nv_bfloat16*>(            \
                    reinterpret_cast<char*>(&B_shared[(buf_)][0][0])               \
                    + ((size_t)_st * BK * STCOL + _r * STCOL + _sc)               \
                      * sizeof(__nv_bfloat16));                                    \
                __pipeline_memcpy_async(_dst,                                      \
                    &B[((k0_)+_r)*N + block_col + _st*STCOL + _c],                \
                    B_ELEM*(int)sizeof(__nv_bfloat16));                            \
            }                                                                       \
        }                                                                           \
        }                                                                           \
        __pipeline_commit();                                                         \
    } while (0)

    // ── COMPUTE_TILE ──────────────────────────────────────────────────────────
    // A: same ldmatrix_x4 with XOR swizzle, but now using warpgroup row indices.
    // B: consumed directly from SMEM via GmmaDescriptor (no ldmatrix).
    // Fencing wraps the entire kk loop.

#define COMPUTE_TILE(buf_)                                                          \
    do {                                                                            \
        /* make SMEM visible to async proxy, then fence wgmma register state */     \
        asm volatile("fence.proxy.async;\n" ::: "memory");                         \
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");                  \
        _Pragma("unroll")                                                           \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                 \
            /* Load A fragment: each thread gets 4 uint32 for this warpgroup's     \
             * 64-row × 16-col A slice, with XOR swizzle. */                       \
            uint32_t _fa[4];                                                        \
            const int _ar   = wg_id * 64 + local_warp * 16 + (lane % 16);         \
            const int _alg  = _kk * 2 + (lane / 16);                               \
            const int _aph  = _alg ^ ((_ar / A_SHIFT) % A_SWZ);                   \
            ldmatrix_x4(_fa[0], _fa[1], _fa[2], _fa[3],                            \
                __cvta_generic_to_shared(&A_shared[(buf_)][_ar][_aph * 8]));       \
            /* B descriptor: sub-tile 0 base + kk × 2048 bytes (= 16 K-rows ×     \
             * 64 BF16-wide × 2 bytes). Same advancement formula as h2_s3.       */ \
            constexpr int K_STEP_BYTES = 16 * 64 * 2;                              \
            uint32_t _b_base = (uint32_t)__cvta_generic_to_shared(                 \
                &B_shared[(buf_)][0][0]);                                           \
            uint64_t _db = make_h3_b_desc<BN, BK>(_b_base + _kk * K_STEP_BYTES); \
            wgmma_call<BN>(acc, _fa, _db);                                         \
        }                                                                           \
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");           \
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");           \
    } while (0)

    // ── Multi-stage pipeline (identical structure to h1_ms) ───────────────────

    const int num_tiles = K / BK;

    // Prologue: fill pipeline with first NS-1 tiles
    #pragma unroll
    for (int s = 0; s < NS - 1; s++) {
        ISSUE_TILE(s * BK, s);
    }

    // Main loop: tiles 0..num_tiles-NS (always issues a new tile)
    for (int k = 0; k < num_tiles - (NS - 1); k++) {
        ISSUE_TILE((k + NS - 1) * BK, (k + NS - 1) % NS);
        __pipeline_wait_prior(NS - 1);
        __syncthreads();
        COMPUTE_TILE(k % NS);
        __syncthreads();
    }

    // Drain: last NS-1 tiles
    #pragma unroll
    for (int d = NS - 2; d >= 0; d--) {
        __pipeline_wait_prior(d);
        __syncthreads();
        COMPUTE_TILE((num_tiles - d - 1) % NS);
        __syncthreads();
    }

#undef ISSUE_TILE
#undef COMPUTE_TILE

    // ── Epilogue: warpgroup output layout (same as h2_s3) ────────────────────
    // Thread (wg=wg_id, local_warp=lw, lane=l) owns for each j in 0..BN/8-1:
    //   acc[j*4+0,1] → row (wg_id*64 + lw*16 + l/4),     col (j*8 + (l%4)*2)
    //   acc[j*4+2,3] → row (wg_id*64 + lw*16 + l/4 + 8), col (j*8 + (l%4)*2)
    const int base_col = (lane % 4) * 2;
    const int base_row = lane / 4;
    #pragma unroll
    for (int j = 0; j < BN / 8; j++) {
        const int gc  = block_col + j * 8 + base_col;
        const int gr0 = block_row + wg_id * 64 + local_warp * 16 + base_row;
        const int gr8 = gr0 + 8;
        if (gr0 < M && gc < N)
            *reinterpret_cast<__nv_bfloat162*>(&C[gr0 * N + gc]) =
                __floats2bfloat162_rn(acc[j*4+0], acc[j*4+1]);
        if (gr8 < M && gc < N)
            *reinterpret_cast<__nv_bfloat162*>(&C[gr8 * N + gc]) =
                __floats2bfloat162_rn(acc[j*4+2], acc[j*4+3]);
    }
}

// ── Kernel entry points ───────────────────────────────────────────────────────

#define MAKE_LAUNCHER(BN_, BK_, NW_, NS_)                                          \
extern "C" __global__ __launch_bounds__(NW_ * 32, LB_MIN_BLOCKS)                  \
void matmul_h3_bn##BN_##_bk##BK_##_nw##NW_##_ns##NS_(                             \
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,      \
    __nv_bfloat16* __restrict__ C, int M, int K, int N)                            \
{                                                                                   \
    h3_impl<BN_, BK_, NW_, NS_>(A, B, C, M, K, N);                                \
}

// Single cubin, LB_MIN_BLOCKS=1 (same model as h2_s2/h2_s3).
// Full register budget → BN=256 included; no per-LB compilation needed.
MAKE_LAUNCHER( 64, 16, 4, 2) MAKE_LAUNCHER( 64, 32, 4, 2) MAKE_LAUNCHER( 64, 64, 4, 2)
MAKE_LAUNCHER(128, 16, 4, 2) MAKE_LAUNCHER(128, 32, 4, 2) MAKE_LAUNCHER(128, 64, 4, 2)
MAKE_LAUNCHER(256, 16, 4, 2) MAKE_LAUNCHER(256, 32, 4, 2) MAKE_LAUNCHER(256, 64, 4, 2)

MAKE_LAUNCHER( 64, 16, 4, 3) MAKE_LAUNCHER( 64, 32, 4, 3) MAKE_LAUNCHER( 64, 64, 4, 3)
MAKE_LAUNCHER(128, 16, 4, 3) MAKE_LAUNCHER(128, 32, 4, 3) MAKE_LAUNCHER(128, 64, 4, 3)
MAKE_LAUNCHER(256, 16, 4, 3) MAKE_LAUNCHER(256, 32, 4, 3) MAKE_LAUNCHER(256, 64, 4, 3)

MAKE_LAUNCHER( 64, 16, 4, 4) MAKE_LAUNCHER( 64, 32, 4, 4) MAKE_LAUNCHER( 64, 64, 4, 4)
MAKE_LAUNCHER(128, 16, 4, 4) MAKE_LAUNCHER(128, 32, 4, 4) MAKE_LAUNCHER(128, 64, 4, 4)
MAKE_LAUNCHER(256, 16, 4, 4) MAKE_LAUNCHER(256, 32, 4, 4) MAKE_LAUNCHER(256, 64, 4, 4)

MAKE_LAUNCHER( 64, 16, 4, 5) MAKE_LAUNCHER( 64, 32, 4, 5) MAKE_LAUNCHER( 64, 64, 4, 5)
MAKE_LAUNCHER(128, 16, 4, 5) MAKE_LAUNCHER(128, 32, 4, 5) MAKE_LAUNCHER(128, 64, 4, 5)
MAKE_LAUNCHER(256, 16, 4, 5) MAKE_LAUNCHER(256, 32, 4, 5) MAKE_LAUNCHER(256, 64, 4, 5)

#undef MAKE_LAUNCHER
