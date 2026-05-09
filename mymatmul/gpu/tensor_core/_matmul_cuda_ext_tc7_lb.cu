#ifndef LB_MIN_BLOCKS
#define LB_MIN_BLOCKS 2
#endif

#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>

/*
 * TC7: TC6 with A-tile and B-tile async copies committed separately.
 *
 * TC6 issues A+B together under one __pipeline_commit() and waits for
 * both before any ldmatrix begins.
 *
 * TC7 commits them independently:
 *   ISSUE_A_TILE → __pipeline_commit()   (group 2k)
 *   ISSUE_B_TILE → __pipeline_commit()   (group 2k+1)
 *
 * Then the compute sequence per tile is:
 *   wait_prior(3) + sync  → A[cur] in smem, ldmatrix all A frags
 *   wait_prior(2) + sync  → B[cur] in smem, ldmatrix all B frags
 *   sync                  → smem[cur] released, MMA from registers
 *
 * wait_prior(3/2) are constant every iteration because we always issue
 * exactly two new commits before waiting (so the oldest pending group
 * is always 3/2 back from the most recent).
 *
 * Fragment arrays are lifted to function scope so LOAD_A / LOAD_B /
 * RUN_MMA macros can all reference them without stacking do-while blocks.
 */

// ── PTX helpers ──────────────────────────────────────────────────────────────

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    uint32_t smem_ptr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& r0, uint32_t& r1,
    uint32_t smem_ptr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

// ── Kernel implementation ─────────────────────────────────────────────────────

template <int BM, int BN, int BK, int NUM_WARPS>
__device__ __forceinline__ void matmul_tc7_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    constexpr int WARP_N = 2;
    constexpr int WARP_M = NUM_WARPS / WARP_N;

    constexpr int WARP_TILE_M = BM / WARP_M;
    constexpr int WARP_TILE_N = BN / WARP_N;

    constexpr int WM_TILES = WARP_TILE_M / 16;
    constexpr int WN_TILES = WARP_TILE_N / 8;

    constexpr int THREADS = NUM_WARPS * 32;

    constexpr int A_ELEM   = (BM * BK / THREADS >= 8) ? 8 : 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = (BK * BN / THREADS >= 8) ? 8 : 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    constexpr int A_SWZ   = BK / 8;
    constexpr int A_SHIFT = 64 / BK;

    constexpr int B_SWZ = BN / 8;

    extern __shared__ __nv_bfloat16 smem[];
    auto A_shared = reinterpret_cast<__nv_bfloat16 (*)[BM][BK]>(smem);
    auto B_shared = reinterpret_cast<__nv_bfloat16 (*)[BK][BN]>(smem + 2 * BM * BK);

    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid % 32;
    const int warp_row = warp_id / WARP_N;
    const int warp_col = warp_id % WARP_N;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[WM_TILES][WN_TILES][4] = {};

    // Fragment arrays at function scope: written by LOAD_A/LOAD_B, read by RUN_MMA.
    uint32_t fa[BK / 16][WM_TILES][4];
    uint32_t fb[BK / 16][WN_TILES][2];

// ── Async issue macros (one commit each) ─────────────────────────────────────

#define ISSUE_A_TILE(k0_, buf_)                                                     \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                    \
            const int _g  = tid + _i * THREADS;                                    \
            const int _r  = (_g * A_ELEM) / BK;                                   \
            const int _c  = (_g * A_ELEM) % BK;                                   \
            const int _sc = ((_c / 8) ^ ((_r / A_SHIFT) % A_SWZ)) * 8 + (_c % 8); \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_sc],                     \
                                    &A[(block_row + _r) * K + (k0_) + _c],        \
                                    A_ELEM * (int)sizeof(__nv_bfloat16));           \
        }                                                                           \
        __pipeline_commit();                                                         \
    } while (0)

#define ISSUE_B_TILE(k0_, buf_)                                                     \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                    \
            const int _g  = tid + _i * THREADS;                                    \
            const int _r  = (_g * B_ELEM) / BN;                                   \
            const int _c  = (_g * B_ELEM) % BN;                                   \
            const int _sc = ((_c / 8) ^ (_r % B_SWZ)) * 8 + (_c % 8);            \
            __pipeline_memcpy_async(&B_shared[(buf_)][_r][_sc],                    \
                                    &B[((k0_) + _r) * N + block_col + _c],        \
                                    B_ELEM * (int)sizeof(__nv_bfloat16));           \
        }                                                                           \
        __pipeline_commit();                                                         \
    } while (0)

// ── Three compute passes (share fa/fb from function scope) ───────────────────

#define LOAD_A_FRAGS(buf_)                                                          \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                 \
            _Pragma("unroll")                                                       \
            for (int _mt = 0; _mt < WM_TILES; _mt++) {                             \
                const int _ar   = warp_row * WARP_TILE_M + _mt * 16 + (lane % 16);\
                const int _lg   = _kk * 2 + (lane / 16);                           \
                const int _phys = _lg ^ ((_ar / A_SHIFT) % A_SWZ);                \
                ldmatrix_x4(fa[_kk][_mt][0], fa[_kk][_mt][1],                     \
                            fa[_kk][_mt][2], fa[_kk][_mt][3],                     \
                    __cvta_generic_to_shared(&A_shared[(buf_)][_ar][_phys * 8]));  \
            }                                                                       \
        }                                                                           \
    } while (0)

#define LOAD_B_FRAGS(buf_)                                                          \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                 \
            _Pragma("unroll")                                                       \
            for (int _nt = 0; _nt < WN_TILES; _nt++) {                            \
                const int _br = _kk * 16 + (lane % 16);                            \
                const int _nc = warp_col * WARP_TILE_N + _nt * 8;                  \
                const int _sc = ((_nc / 8) ^ (_br % B_SWZ)) * 8;                  \
                ldmatrix_x2_trans(fb[_kk][_nt][0], fb[_kk][_nt][1],               \
                    __cvta_generic_to_shared(&B_shared[(buf_)][_br][_sc]));         \
            }                                                                       \
        }                                                                           \
    } while (0)

#define RUN_MMA()                                                                   \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                 \
            _Pragma("unroll")                                                       \
            for (int _mt = 0; _mt < WM_TILES; _mt++) {                             \
                _Pragma("unroll")                                                   \
                for (int _nt = 0; _nt < WN_TILES; _nt++) {                         \
                    mma_m16n8k16(acc[_mt][_nt][0], acc[_mt][_nt][1],               \
                                 acc[_mt][_nt][2], acc[_mt][_nt][3],               \
                                 fa[_kk][_mt][0], fa[_kk][_mt][1],                \
                                 fa[_kk][_mt][2], fa[_kk][_mt][3],                \
                                 fb[_kk][_nt][0], fb[_kk][_nt][1]);               \
                }                                                                   \
            }                                                                       \
        }                                                                           \
    } while (0)

// ── Main pipeline loop ────────────────────────────────────────────────────────

    const int num_tiles = K / BK;

    ISSUE_A_TILE(0, 0);          // group 0: A[0]
    ISSUE_B_TILE(0, 0);          // group 1: B[0]

    for (int k_iter = 0; k_iter < num_tiles - 1; k_iter++) {
        const int cur = k_iter & 1;
        const int nxt = 1 - cur;

        ISSUE_A_TILE((k_iter + 1) * BK, nxt);   // group 2*(k+1)  : A[k+1]
        ISSUE_B_TILE((k_iter + 1) * BK, nxt);   // group 2*(k+1)+1: B[k+1]

        // A[cur] is 3 back from the most-recent commit → wait_prior(3)
        __pipeline_wait_prior(3);
        __syncthreads();
        LOAD_A_FRAGS(cur);

        // B[cur] is 2 back from the most-recent commit → wait_prior(2)
        __pipeline_wait_prior(2);
        __syncthreads();
        LOAD_B_FRAGS(cur);

        __syncthreads();    // smem[cur] released; next iter may overwrite it
        RUN_MMA();
    }

    // Final tile: only A[last] and B[last] remain pending (1 and 0 back)
    __pipeline_wait_prior(1);
    __syncthreads();
    LOAD_A_FRAGS((num_tiles - 1) & 1);

    __pipeline_wait_prior(0);
    __syncthreads();
    LOAD_B_FRAGS((num_tiles - 1) & 1);
    RUN_MMA();

#undef ISSUE_A_TILE
#undef ISSUE_B_TILE
#undef LOAD_A_FRAGS
#undef LOAD_B_FRAGS
#undef RUN_MMA

// ── Vectorized write-back ─────────────────────────────────────────────────────

    const int base_row = lane / 4;
    const int base_col = (lane % 4) * 2;

    #pragma unroll
    for (int mt = 0; mt < WM_TILES; mt++) {
        #pragma unroll
        for (int nt = 0; nt < WN_TILES; nt++) {
            const int gc  = block_col + warp_col * WARP_TILE_N + nt * 8 + base_col;
            const int gr0 = block_row + warp_row * WARP_TILE_M + mt * 16 + base_row;
            if (gr0 < M && gc < N)
                *reinterpret_cast<__nv_bfloat162*>(&C[gr0 * N + gc]) =
                    __floats2bfloat162_rn(acc[mt][nt][0], acc[mt][nt][1]);
            const int gr8 = gr0 + 8;
            if (gr8 < M && gc < N)
                *reinterpret_cast<__nv_bfloat162*>(&C[gr8 * N + gc]) =
                    __floats2bfloat162_rn(acc[mt][nt][2], acc[mt][nt][3]);
        }
    }
}

#define MAKE_LAUNCHER(BM_, BN_, BK_, NW_)                                           \
extern "C" __global__ __launch_bounds__(NW_ * 32, LB_MIN_BLOCKS)                                   \
void matmul_cuda_tc7_bm##BM_##_bn##BN_##_bk##BK_##_nw##NW_(                        \
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,       \
    __nv_bfloat16* __restrict__ C, int M, int K, int N) {                           \
    matmul_tc7_impl<BM_, BN_, BK_, NW_>(A, B, C, M, K, N);                         \
}

// ── NW=4 ─────────────────────────────────────────────────────────────────────
MAKE_LAUNCHER( 64,  64, 16, 4) MAKE_LAUNCHER( 64,  64, 32, 4) MAKE_LAUNCHER( 64,  64, 64, 4)
MAKE_LAUNCHER( 64, 128, 16, 4) MAKE_LAUNCHER( 64, 128, 32, 4) MAKE_LAUNCHER( 64, 128, 64, 4)
MAKE_LAUNCHER( 64, 256, 16, 4) MAKE_LAUNCHER( 64, 256, 32, 4) MAKE_LAUNCHER( 64, 256, 64, 4)
MAKE_LAUNCHER(128,  64, 16, 4) MAKE_LAUNCHER(128,  64, 32, 4) MAKE_LAUNCHER(128,  64, 64, 4)
MAKE_LAUNCHER(128, 128, 16, 4) MAKE_LAUNCHER(128, 128, 32, 4) MAKE_LAUNCHER(128, 128, 64, 4)
MAKE_LAUNCHER(256,  64, 16, 4) MAKE_LAUNCHER(256,  64, 32, 4) MAKE_LAUNCHER(256,  64, 64, 4)

// ── NW=8 ─────────────────────────────────────────────────────────────────────
MAKE_LAUNCHER( 64,  64, 16, 8) MAKE_LAUNCHER( 64,  64, 32, 8) MAKE_LAUNCHER( 64,  64, 64, 8)
MAKE_LAUNCHER( 64, 128, 16, 8) MAKE_LAUNCHER( 64, 128, 32, 8) MAKE_LAUNCHER( 64, 128, 64, 8)
MAKE_LAUNCHER( 64, 256, 16, 8) MAKE_LAUNCHER( 64, 256, 32, 8) MAKE_LAUNCHER( 64, 256, 64, 8)
MAKE_LAUNCHER(128,  64, 16, 8) MAKE_LAUNCHER(128,  64, 32, 8) MAKE_LAUNCHER(128,  64, 64, 8)
MAKE_LAUNCHER(128, 128, 16, 8) MAKE_LAUNCHER(128, 128, 32, 8) MAKE_LAUNCHER(128, 128, 64, 8)
MAKE_LAUNCHER(128, 256, 16, 8) MAKE_LAUNCHER(128, 256, 32, 8) MAKE_LAUNCHER(128, 256, 64, 8)
MAKE_LAUNCHER(256,  64, 16, 8) MAKE_LAUNCHER(256,  64, 32, 8) MAKE_LAUNCHER(256,  64, 64, 8)
MAKE_LAUNCHER(256, 128, 16, 8) MAKE_LAUNCHER(256, 128, 32, 8) MAKE_LAUNCHER(256, 128, 64, 8)

#undef MAKE_LAUNCHER
