#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

/*
 * TC3: BF16 WMMA matmul with tunable A-tile and B-tile smem padding.
 *
 * Template parameters PAD_A and PAD_B pad the A-tile column stride (BKP=BK+PAD_A)
 * and B-tile column stride (BNP=BN+PAD_B).  When both are 0 the code is identical
 * to TC1.  Only PAD ∈ {0, 8} is used — PAD=8 keeps BKP/BNP multiples of 8,
 * preserving 16-byte cp.async alignment.  PAD=4 (not instantiated here) would
 * require 8-byte copies which proved slower despite giving zero conflict.
 *
 * Bank-conflict analysis (32 banks, 4 bytes each, BF16 = 2 bytes):
 *   shift per row = (ldm / 2) % 32
 *   cycle length  = 32 / gcd(shift, 32)   ← distinct banks before wrap
 *
 *   PAD=0 → shift=0  (power-of-2 BK/BN) → cycle=1  → 16-way conflict
 *   PAD=8 → shift=4  (BK∈{16,32})        → cycle=8  →  2-way conflict
 *
 * Both PAD_A and PAD_B can be tuned independently; the autotuner finds
 * the best combination for each (M,N,K).
 *
 * Smem: (2*BM*BKP + 2*BK*BNP) * 2 bytes  (double-buffered, BF16).
 */

template <int BM, int BN, int BK, int NUM_WARPS, int PAD_A, int PAD_B>
__device__ __forceinline__ void matmul_tc3_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    constexpr int BKP = BK + PAD_A;
    constexpr int BNP = BN + PAD_B;

    constexpr int WARP_N = 2;
    constexpr int WARP_M = NUM_WARPS / WARP_N;

    constexpr int WARP_TILE_M = BM / WARP_M;
    constexpr int WARP_TILE_N = BN / WARP_N;

    constexpr int WM_TILES = WARP_TILE_M / 16;
    constexpr int WN_TILES = WARP_TILE_N / 16;

    constexpr int THREADS = NUM_WARPS * 32;

    // BKP and BNP are multiples of 8 for PAD∈{0,8}, so 16-byte cp.async is valid.
    constexpr int A_ELEM   = (BM * BK / THREADS >= 8) ? 8 : 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = (BK * BN / THREADS >= 8) ? 8 : 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    extern __shared__ __nv_bfloat16 smem[];
    auto A_shared = reinterpret_cast<__nv_bfloat16 (*)[BM][BKP]>(smem);
    auto B_shared = reinterpret_cast<__nv_bfloat16 (*)[BK][BNP]>(smem + 2 * BM * BKP);

    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid % 32;
    const int warp_row = warp_id / WARP_N;
    const int warp_col = warp_id % WARP_N;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[WM_TILES][WN_TILES];
    #pragma unroll
    for (int mt = 0; mt < WM_TILES; mt++)
        #pragma unroll
        for (int nt = 0; nt < WN_TILES; nt++)
            wmma::fill_fragment(acc_frag[mt][nt], 0.0f);

// Issue one BM×BK A-tile and BK×BN B-tile into smem[buf_] via cp.async.
// Index arithmetic uses logical BK/BN widths; padded array types handle stride.
#define ISSUE_TILE(k0_, buf_)                                                       \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                    \
            const int _g = tid + _i * THREADS;                                     \
            const int _r = (_g * A_ELEM) / BK, _c = (_g * A_ELEM) % BK;          \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_c],                     \
                                    &A[(block_row + _r) * K + (k0_) + _c],        \
                                    A_ELEM * (int)sizeof(__nv_bfloat16));           \
        }                                                                           \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                    \
            const int _g = tid + _i * THREADS;                                     \
            const int _r = (_g * B_ELEM) / BN, _c = (_g * B_ELEM) % BN;          \
            __pipeline_memcpy_async(&B_shared[(buf_)][_r][_c],                     \
                                    &B[((k0_) + _r) * N + block_col + _c],        \
                                    B_ELEM * (int)sizeof(__nv_bfloat16));           \
        }                                                                           \
        __pipeline_commit();                                                         \
    } while (0)

// WMMA compute over BK/16 k-steps; ldm for A is BKP, for B is BNP.
#define COMPUTE_TILE(buf_)                                                          \
    do {                                                                            \
        wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> _fa; \
        wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::row_major> _fb; \
        _Pragma("unroll")                                                           \
        for (int _mt = 0; _mt < WM_TILES; _mt++) {                                 \
            _Pragma("unroll")                                                       \
            for (int _kk = 0; _kk < BK / 16; _kk++) {                             \
                wmma::load_matrix_sync(_fa,                                         \
                    &A_shared[(buf_)][warp_row * WARP_TILE_M + _mt * 16][_kk * 16], BKP); \
                _Pragma("unroll")                                                   \
                for (int _nt = 0; _nt < WN_TILES; _nt++) {                         \
                    wmma::load_matrix_sync(_fb,                                     \
                        &B_shared[(buf_)][_kk * 16][warp_col * WARP_TILE_N + _nt * 16], BNP); \
                    wmma::mma_sync(acc_frag[_mt][_nt], _fa, _fb, acc_frag[_mt][_nt]); \
                }                                                                   \
            }                                                                       \
        }                                                                           \
    } while (0)

    const int num_tiles = K / BK;

    ISSUE_TILE(0, 0);

    for (int k_iter = 0; k_iter < num_tiles - 1; k_iter++) {
        const int cur = k_iter & 1;
        const int nxt = 1 - cur;
        ISSUE_TILE((k_iter + 1) * BK, nxt);
        __pipeline_wait_prior(1);
        __syncthreads();
        COMPUTE_TILE(cur);
        __syncthreads();
    }

    __pipeline_wait_prior(0);
    __syncthreads();
    COMPUTE_TILE((num_tiles - 1) & 1);

#undef ISSUE_TILE
#undef COMPUTE_TILE

    constexpr int row_off[8] = {0, 0, 8, 8, 0, 0, 8, 8};
    constexpr int col_off[8] = {0, 1, 0, 1, 8, 9, 8, 9};
    const int base_row = lane / 4;
    const int base_col = (lane % 4) * 2;

    #pragma unroll
    for (int mt = 0; mt < WM_TILES; mt++) {
        #pragma unroll
        for (int nt = 0; nt < WN_TILES; nt++) {
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const int gr = block_row + warp_row * WARP_TILE_M + mt * 16
                               + base_row + row_off[e];
                const int gc = block_col + warp_col * WARP_TILE_N + nt * 16
                               + base_col + col_off[e];
                if (gr < M && gc < N)
                    C[gr * N + gc] = __float2bfloat16(acc_frag[mt][nt].x[e]);
            }
        }
    }
}

#define MAKE_LAUNCHER(BM_, BN_, BK_, NW_, PA_, PB_)                                 \
extern "C" __global__ __launch_bounds__(NW_ * 32)                                    \
void matmul_cuda_tc3_bm##BM_##_bn##BN_##_bk##BK_##_nw##NW_##_pa##PA_##_pb##PB_(    \
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,        \
    __nv_bfloat16* __restrict__ C, int M, int K, int N) {                            \
    matmul_tc3_impl<BM_, BN_, BK_, NW_, PA_, PB_>(A, B, C, M, K, N);               \
}

// Expand all 4 (PAD_A, PAD_B) ∈ {0,8}² combos for a given (BM, BN, BK, NW).
#define MAKE_PADS(BM_, BN_, BK_, NW_)          \
    MAKE_LAUNCHER(BM_, BN_, BK_, NW_, 0, 0)    \
    MAKE_LAUNCHER(BM_, BN_, BK_, NW_, 0, 8)    \
    MAKE_LAUNCHER(BM_, BN_, BK_, NW_, 8, 0)    \
    MAKE_LAUNCHER(BM_, BN_, BK_, NW_, 8, 8)

// ── NW=4 (128 threads) ───────────────────────────────────────────────────────
MAKE_PADS( 64,  64, 16, 4) MAKE_PADS( 64,  64, 32, 4)
MAKE_PADS( 64, 128, 16, 4) MAKE_PADS( 64, 128, 32, 4)
MAKE_PADS( 64, 256, 16, 4) MAKE_PADS( 64, 256, 32, 4)
MAKE_PADS(128,  64, 16, 4) MAKE_PADS(128,  64, 32, 4)
MAKE_PADS(128, 128, 16, 4) MAKE_PADS(128, 128, 32, 4)
MAKE_PADS(256,  64, 16, 4) MAKE_PADS(256,  64, 32, 4)

// ── NW=8 (256 threads) ───────────────────────────────────────────────────────
MAKE_PADS( 64,  64, 16, 8) MAKE_PADS( 64,  64, 32, 8)
MAKE_PADS( 64, 128, 16, 8) MAKE_PADS( 64, 128, 32, 8)
MAKE_PADS( 64, 256, 16, 8) MAKE_PADS( 64, 256, 32, 8)
MAKE_PADS(128,  64, 16, 8) MAKE_PADS(128,  64, 32, 8)
MAKE_PADS(128, 128, 16, 8) MAKE_PADS(128, 128, 32, 8)
MAKE_PADS(128, 256, 16, 8) MAKE_PADS(128, 256, 32, 8)
MAKE_PADS(256,  64, 16, 8) MAKE_PADS(256,  64, 32, 8)
MAKE_PADS(256, 128, 16, 8) MAKE_PADS(256, 128, 32, 8)

#undef MAKE_PADS
#undef MAKE_LAUNCHER
