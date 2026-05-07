#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

/*
 * TC2: BF16 WMMA matmul — triple-buffered cp.async, adapted from TC1.
 *
 * Same CTA tiling and WMMA compute as TC1, but uses 3 shared-memory buffers
 * instead of 2.  The prolog issues two tiles before entering the main loop,
 * so there are always 2 in-flight cp.async commits when __pipeline_wait_prior(2)
 * is called.  This hides more memory latency without increasing the compute cost.
 *
 * Smem: 3*(BM*BK + BK*BN)*2 bytes (50% more than TC1).
 * Requires K >= 3*BK (satisfied for all benchmarked sizes with BK=16/32).
 */

template <int BM, int BN, int BK, int NUM_WARPS>
__device__ __forceinline__ void matmul_tc2_impl(
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
    constexpr int WN_TILES = WARP_TILE_N / 16;

    constexpr int THREADS = NUM_WARPS * 32;

    constexpr int A_ELEM   = (BM * BK / THREADS >= 8) ? 8 : 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = (BK * BN / THREADS >= 8) ? 8 : 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    extern __shared__ __nv_bfloat16 smem[];
    auto A_shared = reinterpret_cast<__nv_bfloat16 (*)[BM][BK]>(smem);
    auto B_shared = reinterpret_cast<__nv_bfloat16 (*)[BK][BN]>(smem + 3 * BM * BK);

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

#define COMPUTE_TILE(buf_)                                                          \
    do {                                                                            \
        wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> _fa; \
        wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::row_major> _fb; \
        _Pragma("unroll")                                                           \
        for (int _mt = 0; _mt < WM_TILES; _mt++) {                                 \
            _Pragma("unroll")                                                       \
            for (int _kk = 0; _kk < BK / 16; _kk++) {                             \
                wmma::load_matrix_sync(_fa,                                         \
                    &A_shared[(buf_)][warp_row * WARP_TILE_M + _mt * 16][_kk * 16], BK); \
                _Pragma("unroll")                                                   \
                for (int _nt = 0; _nt < WN_TILES; _nt++) {                         \
                    wmma::load_matrix_sync(_fb,                                     \
                        &B_shared[(buf_)][_kk * 16][warp_col * WARP_TILE_N + _nt * 16], BN); \
                    wmma::mma_sync(acc_frag[_mt][_nt], _fa, _fb, acc_frag[_mt][_nt]); \
                }                                                                   \
            }                                                                       \
        }                                                                           \
    } while (0)

    const int num_tiles = K / BK;

    // Prolog: issue first two tiles
    ISSUE_TILE(0, 0);
    ISSUE_TILE(BK, 1);

    // Main loop: issue tile k+2, wait for tile k, compute tile k
    for (int k_iter = 0; k_iter < num_tiles - 2; k_iter++) {
        const int cur = k_iter % 3;
        const int nxt = (k_iter + 2) % 3;
        ISSUE_TILE((k_iter + 2) * BK, nxt);
        __pipeline_wait_prior(2);  // oldest commit (cur) is ready; 2 still in-flight
        __syncthreads();
        COMPUTE_TILE(cur);
        __syncthreads();
    }

    // Epilog: drain last two tiles
    __pipeline_wait_prior(1);
    __syncthreads();
    COMPUTE_TILE((num_tiles - 2) % 3);
    __syncthreads();

    __pipeline_wait_prior(0);
    __syncthreads();
    COMPUTE_TILE((num_tiles - 1) % 3);

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

#define MAKE_LAUNCHER(BM_, BN_, BK_, NW_)                                           \
extern "C" __global__ __launch_bounds__(NW_ * 32)                                   \
void matmul_cuda_tc2_bm##BM_##_bn##BN_##_bk##BK_##_nw##NW_(                        \
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,       \
    __nv_bfloat16* __restrict__ C, int M, int K, int N) {                           \
    matmul_tc2_impl<BM_, BN_, BK_, NW_>(A, B, C, M, K, N);                         \
}

// ── NW=4 ─────────────────────────────────────────────────────────────────────
MAKE_LAUNCHER( 64,  64, 16, 4) MAKE_LAUNCHER( 64,  64, 32, 4)
MAKE_LAUNCHER( 64, 128, 16, 4) MAKE_LAUNCHER( 64, 128, 32, 4)
MAKE_LAUNCHER( 64, 256, 16, 4) MAKE_LAUNCHER( 64, 256, 32, 4)
MAKE_LAUNCHER(128,  64, 16, 4) MAKE_LAUNCHER(128,  64, 32, 4)
MAKE_LAUNCHER(128, 128, 16, 4) MAKE_LAUNCHER(128, 128, 32, 4)
MAKE_LAUNCHER(256,  64, 16, 4) MAKE_LAUNCHER(256,  64, 32, 4)

// ── NW=8 ─────────────────────────────────────────────────────────────────────
MAKE_LAUNCHER( 64,  64, 16, 8) MAKE_LAUNCHER( 64,  64, 32, 8)
MAKE_LAUNCHER( 64, 128, 16, 8) MAKE_LAUNCHER( 64, 128, 32, 8)
MAKE_LAUNCHER( 64, 256, 16, 8) MAKE_LAUNCHER( 64, 256, 32, 8)
MAKE_LAUNCHER(128,  64, 16, 8) MAKE_LAUNCHER(128,  64, 32, 8)
MAKE_LAUNCHER(128, 128, 16, 8) MAKE_LAUNCHER(128, 128, 32, 8)
MAKE_LAUNCHER(128, 256, 16, 8) MAKE_LAUNCHER(128, 256, 32, 8)
MAKE_LAUNCHER(256,  64, 16, 8) MAKE_LAUNCHER(256,  64, 32, 8)
MAKE_LAUNCHER(256, 128, 16, 8) MAKE_LAUNCHER(256, 128, 32, 8)

#undef MAKE_LAUNCHER
