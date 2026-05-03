#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * Stage 6: unified warp-tiled register-double-buffered matmul.
 *
 * Adds NUM_WARPS as a template parameter, unifying s5_w4r and s5_w4r2:
 *   NUM_WARPS=8: 256 threads, 4×2 inter-warp layout  (= s5_w4r)
 *   NUM_WARPS=4: 128 threads, 2×2 inter-warp layout  (= s5_w4r2)
 *
 * Search space: BM, BN in {64, 128, 256}, BK in {16, 32}, NUM_WARPS in {4, 8}.
 * Register constraint (enforced in Python):
 *   BM*BN <= 4096*NUM_WARPS  (TM*TN <= 128)
 */

template <int BM, int BN, int BK, int UNROLL, int NUM_WARPS>
__device__ __forceinline__ void matmul_s6_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    constexpr int WARP_N  = 2;
    constexpr int WARP_M  = NUM_WARPS / WARP_N;
    constexpr int LWARP_N = 8;
    constexpr int LWARP_M = 4;

    constexpr int WARP_TILE_M = BM / WARP_M;
    constexpr int WARP_TILE_N = BN / WARP_N;

    constexpr int TM      = WARP_TILE_M / LWARP_M;
    constexpr int TN      = WARP_TILE_N / LWARP_N;
    constexpr int THREADS = NUM_WARPS * 32;

    constexpr int A_ELEM   = 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    extern __shared__ float smem[];
    auto A_shared = reinterpret_cast<float (*)[BM][BK]>(smem);
    auto B_shared = reinterpret_cast<float (*)[BK][BN]>(smem + 2 * BM * BK);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int warp_id  = tid / 32;
    const int warp_row = warp_id / WARP_N;
    const int warp_col = warp_id % WARP_N;

    const int tiw       = tid % 32;
    const int intra_lty = tiw / LWARP_N;
    const int intra_ltx = tiw % LWARP_N;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};

#define ISSUE_TILE(k0_, buf_)                                                           \
    do {                                                                                \
        _Pragma("unroll")                                                               \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                        \
            const int _g = tid + _i * THREADS;                                         \
            const int _r = (_g * A_ELEM) / BK, _c = (_g * A_ELEM) % BK;              \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_c],                         \
                                    &A[(block_row + _r) * K + (k0_) + _c],             \
                                    A_ELEM * (int)sizeof(float));                       \
        }                                                                               \
        _Pragma("unroll")                                                               \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                        \
            const int _g = tid + _i * THREADS;                                         \
            const int _r = (_g * B_ELEM) / BN, _c = (_g * B_ELEM) % BN;              \
            __pipeline_memcpy_async(&B_shared[(buf_)][_r][_c],                         \
                                    &B[((k0_) + _r) * N + block_col + _c],             \
                                    B_ELEM * (int)sizeof(float));                       \
        }                                                                               \
        __pipeline_commit();                                                             \
    } while (0)

#define COMPUTE_TILE(buf_)                                                              \
    do {                                                                                \
        float _a[2][TM];                                                               \
        float4 _bv[2][TN / 4];                                                         \
        _Pragma("unroll")                                                               \
        for (int _i = 0; _i < TM; _i++)                                               \
            _a[0][_i] = A_shared[(buf_)][warp_row * WARP_TILE_M + intra_lty           \
                                         + _i * LWARP_M][0];                           \
        _Pragma("unroll")                                                               \
        for (int _j = 0; _j < TN / 4; _j++)                                           \
            _bv[0][_j] = *reinterpret_cast<const float4*>(                             \
                &B_shared[(buf_)][0]                                                   \
                          [warp_col * WARP_TILE_N + 4 * intra_ltx                     \
                           + _j * (4 * LWARP_N)]);                                     \
        _Pragma("unroll UNROLL")                                                        \
        for (int _kk = 0; _kk < BK - 1; _kk++) {                                     \
            const int _cur = _kk & 1;                                                  \
            const int _nxt = 1 - _cur;                                                 \
            _Pragma("unroll")                                                           \
            for (int _i = 0; _i < TM; _i++)                                           \
                _a[_nxt][_i] = A_shared[(buf_)][warp_row * WARP_TILE_M + intra_lty   \
                                                 + _i * LWARP_M][_kk + 1];            \
            _Pragma("unroll")                                                           \
            for (int _j = 0; _j < TN / 4; _j++)                                      \
                _bv[_nxt][_j] = *reinterpret_cast<const float4*>(                     \
                    &B_shared[(buf_)][_kk + 1]                                         \
                              [warp_col * WARP_TILE_N + 4 * intra_ltx                 \
                               + _j * (4 * LWARP_N)]);                                 \
            _Pragma("unroll")                                                           \
            for (int _j = 0; _j < TN / 4; _j++) {                                    \
                _Pragma("unroll")                                                       \
                for (int _i = 0; _i < TM; _i++) {                                    \
                    acc[_i][_j * 4 + 0] += _a[_cur][_i] * _bv[_cur][_j].x;          \
                    acc[_i][_j * 4 + 1] += _a[_cur][_i] * _bv[_cur][_j].y;          \
                    acc[_i][_j * 4 + 2] += _a[_cur][_i] * _bv[_cur][_j].z;          \
                    acc[_i][_j * 4 + 3] += _a[_cur][_i] * _bv[_cur][_j].w;          \
                }                                                                       \
            }                                                                           \
        }                                                                               \
        {                                                                               \
            const int _last = (BK - 1) & 1;                                           \
            _Pragma("unroll")                                                           \
            for (int _j = 0; _j < TN / 4; _j++) {                                    \
                _Pragma("unroll")                                                       \
                for (int _i = 0; _i < TM; _i++) {                                    \
                    acc[_i][_j * 4 + 0] += _a[_last][_i] * _bv[_last][_j].x;        \
                    acc[_i][_j * 4 + 1] += _a[_last][_i] * _bv[_last][_j].y;        \
                    acc[_i][_j * 4 + 2] += _a[_last][_i] * _bv[_last][_j].z;        \
                    acc[_i][_j * 4 + 3] += _a[_last][_i] * _bv[_last][_j].w;        \
                }                                                                       \
            }                                                                           \
        }                                                                               \
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

    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN / 4; j++) {
            const int gr = block_row + warp_row * WARP_TILE_M + intra_lty + i * LWARP_M;
            const int gc = block_col + warp_col * WARP_TILE_N + 4 * intra_ltx + j * (4 * LWARP_N);
            if (gr < M && gc + 3 < N) {
                *reinterpret_cast<float4*>(&C[gr * N + gc]) =
                    make_float4(acc[i][j*4+0], acc[i][j*4+1], acc[i][j*4+2], acc[i][j*4+3]);
            } else if (gr < M) {
                for (int k = 0; k < 4 && gc + k < N; k++)
                    C[gr * N + gc + k] = acc[i][j * 4 + k];
            }
        }
}

#define MAKE_LAUNCHER(BM_, BN_, BK_, U_, NW_)                                      \
extern "C" __global__ __launch_bounds__(NW_ * 32)                                  \
void matmul_cuda_s6_bm##BM_##_bn##BN_##_bk##BK_##_u##U_##_nw##NW_(               \
    const float* __restrict__ A, const float* __restrict__ B,                      \
    float* __restrict__ C, int M, int K, int N) {                                  \
    matmul_s6_impl<BM_, BN_, BK_, U_, NW_>(A, B, C, M, K, N);                     \
}

// ── NW=4 (128 threads, 2×2 inter-warp) ────────────────────────────────────────
// BM=64
MAKE_LAUNCHER( 64,  64, 16,  2, 4) MAKE_LAUNCHER( 64,  64, 16,  4, 4)
MAKE_LAUNCHER( 64,  64, 16,  8, 4) MAKE_LAUNCHER( 64,  64, 16, 16, 4)
MAKE_LAUNCHER( 64,  64, 32,  2, 4) MAKE_LAUNCHER( 64,  64, 32,  4, 4)
MAKE_LAUNCHER( 64,  64, 32,  8, 4) MAKE_LAUNCHER( 64,  64, 32, 16, 4)
MAKE_LAUNCHER( 64, 128, 16,  2, 4) MAKE_LAUNCHER( 64, 128, 16,  4, 4)
MAKE_LAUNCHER( 64, 128, 16,  8, 4) MAKE_LAUNCHER( 64, 128, 16, 16, 4)
MAKE_LAUNCHER( 64, 128, 32,  2, 4) MAKE_LAUNCHER( 64, 128, 32,  4, 4)
MAKE_LAUNCHER( 64, 128, 32,  8, 4) MAKE_LAUNCHER( 64, 128, 32, 16, 4)
MAKE_LAUNCHER( 64, 256, 16,  2, 4) MAKE_LAUNCHER( 64, 256, 16,  4, 4)
MAKE_LAUNCHER( 64, 256, 16,  8, 4) MAKE_LAUNCHER( 64, 256, 16, 16, 4)
MAKE_LAUNCHER( 64, 256, 32,  2, 4) MAKE_LAUNCHER( 64, 256, 32,  4, 4)
MAKE_LAUNCHER( 64, 256, 32,  8, 4) MAKE_LAUNCHER( 64, 256, 32, 16, 4)

// BM=128 (BN=256 excluded: BM*BN=32768 > 16384 for NW=4)
MAKE_LAUNCHER(128,  64, 16,  2, 4) MAKE_LAUNCHER(128,  64, 16,  4, 4)
MAKE_LAUNCHER(128,  64, 16,  8, 4) MAKE_LAUNCHER(128,  64, 16, 16, 4)
MAKE_LAUNCHER(128,  64, 32,  2, 4) MAKE_LAUNCHER(128,  64, 32,  4, 4)
MAKE_LAUNCHER(128,  64, 32,  8, 4) MAKE_LAUNCHER(128,  64, 32, 16, 4)
MAKE_LAUNCHER(128, 128, 16,  2, 4) MAKE_LAUNCHER(128, 128, 16,  4, 4)
MAKE_LAUNCHER(128, 128, 16,  8, 4) MAKE_LAUNCHER(128, 128, 16, 16, 4)
MAKE_LAUNCHER(128, 128, 32,  2, 4) MAKE_LAUNCHER(128, 128, 32,  4, 4)
MAKE_LAUNCHER(128, 128, 32,  8, 4) MAKE_LAUNCHER(128, 128, 32, 16, 4)

// BM=256 (BN>=128 excluded for NW=4)
MAKE_LAUNCHER(256,  64, 16,  2, 4) MAKE_LAUNCHER(256,  64, 16,  4, 4)
MAKE_LAUNCHER(256,  64, 16,  8, 4) MAKE_LAUNCHER(256,  64, 16, 16, 4)
MAKE_LAUNCHER(256,  64, 32,  2, 4) MAKE_LAUNCHER(256,  64, 32,  4, 4)
MAKE_LAUNCHER(256,  64, 32,  8, 4) MAKE_LAUNCHER(256,  64, 32, 16, 4)

// ── NW=8 (256 threads, 4×2 inter-warp) ────────────────────────────────────────
// BM=64
MAKE_LAUNCHER( 64,  64, 16,  2, 8) MAKE_LAUNCHER( 64,  64, 16,  4, 8)
MAKE_LAUNCHER( 64,  64, 16,  8, 8) MAKE_LAUNCHER( 64,  64, 16, 16, 8)
MAKE_LAUNCHER( 64,  64, 32,  2, 8) MAKE_LAUNCHER( 64,  64, 32,  4, 8)
MAKE_LAUNCHER( 64,  64, 32,  8, 8) MAKE_LAUNCHER( 64,  64, 32, 16, 8)
MAKE_LAUNCHER( 64, 128, 16,  2, 8) MAKE_LAUNCHER( 64, 128, 16,  4, 8)
MAKE_LAUNCHER( 64, 128, 16,  8, 8) MAKE_LAUNCHER( 64, 128, 16, 16, 8)
MAKE_LAUNCHER( 64, 128, 32,  2, 8) MAKE_LAUNCHER( 64, 128, 32,  4, 8)
MAKE_LAUNCHER( 64, 128, 32,  8, 8) MAKE_LAUNCHER( 64, 128, 32, 16, 8)
MAKE_LAUNCHER( 64, 256, 16,  2, 8) MAKE_LAUNCHER( 64, 256, 16,  4, 8)
MAKE_LAUNCHER( 64, 256, 16,  8, 8) MAKE_LAUNCHER( 64, 256, 16, 16, 8)
MAKE_LAUNCHER( 64, 256, 32,  2, 8) MAKE_LAUNCHER( 64, 256, 32,  4, 8)
MAKE_LAUNCHER( 64, 256, 32,  8, 8) MAKE_LAUNCHER( 64, 256, 32, 16, 8)

// BM=128
MAKE_LAUNCHER(128,  64, 16,  2, 8) MAKE_LAUNCHER(128,  64, 16,  4, 8)
MAKE_LAUNCHER(128,  64, 16,  8, 8) MAKE_LAUNCHER(128,  64, 16, 16, 8)
MAKE_LAUNCHER(128,  64, 32,  2, 8) MAKE_LAUNCHER(128,  64, 32,  4, 8)
MAKE_LAUNCHER(128,  64, 32,  8, 8) MAKE_LAUNCHER(128,  64, 32, 16, 8)
MAKE_LAUNCHER(128, 128, 16,  2, 8) MAKE_LAUNCHER(128, 128, 16,  4, 8)
MAKE_LAUNCHER(128, 128, 16,  8, 8) MAKE_LAUNCHER(128, 128, 16, 16, 8)
MAKE_LAUNCHER(128, 128, 32,  2, 8) MAKE_LAUNCHER(128, 128, 32,  4, 8)
MAKE_LAUNCHER(128, 128, 32,  8, 8) MAKE_LAUNCHER(128, 128, 32, 16, 8)
MAKE_LAUNCHER(128, 256, 16,  2, 8) MAKE_LAUNCHER(128, 256, 16,  4, 8)
MAKE_LAUNCHER(128, 256, 16,  8, 8) MAKE_LAUNCHER(128, 256, 16, 16, 8)
MAKE_LAUNCHER(128, 256, 32,  2, 8) MAKE_LAUNCHER(128, 256, 32,  4, 8)
MAKE_LAUNCHER(128, 256, 32,  8, 8) MAKE_LAUNCHER(128, 256, 32, 16, 8)

// BM=256 (BN=256 excluded: 256*256 > 32768 for NW=8)
MAKE_LAUNCHER(256,  64, 16,  2, 8) MAKE_LAUNCHER(256,  64, 16,  4, 8)
MAKE_LAUNCHER(256,  64, 16,  8, 8) MAKE_LAUNCHER(256,  64, 16, 16, 8)
MAKE_LAUNCHER(256,  64, 32,  2, 8) MAKE_LAUNCHER(256,  64, 32,  4, 8)
MAKE_LAUNCHER(256,  64, 32,  8, 8) MAKE_LAUNCHER(256,  64, 32, 16, 8)
MAKE_LAUNCHER(256, 128, 16,  2, 8) MAKE_LAUNCHER(256, 128, 16,  4, 8)
MAKE_LAUNCHER(256, 128, 16,  8, 8) MAKE_LAUNCHER(256, 128, 16, 16, 8)
MAKE_LAUNCHER(256, 128, 32,  2, 8) MAKE_LAUNCHER(256, 128, 32,  4, 8)
MAKE_LAUNCHER(256, 128, 32,  8, 8) MAKE_LAUNCHER(256, 128, 32, 16, 8)

#undef MAKE_LAUNCHER
