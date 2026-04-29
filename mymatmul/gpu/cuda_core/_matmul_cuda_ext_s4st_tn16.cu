#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * s4st_tn16: TM=8, TN=16, BM=128, BN=256 — pure C++ inner loop (no inline PTX).
 *
 * Same tile geometry as s4st_tn16_ptx but the COMPUTE_TILE uses plain C++ array
 * indexing for both A and B smem loads. The compiler has full scheduling freedom.
 *
 * Per-kk: 8 scalar A loads + 16 scalar B loads + 128 FMAs  (FMA/load = 5.33)
 * Output layout: strided — thread ltx owns cols ltx, ltx+LCOLS, ..., ltx+(TN-1)*LCOLS
 */

template <int BM, int BN, int BK, int TM, int TN, int UNROLL>
__device__ __forceinline__ void matmul_s4st_tn16_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    constexpr int THREADS = (BM / TM) * (BN / TN);   // 256
    constexpr int LCOLS   = BN / TN;                  // 16
    constexpr int LROWS   = BM / TM;                  // 16

    constexpr int A_ELEM   = 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;   // 2
    constexpr int B_ELEM   = 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;   // 4

    __shared__ float A_shared[2][BM][BK];
    __shared__ float B_shared[2][BK][BN];

    const int tx  = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int ltx = tid % LCOLS;
    const int lty = tid / LCOLS;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};

#define ISSUE_TILE(k0_, buf_)                                                               \
    do {                                                                                    \
        _Pragma("unroll")                                                                   \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                            \
            const int _g = tid + _i * THREADS;                                             \
            const int _r = (_g * A_ELEM) / BK, _c = (_g * A_ELEM) % BK;                  \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_c],                             \
                                    &A[(block_row + _r) * K + (k0_) + _c],                 \
                                    A_ELEM * (int)sizeof(float));                           \
        }                                                                                   \
        _Pragma("unroll")                                                                   \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                            \
            const int _g = tid + _i * THREADS;                                             \
            const int _r = (_g * B_ELEM) / BN, _c = (_g * B_ELEM) % BN;                  \
            __pipeline_memcpy_async(&B_shared[(buf_)][_r][_c],                             \
                                    &B[((k0_) + _r) * N + block_col + _c],                 \
                                    B_ELEM * (int)sizeof(float));                           \
        }                                                                                   \
        __pipeline_commit();                                                                \
    } while (0)

#define COMPUTE_TILE(buf_)                                                              \
    do {                                                                                \
        _Pragma("unroll UNROLL")                                                        \
        for (int _kk = 0; _kk < BK; _kk++) {                                           \
            float _a[TM];                                                               \
            _Pragma("unroll")                                                           \
            for (int _i = 0; _i < TM; _i++)                                            \
                _a[_i] = A_shared[(buf_)][lty + _i * LROWS][_kk];                      \
            float _b[TN];                                                               \
            _Pragma("unroll")                                                           \
            for (int _j = 0; _j < TN; _j++)                                            \
                _b[_j] = B_shared[(buf_)][_kk][ltx + _j * LCOLS];                      \
            _Pragma("unroll")                                                           \
            for (int _i = 0; _i < TM; _i++)                                            \
                _Pragma("unroll")                                                       \
                for (int _j = 0; _j < TN; _j++)                                        \
                    acc[_i][_j] += _a[_i] * _b[_j];                                    \
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

    /* strided writeback: thread (lty, ltx) owns cols ltx, ltx+LCOLS, ..., ltx+(TN-1)*LCOLS */
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const int gr = block_row + lty + i * LROWS;
            const int gc = block_col + ltx + j * LCOLS;
            if (gr < M && gc < N)
                C[gr * N + gc] = acc[i][j];
        }
}

#define MAKE_LAUNCHER(NAME, BM, BN, BK, TM, TN, UNROLL)                           \
extern "C" __global__ void NAME(                                                   \
    const float* __restrict__ A, const float* __restrict__ B,                     \
    float* __restrict__ C, int M, int K, int N) {                                  \
    matmul_s4st_tn16_impl<BM, BN, BK, TM, TN, UNROLL>(A, B, C, M, K, N);        \
}

//                              NAME                                          BM   BN  BK  TM  TN  UNROLL
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_tm8_tn16_bm128_bn256_bk16_u1,   128, 256, 16,  8, 16,  1)
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_tm8_tn16_bm128_bn256_bk16_u2,   128, 256, 16,  8, 16,  2)
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_tm8_tn16_bm128_bn256_bk16_u4,   128, 256, 16,  8, 16,  4)
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_tm8_tn16_bm128_bn256_bk16_u8,   128, 256, 16,  8, 16,  8)
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_tm8_tn16_bm128_bn256_bk16_u16,  128, 256, 16,  8, 16, 16)

#undef MAKE_LAUNCHER
