#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * Stage 4 Strided (s4st): double-buffered cp.async matmul with strided output
 * assignment instead of contiguous blocks.
 *
 * Key difference vs s4:
 *   Contiguous (s4):  thread ltx=k owns output cols  k*TN .. k*TN+TN-1
 *                     → at step _j, warp reads B cols 0+_j, 8+_j, .., 56+_j  (stride TN)
 *                     → BN=64 spans 2 bank periods → 2-way B conflicts
 *
 *   Strided (s4st):   thread ltx=k owns output cols  k, k+LCOLS, k+2*LCOLS, ...
 *                     → at step _j, warp reads B cols _j*LCOLS+0 .. _j*LCOLS+LCOLS-1
 *                     → LCOLS consecutive cols → all distinct banks → zero B conflicts
 *
 * A conflicts: warp spans LROWS=BM/TM consecutive rows (stride 1), vs s4's
 *   stride-TM rows.  With BK=16 (bank period = 2 rows), this gives 2-way A
 *   conflicts instead of 4-way.
 *
 * Global loads and shared memory layout are identical to s4; only COMPUTE_TILE
 * access pattern and writeback differ.
 */
template <int BM, int BN, int BK, int TM, int TN, int UNROLL>
__device__ __forceinline__ void matmul_s4st_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    constexpr int LCOLS   = BN / TN;   // threads along N dimension
    constexpr int LROWS   = BM / TM;   // threads along M dimension

    constexpr int A_THREAD_BYTES = BM * BK * (int)sizeof(float) / THREADS;
    constexpr int A_LOAD_BYTES   = (A_THREAD_BYTES >= 16) ? 16 : (A_THREAD_BYTES >= 8) ? 8 : 4;
    constexpr int A_ELEM         = A_LOAD_BYTES / (int)sizeof(float);
    constexpr int A_GROUPS       = BM * BK / A_ELEM / THREADS;

    constexpr int B_THREAD_BYTES = BK * BN * (int)sizeof(float) / THREADS;
    constexpr int B_LOAD_BYTES   = (B_THREAD_BYTES >= 16) ? 16 : (B_THREAD_BYTES >= 8) ? 8 : 4;
    constexpr int B_ELEM         = B_LOAD_BYTES / (int)sizeof(float);
    constexpr int B_GROUPS       = BK * BN / B_ELEM / THREADS;

    __shared__ float A_shared[2][BM][BK];
    __shared__ float B_shared[2][BK][BN];

    const int tx  = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int ltx = tid % LCOLS, lty = tid / LCOLS;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};

    // Global→shared loads: identical to s4
#define ISSUE_TILE(k0_, buf_)                                                           \
    do {                                                                                \
        _Pragma("unroll")                                                               \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                        \
            const int _g = tid + _i * THREADS;                                         \
            const int _r = (_g * A_ELEM) / BK, _c = (_g * A_ELEM) % BK;              \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_c],                         \
                                    &A[(block_row + _r) * K + (k0_) + _c],             \
                                    A_LOAD_BYTES);                                      \
        }                                                                               \
        _Pragma("unroll")                                                               \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                        \
            const int _g = tid + _i * THREADS;                                         \
            const int _r = (_g * B_ELEM) / BN, _c = (_g * B_ELEM) % BN;              \
            __pipeline_memcpy_async(&B_shared[(buf_)][_r][_c],                         \
                                    &B[((k0_) + _r) * N + block_col + _c],             \
                                    B_LOAD_BYTES);                                      \
        }                                                                               \
        __pipeline_commit();                                                            \
    } while (0)

    // Strided shared memory reads:
    //   A: A_shared[lty + _i*LROWS][_kk]  → warp reads LROWS consecutive rows
    //   B: B_shared[_kk][ltx + _j*LCOLS]  → warp reads LCOLS consecutive cols → no B conflicts
#define COMPUTE_TILE(buf_)                                                              \
    do {                                                                                \
        _Pragma("unroll UNROLL")                                                        \
        for (int _kk = 0; _kk < BK; _kk++) {                                           \
            float _a[TM], _b[TN];                                                       \
            _Pragma("unroll")                                                           \
            for (int _i = 0; _i < TM; _i++)                                            \
                _a[_i] = A_shared[(buf_)][lty + _i * LROWS][_kk];                      \
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

    // Strided writeback: thread (lty, ltx) owns rows lty+i*LROWS, cols ltx+j*LCOLS
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const int gr = block_row + lty + i * LROWS;
            const int gc = block_col + ltx + j * LCOLS;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}

#define MAKE_LAUNCHER_S4ST(NAME, BM, BN, BK, TM, TN, UNROLL)                       \
extern "C" __global__ void NAME(                                                    \
    const float* __restrict__ A, const float* __restrict__ B,                      \
    float* __restrict__ C, int M, int K, int N) {                                   \
    matmul_s4st_impl<BM, BN, BK, TM, TN, UNROLL>(A, B, C, M, K, N);               \
}

//                              NAME                                   BM   BN  BK  TM  TN  UNROLL
MAKE_LAUNCHER_S4ST(matmul_cuda_s4st_tm8_tn8_bm64_bn64_bk16_u1,        64,  64, 16,  8,  8,   1)
MAKE_LAUNCHER_S4ST(matmul_cuda_s4st_tm8_tn8_bm64_bn64_bk16_u4,        64,  64, 16,  8,  8,   4)
MAKE_LAUNCHER_S4ST(matmul_cuda_s4st_tm8_tn8_bm64_bn64_bk16_u8,        64,  64, 16,  8,  8,   8)
MAKE_LAUNCHER_S4ST(matmul_cuda_s4st_tm8_tn8_bm64_bn64_bk16_u16,       64,  64, 16,  8,  8,  16)

MAKE_LAUNCHER_S4ST(matmul_cuda_s4st_tm8_tn8_bm128_bn128_bk16_u1,     128, 128, 16,  8,  8,   1)
MAKE_LAUNCHER_S4ST(matmul_cuda_s4st_tm8_tn8_bm128_bn128_bk16_u4,     128, 128, 16,  8,  8,   4)
MAKE_LAUNCHER_S4ST(matmul_cuda_s4st_tm8_tn8_bm128_bn128_bk16_u8,     128, 128, 16,  8,  8,   8)
MAKE_LAUNCHER_S4ST(matmul_cuda_s4st_tm8_tn8_bm128_bn128_bk16_u16,    128, 128, 16,  8,  8,  16)
