#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * Stage 4 Strided + Padded (s4stp): s4st with A_shared row padding (+1 float).
 *
 * A_shared is declared [2][BM][BK+1] instead of [2][BM][BK].
 * Row stride = BK+1 = 17 floats (odd), so consecutive rows map to distinct
 * banks — eliminating the residual 2-way A bank conflicts in s4st.
 *
 * Bank mapping for 4 consecutive warp rows (lty=0..3) at column _kk:
 *   row 0: bank = (17*0 + _kk) % 32 =  _kk % 32
 *   row 1: bank = (17*1 + _kk) % 32 = (_kk+17) % 32
 *   row 2: bank = (17*2 + _kk) % 32 = (_kk+ 2) % 32
 *   row 3: bank = (17*3 + _kk) % 32 = (_kk+19) % 32
 * All distinct → zero A conflicts. B conflicts already zero (strided assignment).
 *
 * Cost: row stride 17*4=68 bytes is not 16-byte aligned for odd rows, so
 * cp.async must use 4-byte copies (A_LOAD_BYTES drops from 16→4), issuing 4x
 * more load instructions for A. This is expected to hurt performance slightly.
 */
template <int BM, int BN, int BK, int TM, int TN, int UNROLL>
__device__ __forceinline__ void matmul_s4stp_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    constexpr int LCOLS   = BN / TN;
    constexpr int LROWS   = BM / TM;

    // A row stride = BK+1 floats = 68 bytes — not 16-byte aligned on odd rows.
    // Cap A_LOAD_BYTES to the largest power-of-two that keeps cp.async aligned.
    constexpr int A_STRIDE_BYTES = (BK + 1) * (int)sizeof(float);
    constexpr int A_ALIGN_MAX    = (A_STRIDE_BYTES % 16 == 0) ? 16
                                 : (A_STRIDE_BYTES %  8 == 0) ?  8 : 4;
    constexpr int A_THREAD_BYTES = BM * BK * (int)sizeof(float) / THREADS;
    constexpr int A_LOAD_BYTES   = (A_THREAD_BYTES >= 16 && A_ALIGN_MAX >= 16) ? 16
                                 : (A_THREAD_BYTES >=  8 && A_ALIGN_MAX >=  8) ?  8 : 4;
    constexpr int A_ELEM         = A_LOAD_BYTES / (int)sizeof(float);
    constexpr int A_GROUPS       = BM * BK / A_ELEM / THREADS;

    constexpr int B_THREAD_BYTES = BK * BN * (int)sizeof(float) / THREADS;
    constexpr int B_LOAD_BYTES   = (B_THREAD_BYTES >= 16) ? 16 : (B_THREAD_BYTES >= 8) ? 8 : 4;
    constexpr int B_ELEM         = B_LOAD_BYTES / (int)sizeof(float);
    constexpr int B_GROUPS       = BK * BN / B_ELEM / THREADS;

    // Padded A: row stride = BK+1, eliminates row-to-row bank aliasing
    __shared__ float A_shared[2][BM][BK + 1];
    __shared__ float B_shared[2][BK][BN];

    const int tx  = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int ltx = tid % LCOLS, lty = tid / LCOLS;

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

    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const int gr = block_row + lty + i * LROWS;
            const int gc = block_col + ltx + j * LCOLS;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}

#define MAKE_LAUNCHER_S4STP(NAME, BM, BN, BK, TM, TN, UNROLL)                      \
extern "C" __global__ void NAME(                                                    \
    const float* __restrict__ A, const float* __restrict__ B,                      \
    float* __restrict__ C, int M, int K, int N) {                                   \
    matmul_s4stp_impl<BM, BN, BK, TM, TN, UNROLL>(A, B, C, M, K, N);              \
}

MAKE_LAUNCHER_S4STP(matmul_cuda_s4stp_tm8_tn8_bm64_bn64_bk16_u1,   64, 64, 16, 8, 8,  1)
MAKE_LAUNCHER_S4STP(matmul_cuda_s4stp_tm8_tn8_bm64_bn64_bk16_u4,   64, 64, 16, 8, 8,  4)
MAKE_LAUNCHER_S4STP(matmul_cuda_s4stp_tm8_tn8_bm64_bn64_bk16_u8,   64, 64, 16, 8, 8,  8)
MAKE_LAUNCHER_S4STP(matmul_cuda_s4stp_tm8_tn8_bm64_bn64_bk16_u16,  64, 64, 16, 8, 8, 16)
