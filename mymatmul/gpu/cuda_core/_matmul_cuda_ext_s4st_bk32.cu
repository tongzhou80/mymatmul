#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * s4st with BK=32, dynamic shared memory.
 *
 * Identical algorithm to _matmul_cuda_ext_s4st.cu, but uses extern __shared__
 * instead of static arrays so the smem size is set at launch via
 * cudaFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES).
 *
 * This unlocks BK=32 for bm128_bn128, which needs 64 KB smem
 * (2*(128*32 + 32*128)*4 = 65536 B) — above the 48 KB static limit
 * but within Ada Lovelace's 100 KB maximum.
 *
 * Smem layout (flat, two back-to-back 2D arrays):
 *   [0 .. 2*BM*BK)      : A tiles, [2][BM][BK]  → index buf*BM*BK + row*BK + col
 *   [2*BM*BK .. end)    : B tiles, [2][BK][BN]  → index buf*BK*BN + kk*BN  + col
 */
template <int BM, int BN, int BK, int TM, int TN, int UNROLL>
__device__ __forceinline__ void matmul_s4st_bk32_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    constexpr int LCOLS   = BN / TN;
    constexpr int LROWS   = BM / TM;

    constexpr int A_THREAD_BYTES = BM * BK * (int)sizeof(float) / THREADS;
    constexpr int A_LOAD_BYTES   = (A_THREAD_BYTES >= 16) ? 16 : (A_THREAD_BYTES >= 8) ? 8 : 4;
    constexpr int A_ELEM         = A_LOAD_BYTES / (int)sizeof(float);
    constexpr int A_GROUPS       = BM * BK / A_ELEM / THREADS;

    constexpr int B_THREAD_BYTES = BK * BN * (int)sizeof(float) / THREADS;
    constexpr int B_LOAD_BYTES   = (B_THREAD_BYTES >= 16) ? 16 : (B_THREAD_BYTES >= 8) ? 8 : 4;
    constexpr int B_ELEM         = B_LOAD_BYTES / (int)sizeof(float);
    constexpr int B_GROUPS       = BK * BN / B_ELEM / THREADS;

    // Dynamic smem: A tiles first, then B tiles.
    extern __shared__ float smem_buf[];
    float (*A_shared)[BK] = reinterpret_cast<float(*)[BK]>(smem_buf);
    float (*B_shared)[BN] = reinterpret_cast<float(*)[BN]>(smem_buf + 2 * BM * BK);
    // A_shared[buf*BM + row][col]  →  logically [2][BM][BK]
    // B_shared[buf*BK + kk ][col]  →  logically [2][BK][BN]

    const int tx  = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int ltx = tid % LCOLS, lty = tid / LCOLS;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};

#define ISSUE_TILE(k0_, buf_)                                                               \
    do {                                                                                    \
        _Pragma("unroll")                                                                   \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                            \
            const int _g = tid + _i * THREADS;                                             \
            const int _r = (_g * A_ELEM) / BK, _c = (_g * A_ELEM) % BK;                  \
            __pipeline_memcpy_async(&A_shared[(buf_) * BM + _r][_c],                       \
                                    &A[(block_row + _r) * K + (k0_) + _c],                 \
                                    A_LOAD_BYTES);                                          \
        }                                                                                   \
        _Pragma("unroll")                                                                   \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                            \
            const int _g = tid + _i * THREADS;                                             \
            const int _r = (_g * B_ELEM) / BN, _c = (_g * B_ELEM) % BN;                  \
            __pipeline_memcpy_async(&B_shared[(buf_) * BK + _r][_c],                       \
                                    &B[((k0_) + _r) * N + block_col + _c],                 \
                                    B_LOAD_BYTES);                                          \
        }                                                                                   \
        __pipeline_commit();                                                                \
    } while (0)

#define COMPUTE_TILE(buf_)                                                                  \
    do {                                                                                    \
        _Pragma("unroll UNROLL")                                                            \
        for (int _kk = 0; _kk < BK; _kk++) {                                               \
            float _a[TM], _b[TN];                                                           \
            _Pragma("unroll")                                                               \
            for (int _i = 0; _i < TM; _i++)                                                \
                _a[_i] = A_shared[(buf_) * BM + lty + _i * LROWS][_kk];                    \
            _Pragma("unroll")                                                               \
            for (int _j = 0; _j < TN; _j++)                                                \
                _b[_j] = B_shared[(buf_) * BK + _kk][ltx + _j * LCOLS];                    \
            _Pragma("unroll")                                                               \
            for (int _i = 0; _i < TM; _i++)                                                \
                _Pragma("unroll")                                                           \
                for (int _j = 0; _j < TN; _j++)                                            \
                    acc[_i][_j] += _a[_i] * _b[_j];                                        \
        }                                                                                   \
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

#define MAKE_LAUNCHER_S4ST_BK32(NAME, BM, BN, BK, TM, TN, UNROLL)                     \
extern "C" __global__ void NAME(                                                        \
    const float* __restrict__ A, const float* __restrict__ B,                          \
    float* __restrict__ C, int M, int K, int N) {                                       \
    matmul_s4st_bk32_impl<BM, BN, BK, TM, TN, UNROLL>(A, B, C, M, K, N);              \
}

//                                    NAME                                      BM   BN  BK  TM  TN  UNROLL
// smem = 2*(128*32 + 32*128)*4 = 65536 B = 64 KB  (dynamic, requires attribute set)
MAKE_LAUNCHER_S4ST_BK32(matmul_cuda_s4st_bk32_tm8_tn8_bm128_bn128_bk32_u1,   128, 128, 32,  8,  8,   1)
MAKE_LAUNCHER_S4ST_BK32(matmul_cuda_s4st_bk32_tm8_tn8_bm128_bn128_bk32_u4,   128, 128, 32,  8,  8,   4)
MAKE_LAUNCHER_S4ST_BK32(matmul_cuda_s4st_bk32_tm8_tn8_bm128_bn128_bk32_u8,   128, 128, 32,  8,  8,   8)
MAKE_LAUNCHER_S4ST_BK32(matmul_cuda_s4st_bk32_tm8_tn8_bm128_bn128_bk32_u16,  128, 128, 32,  8,  8,  16)
MAKE_LAUNCHER_S4ST_BK32(matmul_cuda_s4st_bk32_tm8_tn8_bm128_bn128_bk32_u32,  128, 128, 32,  8,  8,  32)
