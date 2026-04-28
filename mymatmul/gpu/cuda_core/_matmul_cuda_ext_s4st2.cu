#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * Stage 4 Strided-2 (s4st2): s4st with 2-contiguous output assignment.
 *
 * Difference vs s4st:
 *   s4st:  thread ltx owns cols ltx, ltx+LCOLS, ltx+2*LCOLS, ...  (stride LCOLS)
 *          → B reads at step j: B_shared[kk][ltx + j*LCOLS]  — stride LCOLS, scalar load
 *
 *   s4st2: thread ltx owns col PAIRS: {2*ltx, 2*ltx+1}, {2*ltx+2*LCOLS, 2*ltx+2*LCOLS+1}, ...
 *          → B reads at step j: float2 from B_shared[kk][2*ltx + j*2*LCOLS]  — consecutive, vector load
 *
 * Bank conflict analysis (BM=128, BN=128, BK=16, TM=8, TN=8, LCOLS=16):
 *   B_shared is [BK][BN] = [16][128]; bank(B_shared[kk][c]) = c % 32.
 *   At step j, 16 threads (ltx=0..15) load float2 at cols 2*ltx + j*32, 2*ltx + j*32 + 1:
 *     ltx=0  → banks 0,1;  ltx=1 → banks 2,3;  ...  ltx=15 → banks 30,31
 *   All 32 banks distinct; lanes with same ltx access same address → broadcast.
 *   → Zero B conflicts.  (Same as s4st.)
 *
 *   A reads unchanged (depend on lty, not ltx) → zero A conflicts (same as s4st).
 *
 * Benefit: halves B smem load instructions (4 x float2 vs 8 x scalar per kk).
 */
template <int BM, int BN, int BK, int TM, int TN, int UNROLL>
__device__ __forceinline__ void matmul_s4st2_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    static_assert(TN % 2 == 0, "TN must be even for float2 B loads");

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

    __shared__ float A_shared[2][BM][BK];
    __shared__ float B_shared[2][BK][BN];

    const int tx  = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int ltx = tid % LCOLS, lty = tid / LCOLS;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};

    // Global→shared loads: identical to s4st
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

    // 2-contiguous B reads: use float2 directly without intermediate _b[] array.
    // Each bv is immediately consumed by TM FMAs before the next load.
#define COMPUTE_TILE(buf_)                                                              \
    do {                                                                                \
        _Pragma("unroll UNROLL")                                                        \
        for (int _kk = 0; _kk < BK; _kk++) {                                           \
            float _a[TM];                                                               \
            _Pragma("unroll")                                                           \
            for (int _i = 0; _i < TM; _i++)                                            \
                _a[_i] = A_shared[(buf_)][lty + _i * LROWS][_kk];                      \
            _Pragma("unroll")                                                           \
            for (int _j = 0; _j < TN / 2; _j++) {                                      \
                float2 _bv = *reinterpret_cast<const float2*>(                          \
                    &B_shared[(buf_)][_kk][2 * ltx + _j * 2 * LCOLS]);                 \
                _Pragma("unroll")                                                       \
                for (int _i = 0; _i < TM; _i++) {                                      \
                    acc[_i][2 * _j]     += _a[_i] * _bv.x;                             \
                    acc[_i][2 * _j + 1] += _a[_i] * _bv.y;                             \
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

    // 2-contiguous writeback: thread (lty, ltx) owns pairs at stride 2*LCOLS.
    // acc[i][2*j]   → row lty+i*LROWS, col 2*ltx + j*2*LCOLS
    // acc[i][2*j+1] → row lty+i*LROWS, col 2*ltx + j*2*LCOLS + 1
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN / 2; j++) {
            const int gr   = block_row + lty + i * LROWS;
            const int gc   = block_col + 2 * ltx + j * 2 * LCOLS;
            if (gr < M && gc + 1 < N) {
                *reinterpret_cast<float2*>(&C[gr * N + gc]) =
                    make_float2(acc[i][2 * j], acc[i][2 * j + 1]);
            } else if (gr < M && gc < N) {
                C[gr * N + gc] = acc[i][2 * j];
            }
        }
}

#define MAKE_LAUNCHER_S4ST2(NAME, BM, BN, BK, TM, TN, UNROLL)                      \
extern "C" __global__ void NAME(                                                    \
    const float* __restrict__ A, const float* __restrict__ B,                      \
    float* __restrict__ C, int M, int K, int N) {                                   \
    matmul_s4st2_impl<BM, BN, BK, TM, TN, UNROLL>(A, B, C, M, K, N);              \
}

//                               NAME                                     BM   BN  BK  TM  TN  UNROLL
MAKE_LAUNCHER_S4ST2(matmul_cuda_s4st2_tm8_tn8_bm128_bn128_bk16_u1,      128, 128, 16,  8,  8,   1)
MAKE_LAUNCHER_S4ST2(matmul_cuda_s4st2_tm8_tn8_bm128_bn128_bk16_u4,      128, 128, 16,  8,  8,   4)
MAKE_LAUNCHER_S4ST2(matmul_cuda_s4st2_tm8_tn8_bm128_bn128_bk16_u8,      128, 128, 16,  8,  8,   8)
MAKE_LAUNCHER_S4ST2(matmul_cuda_s4st2_tm8_tn8_bm128_bn128_bk16_u16,     128, 128, 16,  8,  8,  16)
