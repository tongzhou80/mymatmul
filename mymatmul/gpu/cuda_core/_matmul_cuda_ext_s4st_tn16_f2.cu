#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * s4st_tn16_f2: TN=16, BN=256 with float2 B smem loads.
 *
 * Layout vs s4st_tn16:
 *   s4st_tn16:  thread ltx owns B cols ltx, ltx+LCOLS, ..., ltx+(TN-1)*LCOLS  (scalar)
 *   s4st_tn16_f2: thread ltx owns B col PAIRS 2*ltx+j*2*LCOLS, 2*ltx+j*2*LCOLS+1  (float2)
 *
 * Per kk iteration:
 *   A: 8 scalar loads (unchanged, loop back-edge prevents vectorization)
 *   B: 8 float2 loads  (vs 16 scalar in s4st_tn16_u1)
 *   FMAs: 128
 *
 * smem_ld_wf: 8A + 8B = 16 per kk  (vs 8A + 16B = 24 in tn16_u1)
 *
 * Bank conflict analysis (BN=256, LCOLS=16):
 *   float2 at col 2*ltx + j*32; threads ltx=0..15; banks 2k, 2k+1 → all 32 banks distinct.
 *   Lanes 16..31 share ltx with 0..15 → broadcast. Zero conflicts.
 *
 * Register budget (unroll 1):
 *   acc[8][16]=128, _a[8]=8, _bv=2, overhead ~20 → ~158 regs, well below 255 cliff.
 */

template <int BM, int BN, int BK, int TM, int TN, int UNROLL>
__device__ __forceinline__ void matmul_s4st_tn16_f2_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    static_assert(TN % 2 == 0, "TN must be even for float2 B loads");

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

    /* float2 B: TN/2 float2 loads per kk instead of TN scalar loads */
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
                _Pragma("unroll")                                                        \
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

    /* 2-contiguous writeback: thread (lty,ltx) owns col pairs 2*ltx + j*2*LCOLS */
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN / 2; j++) {
            const int gr = block_row + lty + i * LROWS;
            const int gc = block_col + 2 * ltx + j * 2 * LCOLS;
            if (gr < M && gc + 1 < N) {
                *reinterpret_cast<float2*>(&C[gr * N + gc]) =
                    make_float2(acc[i][2 * j], acc[i][2 * j + 1]);
            } else if (gr < M && gc < N) {
                C[gr * N + gc] = acc[i][2 * j];
            }
        }
}

#define MAKE_LAUNCHER(NAME, BM, BN, BK, TM, TN, UNROLL)                       \
extern "C" __global__ void NAME(                                               \
    const float* __restrict__ A, const float* __restrict__ B,                 \
    float* __restrict__ C, int M, int K, int N) {                              \
    matmul_s4st_tn16_f2_impl<BM, BN, BK, TM, TN, UNROLL>(A, B, C, M, K, N); \
}

//                               NAME                                          BM   BN  BK  TM  TN  UNROLL
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_f2_tm8_tn16_bm128_bn256_bk16_u1,  128, 256, 16,  8, 16,  1)
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_f2_tm8_tn16_bm128_bn256_bk16_u2,  128, 256, 16,  8, 16,  2)
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_f2_tm8_tn16_bm128_bn256_bk16_u4,  128, 256, 16,  8, 16,  4)
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_f2_tm8_tn16_bm128_bn256_bk16_u8,  128, 256, 16,  8, 16,  8)
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_f2_tm8_tn16_bm128_bn256_bk16_u16, 128, 256, 16,  8, 16, 16)

#undef MAKE_LAUNCHER
