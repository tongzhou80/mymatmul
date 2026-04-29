#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * s4st2 with BK=32, dynamic shared memory.
 *
 * Combines the 2-contiguous output layout (float2 B loads) from s4st2 with the
 * dynamic smem approach from s4st_bk32, unlocking BK=32 for bm128_bn128 which
 * needs 64 KB smem (2*(128*32 + 32*128)*4 = 65536 B > 48 KB static limit).
 *
 * Thread layout (bm128_bn128): LCOLS=16, LROWS=16, THREADS=256.
 * Thread ltx owns B columns: 2*ltx, 2*ltx+1, 2*ltx+32, 2*ltx+33, ... (stride 2*LCOLS=32).
 *
 * Bank conflict analysis (BM=128, BN=128, BK=32):
 *   B_shared [BK][BN] = [32][128]; bank(B_shared[kk][c]) = c % 32.
 *   At step j, 16 threads load float2 at cols 2*ltx + j*32:
 *     ltx=0 → banks 0,1; ltx=1 → banks 2,3; ... ltx=15 → banks 30,31 → zero B conflicts.
 *
 *   A_shared [2*BM][BK]; bank = ((buf*BM + row)*BK + kk) % 32 = kk % 32 (BK=32).
 *   Warp has 2 lty values, both reading same kk → same bank → 2-way A conflict.
 *   (Same as s4st bk32; accepted cost for halved tile iterations.)
 *
 * Smem layout (flat):
 *   [0 .. 2*BM*BK)   : A tiles [2][BM][BK]  → A_shared[buf*BM + row][col]
 *   [2*BM*BK .. end) : B tiles [2][BK][BN]  → B_shared[buf*BK + kk ][col]
 */
template <int BM, int BN, int BK, int TM, int TN, int UNROLL>
__device__ __forceinline__ void matmul_s4st2_bk32_impl(
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

    extern __shared__ float smem_buf[];
    float (*A_shared)[BK] = reinterpret_cast<float(*)[BK]>(smem_buf);
    float (*B_shared)[BN] = reinterpret_cast<float(*)[BN]>(smem_buf + 2 * BM * BK);

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
            float _a[TM];                                                                   \
            _Pragma("unroll")                                                               \
            for (int _i = 0; _i < TM; _i++)                                                \
                _a[_i] = A_shared[(buf_) * BM + lty + _i * LROWS][_kk];                    \
            _Pragma("unroll")                                                               \
            for (int _j = 0; _j < TN / 2; _j++) {                                          \
                float2 _bv = *reinterpret_cast<const float2*>(                              \
                    &B_shared[(buf_) * BK + _kk][2 * ltx + _j * 2 * LCOLS]);               \
                _Pragma("unroll")                                                           \
                for (int _i = 0; _i < TM; _i++) {                                          \
                    acc[_i][2 * _j]     += _a[_i] * _bv.x;                                 \
                    acc[_i][2 * _j + 1] += _a[_i] * _bv.y;                                 \
                }                                                                           \
            }                                                                               \
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
        for (int j = 0; j < TN / 2; j++) {
            const int gr  = block_row + lty + i * LROWS;
            const int gc  = block_col + 2 * ltx + j * 2 * LCOLS;
            if (gr < M && gc + 1 < N) {
                *reinterpret_cast<float2*>(&C[gr * N + gc]) =
                    make_float2(acc[i][2 * j], acc[i][2 * j + 1]);
            } else if (gr < M && gc < N) {
                C[gr * N + gc] = acc[i][2 * j];
            }
        }
}

#define MAKE_LAUNCHER_S4ST2_BK32(NAME, BM, BN, BK, TM, TN, UNROLL)                     \
extern "C" __global__ void NAME(                                                         \
    const float* __restrict__ A, const float* __restrict__ B,                           \
    float* __restrict__ C, int M, int K, int N) {                                        \
    matmul_s4st2_bk32_impl<BM, BN, BK, TM, TN, UNROLL>(A, B, C, M, K, N);              \
}

//                                     NAME                                       BM   BN  BK  TM  TN  UNROLL
// smem = 2*(128*32 + 32*128)*4 = 65536 B = 64 KB  (dynamic, requires attribute set)
MAKE_LAUNCHER_S4ST2_BK32(matmul_cuda_s4st2_bk32_tm8_tn8_bm128_bn128_bk32_u1,   128, 128, 32,  8,  8,   1)
MAKE_LAUNCHER_S4ST2_BK32(matmul_cuda_s4st2_bk32_tm8_tn8_bm128_bn128_bk32_u4,   128, 128, 32,  8,  8,   4)
MAKE_LAUNCHER_S4ST2_BK32(matmul_cuda_s4st2_bk32_tm8_tn8_bm128_bn128_bk32_u8,   128, 128, 32,  8,  8,   8)
MAKE_LAUNCHER_S4ST2_BK32(matmul_cuda_s4st2_bk32_tm8_tn8_bm128_bn128_bk32_u16,  128, 128, 32,  8,  8,  16)
MAKE_LAUNCHER_S4ST2_BK32(matmul_cuda_s4st2_bk32_tm8_tn8_bm128_bn128_bk32_u32,  128, 128, 32,  8,  8,  32)
