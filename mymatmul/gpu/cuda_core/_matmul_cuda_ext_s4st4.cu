#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * Stage 4 Strided-4 (s4st4): float4 B reads + 8×4 warp layout, no A swizzle.
 *
 * 8×4 warp layout (8 maps to N dim, LCOLS_W=8):
 *   float4 B reads: 8 threads × 4 floats = 32 floats = exactly 32 banks → zero B conflicts.
 *   A reads: 4 distinct lty per warp. With BK=16:
 *     bank(A_shared[row][kk]) = (row*16 + kk) % 32
 *     lty=0 → bank=kk; lty=1 → bank=(kk+16)%32; lty=2 → bank=kk; lty=3 → bank=(kk+16)%32
 *     → 2-way A bank conflict (vs 0 for s4st4_xor, vs 0 for s4st2 with 16×2 layout).
 *
 * See s4st4_xor for zero-conflict version using XOR swizzle on A_shared.
 */
template <int BM, int BN, int BK, int TM, int TN, int UNROLL>
__device__ __forceinline__ void matmul_s4st4_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    static_assert(TN % 4 == 0, "TN must be divisible by 4 for float4 B loads");

    constexpr int THREADS = (BM / TM) * (BN / TN);
    constexpr int LCOLS   = BN / TN;
    constexpr int LROWS   = BM / TM;
    constexpr int LCOLS_W = LCOLS / 2;  // ltx per warp along N (= 8)

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

    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int ltx = (warp_id % 2) * LCOLS_W + lane % LCOLS_W;
    const int lty = (warp_id / 2) * 4        + lane / LCOLS_W;

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
                                    A_LOAD_BYTES);                                          \
        }                                                                                   \
        _Pragma("unroll")                                                                   \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                            \
            const int _g = tid + _i * THREADS;                                             \
            const int _r = (_g * B_ELEM) / BN, _c = (_g * B_ELEM) % BN;                  \
            __pipeline_memcpy_async(&B_shared[(buf_)][_r][_c],                             \
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
                _a[_i] = A_shared[(buf_)][lty + _i * LROWS][_kk];                          \
            _Pragma("unroll")                                                               \
            for (int _j = 0; _j < TN / 4; _j++) {                                          \
                float4 _bv = *reinterpret_cast<const float4*>(                              \
                    &B_shared[(buf_)][_kk][4 * ltx + _j * 4 * LCOLS]);                     \
                _Pragma("unroll")                                                           \
                for (int _i = 0; _i < TM; _i++) {                                          \
                    acc[_i][4 * _j + 0] += _a[_i] * _bv.x;                                 \
                    acc[_i][4 * _j + 1] += _a[_i] * _bv.y;                                 \
                    acc[_i][4 * _j + 2] += _a[_i] * _bv.z;                                 \
                    acc[_i][4 * _j + 3] += _a[_i] * _bv.w;                                 \
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
        for (int j = 0; j < TN / 4; j++) {
            const int gr = block_row + lty + i * LROWS;
            const int gc = block_col + 4 * ltx + j * 4 * LCOLS;
            if (gr < M && gc + 3 < N) {
                *reinterpret_cast<float4*>(&C[gr * N + gc]) =
                    make_float4(acc[i][4*j], acc[i][4*j+1], acc[i][4*j+2], acc[i][4*j+3]);
            } else if (gr < M) {
                for (int k = 0; k < 4 && gc + k < N; k++)
                    C[gr * N + gc + k] = acc[i][4*j+k];
            }
        }
}

#define MAKE_LAUNCHER_S4ST4(NAME, BM, BN, BK, TM, TN, UNROLL)                       \
extern "C" __global__ void NAME(                                                     \
    const float* __restrict__ A, const float* __restrict__ B,                       \
    float* __restrict__ C, int M, int K, int N) {                                    \
    matmul_s4st4_impl<BM, BN, BK, TM, TN, UNROLL>(A, B, C, M, K, N);               \
}

//                               NAME                                     BM   BN  BK  TM  TN  UNROLL
MAKE_LAUNCHER_S4ST4(matmul_cuda_s4st4_tm8_tn8_bm128_bn128_bk16_u1,      128, 128, 16,  8,  8,   1)
MAKE_LAUNCHER_S4ST4(matmul_cuda_s4st4_tm8_tn8_bm128_bn128_bk16_u4,      128, 128, 16,  8,  8,   4)
MAKE_LAUNCHER_S4ST4(matmul_cuda_s4st4_tm8_tn8_bm128_bn128_bk16_u8,      128, 128, 16,  8,  8,   8)
MAKE_LAUNCHER_S4ST4(matmul_cuda_s4st4_tm8_tn8_bm128_bn128_bk16_u16,     128, 128, 16,  8,  8,  16)
