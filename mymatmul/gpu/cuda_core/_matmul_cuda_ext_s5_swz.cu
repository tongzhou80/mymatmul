#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * Stage 5 SWZ: s5 BK=32 with A-tile swizzle to eliminate the 2-way bank conflict.
 *
 * Root cause in s5 (BK=32):
 *   lty = tid/16, so within each warp the first 16 threads have lty=2k (even) and
 *   the last 16 have lty=2k+1 (odd).  For a given _kk, both halves access column _kk
 *   of their respective rows.  Since A_shared rows are 32 floats wide, consecutive
 *   rows map to the same bank → 2-way conflict every iteration of the BK loop.
 *
 * Fix — XOR-based swizzle on the K column index:
 *   Store: A_shared[buf][_r][_c ^ ((_r & 1) << 4)]  ← A_matrix[_r][_c]
 *   Read:  A_shared[buf][row][_kk ^ ((lty & 1) << 4)]
 *
 *   Odd rows store column _c at physical column _c^16; odd-lty threads read column
 *   _kk at physical column _kk^16. The two halves of the warp now hit banks _kk and
 *   _kk^16 respectively — always different → zero conflicts.
 *
 * Cost: (lty & 1) << 4 is precomputed once per thread (lty is already live); the XOR
 *   inside the BK loop is 1 instruction per kk step. Store swizzle is one XOR per
 *   cp.async call (outside the BK loop). No additional register pressure.
 */

template <int BM, int BN, int UNROLL>
__device__ __forceinline__ void matmul_s5_swz_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    constexpr int BK      = 32;
    constexpr int TM      = BM / 16;
    constexpr int TN      = BN / 16;
    constexpr int LROWS   = 16;
    constexpr int LCOLS   = 16;
    constexpr int THREADS = 256;

    constexpr int A_ELEM   = 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    extern __shared__ float smem[];
    auto A_shared = reinterpret_cast<float (*)[BM][BK]>(smem);
    auto B_shared = reinterpret_cast<float (*)[BK][BN]>(smem + 2 * BM * BK);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int ltx = tid % LCOLS;
    const int lty = tid / LCOLS;

    // Precomputed per-thread swizzle for A reads: 0 for even lty, 16 for odd lty.
    const int a_col_xor = (lty & 1) << 4;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};

#define ISSUE_TILE(k0_, buf_)                                                           \
    do {                                                                                \
        _Pragma("unroll")                                                               \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                        \
            const int _g = tid + _i * THREADS;                                         \
            const int _r = (_g * A_ELEM) / BK, _c = (_g * A_ELEM) % BK;              \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_c ^ ((_r & 1) << 4)],      \
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
        _Pragma("unroll UNROLL")                                                        \
        for (int _kk = 0; _kk < BK; _kk++) {                                           \
            float _a[TM];                                                               \
            _Pragma("unroll")                                                           \
            for (int _i = 0; _i < TM; _i++)                                            \
                _a[_i] = A_shared[(buf_)][lty + _i * LROWS][_kk ^ a_col_xor];         \
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

#define MAKE_LAUNCHER(BM_, BN_, U_)                                                 \
extern "C" __global__ __launch_bounds__(256)                                        \
void matmul_cuda_s5_swz_bm##BM_##_bn##BN_##_u##U_(                                 \
    const float* __restrict__ A, const float* __restrict__ B,                      \
    float* __restrict__ C, int M, int K, int N) {                                  \
    matmul_s5_swz_impl<BM_, BN_, U_>(A, B, C, M, K, N);                           \
}

// BK=32 fixed. BM=256,BN=256 excluded: smem=131072 B > 100352 B limit + acc spill.
MAKE_LAUNCHER( 64,  64,  2) MAKE_LAUNCHER( 64,  64,  4)
MAKE_LAUNCHER( 64,  64,  8) MAKE_LAUNCHER( 64,  64, 16)
MAKE_LAUNCHER( 64, 128,  2) MAKE_LAUNCHER( 64, 128,  4)
MAKE_LAUNCHER( 64, 128,  8) MAKE_LAUNCHER( 64, 128, 16)
MAKE_LAUNCHER( 64, 256,  2) MAKE_LAUNCHER( 64, 256,  4)
MAKE_LAUNCHER( 64, 256,  8) MAKE_LAUNCHER( 64, 256, 16)
MAKE_LAUNCHER(128,  64,  2) MAKE_LAUNCHER(128,  64,  4)
MAKE_LAUNCHER(128,  64,  8) MAKE_LAUNCHER(128,  64, 16)
MAKE_LAUNCHER(128, 128,  2) MAKE_LAUNCHER(128, 128,  4)
MAKE_LAUNCHER(128, 128,  8) MAKE_LAUNCHER(128, 128, 16)
MAKE_LAUNCHER(128, 256,  2) MAKE_LAUNCHER(128, 256,  4)
MAKE_LAUNCHER(128, 256,  8) MAKE_LAUNCHER(128, 256, 16)
MAKE_LAUNCHER(256,  64,  2) MAKE_LAUNCHER(256,  64,  4)
MAKE_LAUNCHER(256,  64,  8) MAKE_LAUNCHER(256,  64, 16)
MAKE_LAUNCHER(256, 128,  2) MAKE_LAUNCHER(256, 128,  4)
MAKE_LAUNCHER(256, 128,  8) MAKE_LAUNCHER(256, 128, 16)

#undef MAKE_LAUNCHER
