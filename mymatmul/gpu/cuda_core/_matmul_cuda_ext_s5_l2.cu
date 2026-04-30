#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * Stage 5 with L2 cache optimisation (grouped block ordering).
 *
 * Identical to _matmul_cuda_ext_s5.cu except the output tile assigned to each
 * thread block is remapped so that GROUP_M consecutive row-tiles share the same
 * column tile, keeping that B tile hot in L2 across GROUP_M blocks.
 *
 * Mapping (same as Triton's "grouped ordering"):
 *   pid            = blockIdx.y * gridDim.x + blockIdx.x   (linear block id)
 *   group_id       = pid / (GROUP_M * num_pid_n)
 *   first_pid_m    = group_id * GROUP_M
 *   group_size_m   = min(num_pid_m - first_pid_m, GROUP_M)
 *   pid_m          = first_pid_m + (pid % (GROUP_M * num_pid_n)) % group_size_m
 *   pid_n          = (pid % (GROUP_M * num_pid_n)) / group_size_m
 *
 * Tunable: BM,BN in {64,128,256}; BK in {16,32}; UNROLL in {2,4,8,16};
 *          GROUP_M in {4,8}.
 * Excluded: BM=BN=256 (register spill), BM=256,BN=256,BK=32 (smem > 100 KB).
 */

template <int BM, int BN, int BK, int UNROLL, int GROUP_M>
__device__ __forceinline__ void matmul_s5_l2_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
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

    // L2-optimised block-id remapping.
    const int pid       = blockIdx.y * gridDim.x + blockIdx.x;
    const int num_pid_m = gridDim.y;
    const int num_pid_n = gridDim.x;
    const int num_in_group  = GROUP_M * num_pid_n;
    const int group_id      = pid / num_in_group;
    const int first_pid_m   = group_id * GROUP_M;
    const int group_size_m  = min(num_pid_m - first_pid_m, GROUP_M);
    const int pid_m = first_pid_m + (pid % num_in_group) % group_size_m;
    const int pid_n = (pid % num_in_group) / group_size_m;

    const int block_row = pid_m * BM;
    const int block_col = pid_n * BN;

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

#define MAKE_LAUNCHER(BM_, BN_, BK_, U_, GM_)                                      \
extern "C" __global__ __launch_bounds__(256)                                        \
void matmul_cuda_s5_l2_bm##BM_##_bn##BN_##_bk##BK_##_u##U_##_gm##GM_(            \
    const float* __restrict__ A, const float* __restrict__ B,                      \
    float* __restrict__ C, int M, int K, int N) {                                  \
    matmul_s5_l2_impl<BM_, BN_, BK_, U_, GM_>(A, B, C, M, K, N);                 \
}

// BM=64, GROUP_M=4
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
// BM=128, GROUP_M=4
MAKE_LAUNCHER(128,  64, 16,  2, 4) MAKE_LAUNCHER(128,  64, 16,  4, 4)
MAKE_LAUNCHER(128,  64, 16,  8, 4) MAKE_LAUNCHER(128,  64, 16, 16, 4)
MAKE_LAUNCHER(128,  64, 32,  2, 4) MAKE_LAUNCHER(128,  64, 32,  4, 4)
MAKE_LAUNCHER(128,  64, 32,  8, 4) MAKE_LAUNCHER(128,  64, 32, 16, 4)
MAKE_LAUNCHER(128, 128, 16,  2, 4) MAKE_LAUNCHER(128, 128, 16,  4, 4)
MAKE_LAUNCHER(128, 128, 16,  8, 4) MAKE_LAUNCHER(128, 128, 16, 16, 4)
MAKE_LAUNCHER(128, 128, 32,  2, 4) MAKE_LAUNCHER(128, 128, 32,  4, 4)
MAKE_LAUNCHER(128, 128, 32,  8, 4) MAKE_LAUNCHER(128, 128, 32, 16, 4)
MAKE_LAUNCHER(128, 256, 16,  2, 4) MAKE_LAUNCHER(128, 256, 16,  4, 4)
MAKE_LAUNCHER(128, 256, 16,  8, 4) MAKE_LAUNCHER(128, 256, 16, 16, 4)
MAKE_LAUNCHER(128, 256, 32,  2, 4) MAKE_LAUNCHER(128, 256, 32,  4, 4)
MAKE_LAUNCHER(128, 256, 32,  8, 4) MAKE_LAUNCHER(128, 256, 32, 16, 4)
// BM=256, GROUP_M=4
MAKE_LAUNCHER(256,  64, 16,  2, 4) MAKE_LAUNCHER(256,  64, 16,  4, 4)
MAKE_LAUNCHER(256,  64, 16,  8, 4) MAKE_LAUNCHER(256,  64, 16, 16, 4)
MAKE_LAUNCHER(256,  64, 32,  2, 4) MAKE_LAUNCHER(256,  64, 32,  4, 4)
MAKE_LAUNCHER(256,  64, 32,  8, 4) MAKE_LAUNCHER(256,  64, 32, 16, 4)
MAKE_LAUNCHER(256, 128, 16,  2, 4) MAKE_LAUNCHER(256, 128, 16,  4, 4)
MAKE_LAUNCHER(256, 128, 16,  8, 4) MAKE_LAUNCHER(256, 128, 16, 16, 4)
MAKE_LAUNCHER(256, 128, 32,  2, 4) MAKE_LAUNCHER(256, 128, 32,  4, 4)
MAKE_LAUNCHER(256, 128, 32,  8, 4) MAKE_LAUNCHER(256, 128, 32, 16, 4)
// BM=256, BN=256 excluded (register spill)

// BM=64, GROUP_M=8
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
// BM=128, GROUP_M=8
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
// BM=256, GROUP_M=8
MAKE_LAUNCHER(256,  64, 16,  2, 8) MAKE_LAUNCHER(256,  64, 16,  4, 8)
MAKE_LAUNCHER(256,  64, 16,  8, 8) MAKE_LAUNCHER(256,  64, 16, 16, 8)
MAKE_LAUNCHER(256,  64, 32,  2, 8) MAKE_LAUNCHER(256,  64, 32,  4, 8)
MAKE_LAUNCHER(256,  64, 32,  8, 8) MAKE_LAUNCHER(256,  64, 32, 16, 8)
MAKE_LAUNCHER(256, 128, 16,  2, 8) MAKE_LAUNCHER(256, 128, 16,  4, 8)
MAKE_LAUNCHER(256, 128, 16,  8, 8) MAKE_LAUNCHER(256, 128, 16, 16, 8)
MAKE_LAUNCHER(256, 128, 32,  2, 8) MAKE_LAUNCHER(256, 128, 32,  4, 8)
MAKE_LAUNCHER(256, 128, 32,  8, 8) MAKE_LAUNCHER(256, 128, 32, 16, 8)
// BM=256, BN=256 excluded (register spill)

#undef MAKE_LAUNCHER
