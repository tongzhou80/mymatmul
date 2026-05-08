#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * Stage 5 PTX: s5 with raw PTX cp.async instead of __pipeline_memcpy_async.
 *
 * Difference from s5:
 *   __pipeline_memcpy_async uses cp.async.ca (cache-all), which fills both L1 and L2.
 *   For large GEMM tiles that will not be reused from L1, this wastes L1 bandwidth.
 *   Raw PTX cp.async.cg.shared.global.L2::128B bypasses L1 entirely and hints the
 *   hardware to prefetch a full 128-byte cache line into L2 — better for streaming
 *   access patterns. __pipeline_commit / __pipeline_wait_prior are unchanged.
 *
 * Expected impact: small or zero at large sizes where we are already compute-bound
 *   (membar=0%, mio_throt=1.7% from NCU). Included as a documented experiment.
 */

__device__ __forceinline__ void cp_async16_cg_l2(void* smem, const void* gmem) {
    unsigned smem32 = __cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(smem32), "l"(gmem) : "memory"
    );
}

template <int BM, int BN, int BK, int UNROLL>
__device__ __forceinline__ void matmul_s5_ptx_impl(
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

    // Each cp.async call copies 4 floats (16 bytes).
    constexpr int A_ELEM   = 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    // Dynamic shared: A_shared[2][BM][BK] then B_shared[2][BK][BN]
    extern __shared__ float smem[];
    auto A_shared = reinterpret_cast<float (*)[BM][BK]>(smem);
    auto B_shared = reinterpret_cast<float (*)[BK][BN]>(smem + 2 * BM * BK);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int ltx = tid % LCOLS;
    const int lty = tid / LCOLS;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};

    // Issue async copies for one A+B tile into smem buffer buf_.
#define ISSUE_TILE(k0_, buf_)                                                           \
    do {                                                                                \
        _Pragma("unroll")                                                               \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                        \
            const int _g = tid + _i * THREADS;                                         \
            const int _r = (_g * A_ELEM) / BK, _c = (_g * A_ELEM) % BK;              \
            cp_async16_cg_l2(&A_shared[(buf_)][_r][_c],                                \
                             &A[(block_row + _r) * K + (k0_) + _c]);                   \
        }                                                                               \
        _Pragma("unroll")                                                               \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                        \
            const int _g = tid + _i * THREADS;                                         \
            const int _r = (_g * B_ELEM) / BN, _c = (_g * B_ELEM) % BN;              \
            cp_async16_cg_l2(&B_shared[(buf_)][_r][_c],                                \
                             &B[((k0_) + _r) * N + block_col + _c]);                   \
        }                                                                               \
        __pipeline_commit();                                                             \
    } while (0)

    // Compute from smem buffer buf_: TM*TN FMAs per kk, UNROLL kk iterations unrolled.
    // A: TM scalar loads (strided). B: TN/2 float2 loads (2-contiguous).
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

    // Writeback: 2-contiguous pairs, with boundary guard.
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

#define MAKE_LAUNCHER(BM_, BN_, BK_, U_)                                            \
extern "C" __global__ __launch_bounds__(256)                                        \
void matmul_cuda_s5_ptx_bm##BM_##_bn##BN_##_bk##BK_##_u##U_(                      \
    const float* __restrict__ A, const float* __restrict__ B,                      \
    float* __restrict__ C, int M, int K, int N) {                                  \
    matmul_s5_ptx_impl<BM_, BN_, BK_, U_>(A, B, C, M, K, N);                      \
}

// BM=64
MAKE_LAUNCHER( 64,  64, 16,  2) MAKE_LAUNCHER( 64,  64, 16,  4)
MAKE_LAUNCHER( 64,  64, 16,  8) MAKE_LAUNCHER( 64,  64, 16, 16)
MAKE_LAUNCHER( 64,  64, 32,  2) MAKE_LAUNCHER( 64,  64, 32,  4)
MAKE_LAUNCHER( 64,  64, 32,  8) MAKE_LAUNCHER( 64,  64, 32, 16)
MAKE_LAUNCHER( 64, 128, 16,  2) MAKE_LAUNCHER( 64, 128, 16,  4)
MAKE_LAUNCHER( 64, 128, 16,  8) MAKE_LAUNCHER( 64, 128, 16, 16)
MAKE_LAUNCHER( 64, 128, 32,  2) MAKE_LAUNCHER( 64, 128, 32,  4)
MAKE_LAUNCHER( 64, 128, 32,  8) MAKE_LAUNCHER( 64, 128, 32, 16)
MAKE_LAUNCHER( 64, 256, 16,  2) MAKE_LAUNCHER( 64, 256, 16,  4)
MAKE_LAUNCHER( 64, 256, 16,  8) MAKE_LAUNCHER( 64, 256, 16, 16)
MAKE_LAUNCHER( 64, 256, 32,  2) MAKE_LAUNCHER( 64, 256, 32,  4)
MAKE_LAUNCHER( 64, 256, 32,  8) MAKE_LAUNCHER( 64, 256, 32, 16)

// BM=128
MAKE_LAUNCHER(128,  64, 16,  2) MAKE_LAUNCHER(128,  64, 16,  4)
MAKE_LAUNCHER(128,  64, 16,  8) MAKE_LAUNCHER(128,  64, 16, 16)
MAKE_LAUNCHER(128,  64, 32,  2) MAKE_LAUNCHER(128,  64, 32,  4)
MAKE_LAUNCHER(128,  64, 32,  8) MAKE_LAUNCHER(128,  64, 32, 16)
MAKE_LAUNCHER(128, 128, 16,  2) MAKE_LAUNCHER(128, 128, 16,  4)
MAKE_LAUNCHER(128, 128, 16,  8) MAKE_LAUNCHER(128, 128, 16, 16)
MAKE_LAUNCHER(128, 128, 32,  2) MAKE_LAUNCHER(128, 128, 32,  4)
MAKE_LAUNCHER(128, 128, 32,  8) MAKE_LAUNCHER(128, 128, 32, 16)
MAKE_LAUNCHER(128, 256, 16,  2) MAKE_LAUNCHER(128, 256, 16,  4)
MAKE_LAUNCHER(128, 256, 16,  8) MAKE_LAUNCHER(128, 256, 16, 16)
MAKE_LAUNCHER(128, 256, 32,  2) MAKE_LAUNCHER(128, 256, 32,  4)
MAKE_LAUNCHER(128, 256, 32,  8) MAKE_LAUNCHER(128, 256, 32, 16)

// BM=256
MAKE_LAUNCHER(256,  64, 16,  2) MAKE_LAUNCHER(256,  64, 16,  4)
MAKE_LAUNCHER(256,  64, 16,  8) MAKE_LAUNCHER(256,  64, 16, 16)
MAKE_LAUNCHER(256,  64, 32,  2) MAKE_LAUNCHER(256,  64, 32,  4)
MAKE_LAUNCHER(256,  64, 32,  8) MAKE_LAUNCHER(256,  64, 32, 16)
MAKE_LAUNCHER(256, 128, 16,  2) MAKE_LAUNCHER(256, 128, 16,  4)
MAKE_LAUNCHER(256, 128, 16,  8) MAKE_LAUNCHER(256, 128, 16, 16)
MAKE_LAUNCHER(256, 128, 32,  2) MAKE_LAUNCHER(256, 128, 32,  4)
MAKE_LAUNCHER(256, 128, 32,  8) MAKE_LAUNCHER(256, 128, 32, 16)
// BM=256, BN=256 excluded: acc[TM][TN]=acc[16][16]=256 floats → register spill
// BM=256, BN=256, BK=32 also excluded: smem=131072 B > 100352 B hardware limit

#undef MAKE_LAUNCHER
