#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * Stage 5 W4P: s5_w4 with 4-buffer paired loading to halve __syncthreads__ overhead.
 *
 * Problem with s5_w4:
 *   Uses BK=16 (avoids 4-way A bank conflict), but twice as many K-tile iterations
 *   as s5's BK=32 → twice as many __syncthreads__ calls. NCU shows 5.5% barrier
 *   stall vs 2.1% for s5.
 *
 * Fix — load and compute 2 BK=16 tiles per "outer step":
 *   4 smem buffers (buf 0..3). ISSUE_PAIR loads 2 consecutive tiles and commits
 *   ONE pipeline group. wait_prior(1) drains a full pair. Result: 2 __syncthreads__
 *   per 32 K-elements — identical to s5 with BK=32.
 *
 * Smem footprint:
 *   4 × (BM×16 + 16×BN) × 4 bytes  ==  2 × (BM×32 + 32×BN) × 4 bytes
 *   Identical to s5's 2×BK=32 budget. No config space shrinkage.
 *
 * Bank conflicts (same as s5_w4):
 *   A_shared[4][BM][16]: 2-way conflict (rows differ by 1 → period-2 bank cycle).
 *   B_shared[4][16][BN]: zero conflicts (float4 loads across contiguous N columns).
 *
 * Buffer cycling (NBUF=4, one commit per pair):
 *   pair p:  cur = (p&1)*2,  nxt = cur^2   → {0,1} ↔ {2,3} alternating.
 *
 * Requires K divisible by 32 (= 2*BK). All benchmark sizes satisfy this.
 */

template <int BM, int BN, int UNROLL>
__device__ __forceinline__ void matmul_s5_w4p_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    constexpr int BK      = 16;
    constexpr int NBUF    = 4;

    constexpr int WARP_M  = 4;
    constexpr int WARP_N  = 2;
    constexpr int LWARP_M = 4;
    constexpr int LWARP_N = 8;

    constexpr int WARP_TILE_M = BM / WARP_M;
    constexpr int WARP_TILE_N = BN / WARP_N;

    constexpr int TM      = WARP_TILE_M / LWARP_M;   // BM/16
    constexpr int TN      = WARP_TILE_N / LWARP_N;   // BN/16
    constexpr int THREADS = 256;

    constexpr int A_ELEM   = 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;   // BM/64
    constexpr int B_ELEM   = 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;   // BN/64

    extern __shared__ float smem[];
    auto A_shared = reinterpret_cast<float (*)[BM][BK]>(smem);
    auto B_shared = reinterpret_cast<float (*)[BK][BN]>(smem + NBUF * BM * BK);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int warp_id  = tid / 32;
    const int warp_row = warp_id / WARP_N;
    const int warp_col = warp_id % WARP_N;
    const int tiw       = tid % 32;
    const int intra_lty = tiw / LWARP_N;
    const int intra_ltx = tiw % LWARP_N;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};

// Load one BK=16 tile into buf_ from global K-offset k0_. No commit.
#define LOAD_TILE(k0_, buf_)                                                            \
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
    } while (0)

// Load two consecutive BK=16 tiles into buf0_ and buf1_, then commit one group.
#define ISSUE_PAIR(k0_, buf0_, buf1_)       \
    do {                                     \
        LOAD_TILE((k0_),       (buf0_));     \
        LOAD_TILE((k0_) + BK,  (buf1_));    \
        __pipeline_commit();                 \
    } while (0)

// Compute from one BK=16 smem tile (float4 B loads, same as s5_w4).
#define COMPUTE_TILE(buf_)                                                              \
    do {                                                                                \
        _Pragma("unroll UNROLL")                                                        \
        for (int _kk = 0; _kk < BK; _kk++) {                                           \
            float _a[TM];                                                               \
            _Pragma("unroll")                                                           \
            for (int _i = 0; _i < TM; _i++)                                            \
                _a[_i] = A_shared[(buf_)][warp_row * WARP_TILE_M + intra_lty + _i * LWARP_M][_kk]; \
            _Pragma("unroll")                                                           \
            for (int _j = 0; _j < TN / 4; _j++) {                                      \
                float4 _bv = *reinterpret_cast<const float4*>(                          \
                    &B_shared[(buf_)][_kk]                                              \
                              [warp_col * WARP_TILE_N + 4 * intra_ltx + _j * (4 * LWARP_N)]); \
                _Pragma("unroll")                                                       \
                for (int _i = 0; _i < TM; _i++) {                                      \
                    acc[_i][_j * 4 + 0] += _a[_i] * _bv.x;                            \
                    acc[_i][_j * 4 + 1] += _a[_i] * _bv.y;                            \
                    acc[_i][_j * 4 + 2] += _a[_i] * _bv.z;                            \
                    acc[_i][_j * 4 + 3] += _a[_i] * _bv.w;                            \
                }                                                                       \
            }                                                                           \
        }                                                                               \
    } while (0)

    // K must be divisible by 2*BK=32.
    const int num_pairs = K / (2 * BK);

    // Prologue: load first pair into buffers 0, 1.
    ISSUE_PAIR(0, 0, 1);

    // Main loop: process pairs p=0..num_pairs-2, prefetching pair p+1.
    // Buffer assignment alternates: even p uses {0,1}, odd p uses {2,3}.
    for (int p = 0; p < num_pairs - 1; p++) {
        const int cur = (p & 1) ? 2 : 0;
        const int nxt = (p & 1) ? 0 : 2;

        ISSUE_PAIR((p + 1) * 2 * BK, nxt, nxt + 1);
        __pipeline_wait_prior(1);   // cur pair's group is done; nxt pair in flight
        __syncthreads();
        // #pragma unroll 1 forces a real loop boundary so nvcc scopes _a/_bv
        // liveness to each iteration — both tiles reuse the same registers.
        _Pragma("unroll 1")
        for (int _t = 0; _t < 2; _t++) COMPUTE_TILE(cur + _t);
        __syncthreads();
    }

    // Epilogue: drain the last pair.
    {
        const int last = ((num_pairs - 1) & 1) ? 2 : 0;
        __pipeline_wait_prior(0);
        __syncthreads();
        _Pragma("unroll 1")
        for (int _t = 0; _t < 2; _t++) COMPUTE_TILE(last + _t);
    }

#undef LOAD_TILE
#undef ISSUE_PAIR
#undef COMPUTE_TILE

    // Writeback: float4 stores with scalar fallback at boundaries.
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN / 4; j++) {
            const int gr = block_row + warp_row * WARP_TILE_M + intra_lty + i * LWARP_M;
            const int gc = block_col + warp_col * WARP_TILE_N + 4 * intra_ltx + j * (4 * LWARP_N);
            if (gr < M && gc + 3 < N) {
                *reinterpret_cast<float4*>(&C[gr * N + gc]) =
                    make_float4(acc[i][j*4+0], acc[i][j*4+1], acc[i][j*4+2], acc[i][j*4+3]);
            } else if (gr < M) {
                for (int k = 0; k < 4 && gc + k < N; k++)
                    C[gr * N + gc + k] = acc[i][j * 4 + k];
            }
        }
}

#define MAKE_LAUNCHER(BM_, BN_, U_)                                                  \
extern "C" __global__ __launch_bounds__(256)                                         \
void matmul_cuda_s5_w4p_bm##BM_##_bn##BN_##_u##U_(                                  \
    const float* __restrict__ A, const float* __restrict__ B,                       \
    float* __restrict__ C, int M, int K, int N) {                                   \
    matmul_s5_w4p_impl<BM_, BN_, U_>(A, B, C, M, K, N);                            \
}

// Valid configs: smem = 4*(BM+BN)*16*4 bytes <= 100352 → BM+BN <= 391
// BM=BN=256 excluded: acc[16][16]=256 float regs → register spill
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
// BM=256, BN=256: smem=131072 bytes > 100352 limit; also acc spill

#undef MAKE_LAUNCHER
