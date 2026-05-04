#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * Stage 7 swz: s7 with CTA swizzle-by-2 with M, N, K baked as compile-time constants.
 *
 * Compiled JIT per (M, N, K) via:
 *   nvcc -DM_VAL=<M> -DN_VAL=<N> -DK_VAL=<K> ...
 *
 * With M/N/K constexpr:
 *   - num_tiles = K/BK is a compile-time constant → K-tile loop has known
 *     trip count and the compiler can schedule/unroll it freely.
 *   - Bounds checks in the store epilog are statically eliminated when
 *     M%BM==0 and N%BN==0 (enforced by static_assert).
 *   - Index arithmetic involving N and K is simplified at compile time.
 */

template <int BM, int BN, int BK, int UNROLL, int NUM_WARPS>
__device__ __forceinline__ void matmul_s7_swz4_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C
) {
    static_assert(M_VAL % BM == 0, "M must be a multiple of BM");
    static_assert(N_VAL % BN == 0, "N must be a multiple of BN");
    static_assert(K_VAL % BK == 0, "K must be a multiple of BK");

    constexpr int M = M_VAL;
    constexpr int N = N_VAL;
    constexpr int K = K_VAL;

    constexpr int WARP_N  = 2;
    constexpr int WARP_M  = NUM_WARPS / WARP_N;
    constexpr int LWARP_N = 8;
    constexpr int LWARP_M = 4;

    constexpr int WARP_TILE_M = BM / WARP_M;
    constexpr int WARP_TILE_N = BN / WARP_N;

    constexpr int TM      = WARP_TILE_M / LWARP_M;
    constexpr int TN      = WARP_TILE_N / LWARP_N;
    constexpr int THREADS = NUM_WARPS * 32;

    constexpr int A_ELEM   = 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    constexpr int num_tiles = K / BK;

    extern __shared__ float smem[];
    auto A_shared = reinterpret_cast<float (*)[BM][BK]>(smem);
    auto B_shared = reinterpret_cast<float (*)[BK][BN]>(smem + 2 * BM * BK);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int warp_id  = tid / 32;
    const int warp_row = warp_id / WARP_N;
    const int warp_col = warp_id % WARP_N;

    const int tiw       = tid % 32;
    const int intra_lty = tiw / LWARP_N;
    const int intra_ltx = tiw % LWARP_N;

    // Swizzle-by-4: groups of 4 consecutive CTAs cover four adjacent rows in
    // the same column before advancing to the next column.
    //   CTA 0 → (row=0,col=0), …, CTA 3 → (row=3,col=0),
    //   CTA 4 → (row=0,col=1), …, CTA 7 → (row=3,col=1), …
    // GN is a compile-time constant → % and / are optimised by the compiler.
    constexpr int GN    = N_VAL / BN;
    const int pid       = blockIdx.x;
    const int group     = pid >> 2;
    const int m_id      = (group / GN) * 4 + (pid & 3);
    const int n_id      = group % GN;
    const int block_row = m_id * BM;
    const int block_col = n_id * BN;

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
        float _a[2][TM];                                                               \
        float4 _bv[2][TN / 4];                                                         \
        _Pragma("unroll")                                                               \
        for (int _i = 0; _i < TM; _i++)                                               \
            _a[0][_i] = A_shared[(buf_)][warp_row * WARP_TILE_M + intra_lty           \
                                         + _i * LWARP_M][0];                           \
        _Pragma("unroll")                                                               \
        for (int _j = 0; _j < TN / 4; _j++)                                           \
            _bv[0][_j] = *reinterpret_cast<const float4*>(                             \
                &B_shared[(buf_)][0]                                                   \
                          [warp_col * WARP_TILE_N + 4 * intra_ltx                     \
                           + _j * (4 * LWARP_N)]);                                     \
        _Pragma("unroll UNROLL")                                                        \
        for (int _kk = 0; _kk < BK - 1; _kk++) {                                     \
            const int _cur = _kk & 1;                                                  \
            const int _nxt = 1 - _cur;                                                 \
            _Pragma("unroll")                                                           \
            for (int _i = 0; _i < TM; _i++)                                           \
                _a[_nxt][_i] = A_shared[(buf_)][warp_row * WARP_TILE_M + intra_lty   \
                                                 + _i * LWARP_M][_kk + 1];            \
            _Pragma("unroll")                                                           \
            for (int _j = 0; _j < TN / 4; _j++)                                      \
                _bv[_nxt][_j] = *reinterpret_cast<const float4*>(                     \
                    &B_shared[(buf_)][_kk + 1]                                         \
                              [warp_col * WARP_TILE_N + 4 * intra_ltx                 \
                               + _j * (4 * LWARP_N)]);                                 \
            _Pragma("unroll")                                                           \
            for (int _j = 0; _j < TN / 4; _j++) {                                    \
                _Pragma("unroll")                                                       \
                for (int _i = 0; _i < TM; _i++) {                                    \
                    acc[_i][_j * 4 + 0] += _a[_cur][_i] * _bv[_cur][_j].x;          \
                    acc[_i][_j * 4 + 1] += _a[_cur][_i] * _bv[_cur][_j].y;          \
                    acc[_i][_j * 4 + 2] += _a[_cur][_i] * _bv[_cur][_j].z;          \
                    acc[_i][_j * 4 + 3] += _a[_cur][_i] * _bv[_cur][_j].w;          \
                }                                                                       \
            }                                                                           \
        }                                                                               \
        {                                                                               \
            const int _last = (BK - 1) & 1;                                           \
            _Pragma("unroll")                                                           \
            for (int _j = 0; _j < TN / 4; _j++) {                                    \
                _Pragma("unroll")                                                       \
                for (int _i = 0; _i < TM; _i++) {                                    \
                    acc[_i][_j * 4 + 0] += _a[_last][_i] * _bv[_last][_j].x;        \
                    acc[_i][_j * 4 + 1] += _a[_last][_i] * _bv[_last][_j].y;        \
                    acc[_i][_j * 4 + 2] += _a[_last][_i] * _bv[_last][_j].z;        \
                    acc[_i][_j * 4 + 3] += _a[_last][_i] * _bv[_last][_j].w;        \
                }                                                                       \
            }                                                                           \
        }                                                                               \
    } while (0)

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

    // M%BM==0 and N%BN==0 are guaranteed by static_assert above, so every
    // output element written here is within bounds — no branch needed.
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN / 4; j++) {
            const int gr = block_row + warp_row * WARP_TILE_M + intra_lty + i * LWARP_M;
            const int gc = block_col + warp_col * WARP_TILE_N + 4 * intra_ltx + j * (4 * LWARP_N);
            *reinterpret_cast<float4*>(&C[gr * N + gc]) =
                make_float4(acc[i][j*4+0], acc[i][j*4+1], acc[i][j*4+2], acc[i][j*4+3]);
        }
}

#define MAKE_LAUNCHER(BM_, BN_, BK_, U_, NW_)                                      \
extern "C" __global__ __launch_bounds__(NW_ * 32)                                  \
void matmul_cuda_s7_swz4_bm##BM_##_bn##BN_##_bk##BK_##_u##U_##_nw##NW_(               \
    const float* __restrict__ A, const float* __restrict__ B,                      \
    float* __restrict__ C) {                                                        \
    matmul_s7_swz4_impl<BM_, BN_, BK_, U_, NW_>(A, B, C);                               \
}

// ── NW=4 (128 threads, 2×2 inter-warp) ────────────────────────────────────────
// BM=64
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

// BM=128 (BN=256 excluded: BM*BN=32768 > 16384 for NW=4)
MAKE_LAUNCHER(128,  64, 16,  2, 4) MAKE_LAUNCHER(128,  64, 16,  4, 4)
MAKE_LAUNCHER(128,  64, 16,  8, 4) MAKE_LAUNCHER(128,  64, 16, 16, 4)
MAKE_LAUNCHER(128,  64, 32,  2, 4) MAKE_LAUNCHER(128,  64, 32,  4, 4)
MAKE_LAUNCHER(128,  64, 32,  8, 4) MAKE_LAUNCHER(128,  64, 32, 16, 4)
MAKE_LAUNCHER(128, 128, 16,  2, 4) MAKE_LAUNCHER(128, 128, 16,  4, 4)
MAKE_LAUNCHER(128, 128, 16,  8, 4) MAKE_LAUNCHER(128, 128, 16, 16, 4)
MAKE_LAUNCHER(128, 128, 32,  2, 4) MAKE_LAUNCHER(128, 128, 32,  4, 4)
MAKE_LAUNCHER(128, 128, 32,  8, 4) MAKE_LAUNCHER(128, 128, 32, 16, 4)

// BM=256 (BN>=128 excluded for NW=4)
MAKE_LAUNCHER(256,  64, 16,  2, 4) MAKE_LAUNCHER(256,  64, 16,  4, 4)
MAKE_LAUNCHER(256,  64, 16,  8, 4) MAKE_LAUNCHER(256,  64, 16, 16, 4)
MAKE_LAUNCHER(256,  64, 32,  2, 4) MAKE_LAUNCHER(256,  64, 32,  4, 4)
MAKE_LAUNCHER(256,  64, 32,  8, 4) MAKE_LAUNCHER(256,  64, 32, 16, 4)

// ── NW=8 (256 threads, 4×2 inter-warp) ────────────────────────────────────────
// BM=64
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

// BM=128
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

// BM=256 (BN=256 excluded: 256*256 > 32768 for NW=8)
MAKE_LAUNCHER(256,  64, 16,  2, 8) MAKE_LAUNCHER(256,  64, 16,  4, 8)
MAKE_LAUNCHER(256,  64, 16,  8, 8) MAKE_LAUNCHER(256,  64, 16, 16, 8)
MAKE_LAUNCHER(256,  64, 32,  2, 8) MAKE_LAUNCHER(256,  64, 32,  4, 8)
MAKE_LAUNCHER(256,  64, 32,  8, 8) MAKE_LAUNCHER(256,  64, 32, 16, 8)
MAKE_LAUNCHER(256, 128, 16,  2, 8) MAKE_LAUNCHER(256, 128, 16,  4, 8)
MAKE_LAUNCHER(256, 128, 16,  8, 8) MAKE_LAUNCHER(256, 128, 16, 16, 8)
MAKE_LAUNCHER(256, 128, 32,  2, 8) MAKE_LAUNCHER(256, 128, 32,  4, 8)
MAKE_LAUNCHER(256, 128, 32,  8, 8) MAKE_LAUNCHER(256, 128, 32, 16, 8)

#undef MAKE_LAUNCHER
