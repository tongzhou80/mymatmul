#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

/*
 * TC4: TC3 + CTA swizzling for improved L2 cache reuse.
 *
 * SWIZZLE (= GROUP_M) groups consecutive SWIZZLE M-tiles into a super-group
 * that spans the full N dimension.  Within the super-group, block IDs are
 * traversed M-first (column-major), so consecutive PIDs share the same B-tile
 * column → better L2 reuse for B.  SWIZZLE=1 is the identity (no remapping).
 *
 * Mapping (Triton-style GROUP_M):
 *   pid      = blockIdx.y * grid_n + blockIdx.x   (linear block ID)
 *   width    = SWIZZLE * grid_n
 *   group_id = pid / width
 *   group_m  = min(grid_m - group_id*SWIZZLE, SWIZZLE)
 *   tile_m   = group_id*SWIZZLE + (pid % width) % group_m
 *   tile_n   = (pid % width) / group_m
 *
 * All other aspects identical to TC3 (PAD_A, PAD_B smem padding).
 */

template <int BM, int BN, int BK, int NUM_WARPS, int PAD_A, int PAD_B, int SWIZZLE>
__device__ __forceinline__ void matmul_tc4_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    constexpr int BKP = BK + PAD_A;
    constexpr int BNP = BN + PAD_B;

    constexpr int WARP_N = 2;
    constexpr int WARP_M = NUM_WARPS / WARP_N;

    constexpr int WARP_TILE_M = BM / WARP_M;
    constexpr int WARP_TILE_N = BN / WARP_N;

    constexpr int WM_TILES = WARP_TILE_M / 16;
    constexpr int WN_TILES = WARP_TILE_N / 16;

    constexpr int THREADS = NUM_WARPS * 32;

    constexpr int A_ELEM   = (BM * BK / THREADS >= 8) ? 8 : 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = (BK * BN / THREADS >= 8) ? 8 : 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    extern __shared__ __nv_bfloat16 smem[];
    auto A_shared = reinterpret_cast<__nv_bfloat16 (*)[BM][BKP]>(smem);
    auto B_shared = reinterpret_cast<__nv_bfloat16 (*)[BK][BNP]>(smem + 2 * BM * BKP);

    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid % 32;
    const int warp_row = warp_id / WARP_N;
    const int warp_col = warp_id % WARP_N;

    // CTA swizzle: remap (blockIdx.y, blockIdx.x) → (tile_m, tile_n)
    const int grid_m = gridDim.y;
    const int grid_n = gridDim.x;
    const int pid    = blockIdx.y * grid_n + blockIdx.x;

    const int width         = SWIZZLE * grid_n;
    const int group_id      = pid / width;
    const int group_start_m = group_id * SWIZZLE;
    const int group_m       = min(grid_m - group_start_m, SWIZZLE);
    const int pid_in_group  = pid % width;
    const int tile_m        = group_start_m + pid_in_group % group_m;
    const int tile_n        = pid_in_group / group_m;

    const int block_row = tile_m * BM;
    const int block_col = tile_n * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[WM_TILES][WN_TILES];
    #pragma unroll
    for (int mt = 0; mt < WM_TILES; mt++)
        #pragma unroll
        for (int nt = 0; nt < WN_TILES; nt++)
            wmma::fill_fragment(acc_frag[mt][nt], 0.0f);

#define ISSUE_TILE(k0_, buf_)                                                       \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                    \
            const int _g = tid + _i * THREADS;                                     \
            const int _r = (_g * A_ELEM) / BK, _c = (_g * A_ELEM) % BK;          \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_c],                     \
                                    &A[(block_row + _r) * K + (k0_) + _c],        \
                                    A_ELEM * (int)sizeof(__nv_bfloat16));           \
        }                                                                           \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                    \
            const int _g = tid + _i * THREADS;                                     \
            const int _r = (_g * B_ELEM) / BN, _c = (_g * B_ELEM) % BN;          \
            __pipeline_memcpy_async(&B_shared[(buf_)][_r][_c],                     \
                                    &B[((k0_) + _r) * N + block_col + _c],        \
                                    B_ELEM * (int)sizeof(__nv_bfloat16));           \
        }                                                                           \
        __pipeline_commit();                                                         \
    } while (0)

#define COMPUTE_TILE(buf_)                                                          \
    do {                                                                            \
        wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> _fa[WM_TILES]; \
        wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::row_major> _fb[WN_TILES]; \
        _Pragma("unroll")                                                           \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                 \
            _Pragma("unroll")                                                       \
            for (int _mt = 0; _mt < WM_TILES; _mt++)                               \
                wmma::load_matrix_sync(_fa[_mt],                                    \
                    &A_shared[(buf_)][warp_row * WARP_TILE_M + _mt * 16][_kk * 16], BKP); \
            _Pragma("unroll")                                                       \
            for (int _nt = 0; _nt < WN_TILES; _nt++)                               \
                wmma::load_matrix_sync(_fb[_nt],                                    \
                    &B_shared[(buf_)][_kk * 16][warp_col * WARP_TILE_N + _nt * 16], BNP); \
            _Pragma("unroll")                                                       \
            for (int _mt = 0; _mt < WM_TILES; _mt++)                               \
                _Pragma("unroll")                                                   \
                for (int _nt = 0; _nt < WN_TILES; _nt++)                           \
                    wmma::mma_sync(acc_frag[_mt][_nt], _fa[_mt], _fb[_nt], acc_frag[_mt][_nt]); \
        }                                                                           \
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

    constexpr int row_off[8] = {0, 0, 8, 8, 0, 0, 8, 8};
    constexpr int col_off[8] = {0, 1, 0, 1, 8, 9, 8, 9};
    const int base_row = lane / 4;
    const int base_col = (lane % 4) * 2;

    #pragma unroll
    for (int mt = 0; mt < WM_TILES; mt++) {
        #pragma unroll
        for (int nt = 0; nt < WN_TILES; nt++) {
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const int gr = block_row + warp_row * WARP_TILE_M + mt * 16
                               + base_row + row_off[e];
                const int gc = block_col + warp_col * WARP_TILE_N + nt * 16
                               + base_col + col_off[e];
                if (gr < M && gc < N)
                    C[gr * N + gc] = __float2bfloat16(acc_frag[mt][nt].x[e]);
            }
        }
    }
}

#define MAKE_LAUNCHER(BM_, BN_, BK_, NW_, PA_, PB_, SW_)                            \
extern "C" __global__ __launch_bounds__(NW_ * 32)                                    \
void matmul_cuda_tc4_bm##BM_##_bn##BN_##_bk##BK_##_nw##NW_##_pa##PA_##_pb##PB_##_sw##SW_( \
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,        \
    __nv_bfloat16* __restrict__ C, int M, int K, int N) {                            \
    matmul_tc4_impl<BM_, BN_, BK_, NW_, PA_, PB_, SW_>(A, B, C, M, K, N);          \
}

#define MAKE_SWIZZLES(BM_, BN_, BK_, NW_, PA_, PB_) \
    MAKE_LAUNCHER(BM_, BN_, BK_, NW_, PA_, PB_, 1)  \
    MAKE_LAUNCHER(BM_, BN_, BK_, NW_, PA_, PB_, 2)  \
    MAKE_LAUNCHER(BM_, BN_, BK_, NW_, PA_, PB_, 4)  \
    MAKE_LAUNCHER(BM_, BN_, BK_, NW_, PA_, PB_, 8)

#define MAKE_PADS(BM_, BN_, BK_, NW_)              \
    MAKE_SWIZZLES(BM_, BN_, BK_, NW_, 0, 0)        \
    MAKE_SWIZZLES(BM_, BN_, BK_, NW_, 0, 8)        \
    MAKE_SWIZZLES(BM_, BN_, BK_, NW_, 8, 0)        \
    MAKE_SWIZZLES(BM_, BN_, BK_, NW_, 8, 8)

// ── NW=4 ─────────────────────────────────────────────────────────────────────
MAKE_PADS( 64,  64, 16, 4) MAKE_PADS( 64,  64, 32, 4)
MAKE_PADS( 64, 128, 16, 4) MAKE_PADS( 64, 128, 32, 4)
MAKE_PADS( 64, 256, 16, 4) MAKE_PADS( 64, 256, 32, 4)
MAKE_PADS(128,  64, 16, 4) MAKE_PADS(128,  64, 32, 4)
MAKE_PADS(128, 128, 16, 4) MAKE_PADS(128, 128, 32, 4)
MAKE_PADS(256,  64, 16, 4) MAKE_PADS(256,  64, 32, 4)

// ── NW=8 ─────────────────────────────────────────────────────────────────────
MAKE_PADS( 64,  64, 16, 8) MAKE_PADS( 64,  64, 32, 8)
MAKE_PADS( 64, 128, 16, 8) MAKE_PADS( 64, 128, 32, 8)
MAKE_PADS( 64, 256, 16, 8) MAKE_PADS( 64, 256, 32, 8)
MAKE_PADS(128,  64, 16, 8) MAKE_PADS(128,  64, 32, 8)
MAKE_PADS(128, 128, 16, 8) MAKE_PADS(128, 128, 32, 8)
MAKE_PADS(128, 256, 16, 8) MAKE_PADS(128, 256, 32, 8)
MAKE_PADS(256,  64, 16, 8) MAKE_PADS(256,  64, 32, 8)
MAKE_PADS(256, 128, 16, 8) MAKE_PADS(256, 128, 32, 8)

#undef MAKE_PADS
#undef MAKE_SWIZZLES
#undef MAKE_LAUNCHER
