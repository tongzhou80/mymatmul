#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

/*
 * TC4: BF16 WMMA matmul — TC1 with B-tile smem padding to reduce bank conflicts.
 *
 * B_shared row stride padded from BN to BN+8 (bf16 elements).
 * Without padding: stride=BN=64 → (64/2)%32=0 → ALL rows alias to bank 0 (worst case).
 * With PAD_B=8: stride=72 → (72/2)%32=36%32=4 → period-8 cycling, 8 distinct banks.
 * This reduces the B-tile 16-row conflict from 16-way down to 2-way.
 *
 * A_shared is unchanged (no padding).  Smem: (2*BM*BK + 2*BK*(BN+8))*2 bytes.
 * load_matrix_sync for B receives leading dimension BN+8 (padded stride).
 * cp.async for B is 16-byte aligned: row stride=72*2=144 bytes (144/16=9 ✓),
 * column offsets are multiples of 8 BF16 = 16 bytes ✓.
 */

template <int BM, int BN, int BK, int NUM_WARPS>
__device__ __forceinline__ void matmul_tc3_padB2_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    constexpr int PAD_B = 8;
    constexpr int BNP   = BN + PAD_B;  // padded B-tile column stride

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
    auto A_shared = reinterpret_cast<__nv_bfloat16 (*)[BM][BK]>(smem);
    // B_shared uses padded row stride BNP to reduce bank conflicts
    auto B_shared = reinterpret_cast<__nv_bfloat16 (*)[BK][BNP]>(smem + 2 * BM * BK);

    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid % 32;
    const int warp_row = warp_id / WARP_N;
    const int warp_col = warp_id % WARP_N;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[WM_TILES][WN_TILES];
    #pragma unroll
    for (int mt = 0; mt < WM_TILES; mt++)
        #pragma unroll
        for (int nt = 0; nt < WN_TILES; nt++)
            wmma::fill_fragment(acc_frag[mt][nt], 0.0f);

// ── Async load A (unpadded) and B (padded stride BNP) ────────────────────────
// B index uses logical BN width; B_shared[buf_][r][c] handles padded stride.
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

// ── WMMA compute — outer-product order: load all A frags, load all B frags, mma ─
// Fixes redundant B loads: original loaded each B tile WM_TILES times; now once.
#define COMPUTE_TILE(buf_)                                                          \
    do {                                                                            \
        wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> _fa[WM_TILES]; \
        wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::row_major> _fb[WN_TILES]; \
        _Pragma("unroll")                                                           \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                 \
            _Pragma("unroll")                                                       \
            for (int _mt = 0; _mt < WM_TILES; _mt++)                               \
                wmma::load_matrix_sync(_fa[_mt],                                    \
                    &A_shared[(buf_)][warp_row * WARP_TILE_M + _mt * 16][_kk * 16], BK); \
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

#define MAKE_LAUNCHER(BM_, BN_, BK_, NW_)                                           \
extern "C" __global__ __launch_bounds__(NW_ * 32)                                   \
void matmul_cuda_tc3_padB2_bm##BM_##_bn##BN_##_bk##BK_##_nw##NW_(                        \
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,       \
    __nv_bfloat16* __restrict__ C, int M, int K, int N) {                           \
    matmul_tc3_padB2_impl<BM_, BN_, BK_, NW_>(A, B, C, M, K, N);                         \
}

// ── NW=4 ─────────────────────────────────────────────────────────────────────
MAKE_LAUNCHER( 64,  64, 16, 4) MAKE_LAUNCHER( 64,  64, 32, 4)
MAKE_LAUNCHER( 64, 128, 16, 4) MAKE_LAUNCHER( 64, 128, 32, 4)
MAKE_LAUNCHER( 64, 256, 16, 4) MAKE_LAUNCHER( 64, 256, 32, 4)
MAKE_LAUNCHER(128,  64, 16, 4) MAKE_LAUNCHER(128,  64, 32, 4)
MAKE_LAUNCHER(128, 128, 16, 4) MAKE_LAUNCHER(128, 128, 32, 4)
MAKE_LAUNCHER(256,  64, 16, 4) MAKE_LAUNCHER(256,  64, 32, 4)

// ── NW=8 ─────────────────────────────────────────────────────────────────────
MAKE_LAUNCHER( 64,  64, 16, 8) MAKE_LAUNCHER( 64,  64, 32, 8)
MAKE_LAUNCHER( 64, 128, 16, 8) MAKE_LAUNCHER( 64, 128, 32, 8)
MAKE_LAUNCHER( 64, 256, 16, 8) MAKE_LAUNCHER( 64, 256, 32, 8)
MAKE_LAUNCHER(128,  64, 16, 8) MAKE_LAUNCHER(128,  64, 32, 8)
MAKE_LAUNCHER(128, 128, 16, 8) MAKE_LAUNCHER(128, 128, 32, 8)
MAKE_LAUNCHER(128, 256, 16, 8) MAKE_LAUNCHER(128, 256, 32, 8)
MAKE_LAUNCHER(256,  64, 16, 8) MAKE_LAUNCHER(256,  64, 32, 8)
MAKE_LAUNCHER(256, 128, 16, 8) MAKE_LAUNCHER(256, 128, 32, 8)

#undef MAKE_LAUNCHER
