#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

/*
 * TC1: BF16 WMMA matmul, adapted from s6.
 *
 * Same CTA tiling and cp.async double-buffer as s6.  The scalar FMA inner
 * loop (and per-thread register tile) is replaced by warp-level WMMA:
 *   - Each warp holds WM_TILES × WN_TILES = (WARP_TILE_M/16) × (WARP_TILE_N/16)
 *     float32 accumulator fragments.
 *   - COMPUTE_TILE iterates over BK/16 k-steps; each step loads one A fragment
 *     per M-tile (reused across all N-tiles) and one B fragment per N-tile,
 *     then calls mma_sync.
 *   - Intra-warp thread layout (4×8 in s6) disappears: WMMA handles thread
 *     participation internally.
 *
 * Template: (BM, BN, BK, NUM_WARPS)  — UNROLL dropped (no scalar loop).
 * Inputs: bfloat16.  Output: bfloat16 (float32 accumulators converted on write).
 * BK must be a multiple of 16 (WMMA k-dimension).
 *
 * Write-back uses the SM 8.x wmma float32 accumulator fragment layout directly:
 *   thread t (lane 0..31), element e (0..7):
 *     row = (t/4) + row_off[e],  row_off = {0,0,8,8,0,0,8,8}
 *     col = (t%4)*2 + col_off[e], col_off = {0,1,0,1,8,9,8,9}
 * Each thread writes 8 bf16 values per 16×16 tile — no scratch buffer needed.
 */

template <int BM, int BN, int BK, int NUM_WARPS>
__device__ __forceinline__ void matmul_tc1_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    constexpr int WARP_N = 2;
    constexpr int WARP_M = NUM_WARPS / WARP_N;

    constexpr int WARP_TILE_M = BM / WARP_M;
    constexpr int WARP_TILE_N = BN / WARP_N;

    constexpr int WM_TILES = WARP_TILE_M / 16;
    constexpr int WN_TILES = WARP_TILE_N / 16;

    constexpr int THREADS = NUM_WARPS * 32;

    // 8 bf16 (16 bytes) per cp.async when tile is large enough, else 4 bf16 (8 bytes).
    constexpr int A_ELEM   = (BM * BK / THREADS >= 8) ? 8 : 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = (BK * BN / THREADS >= 8) ? 8 : 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    extern __shared__ __nv_bfloat16 smem[];
    auto A_shared = reinterpret_cast<__nv_bfloat16 (*)[BM][BK]>(smem);
    auto B_shared = reinterpret_cast<__nv_bfloat16 (*)[BK][BN]>(smem + 2 * BM * BK);

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

// ── Async load one BM×BK A-tile and BK×BN B-tile into smem[buf_] ────────────
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

// ── WMMA compute over BK/16 k-steps using smem[buf_] ────────────────────────
// Loop order: mt → kk → nt so each A fragment is loaded once per (mt, kk)
// and reused across all WN_TILES B-fragment loads + mma_sync calls.
#define COMPUTE_TILE(buf_)                                                          \
    do {                                                                            \
        wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> _fa; \
        wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::row_major> _fb; \
        _Pragma("unroll")                                                           \
        for (int _mt = 0; _mt < WM_TILES; _mt++) {                                 \
            _Pragma("unroll")                                                       \
            for (int _kk = 0; _kk < BK / 16; _kk++) {                             \
                wmma::load_matrix_sync(_fa,                                         \
                    &A_shared[(buf_)][warp_row * WARP_TILE_M + _mt * 16][_kk * 16], BK); \
                _Pragma("unroll")                                                   \
                for (int _nt = 0; _nt < WN_TILES; _nt++) {                         \
                    wmma::load_matrix_sync(_fb,                                     \
                        &B_shared[(buf_)][_kk * 16][warp_col * WARP_TILE_N + _nt * 16], BN); \
                    wmma::mma_sync(acc_frag[_mt][_nt], _fa, _fb, acc_frag[_mt][_nt]); \
                }                                                                   \
            }                                                                       \
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

    // Write-back: convert float32 accumulators to bf16 using the SM 8.x
    // wmma accumulator fragment layout (no scratch buffer needed).
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
void matmul_cuda_tc1_bm##BM_##_bn##BN_##_bk##BK_##_nw##NW_(                        \
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,       \
    __nv_bfloat16* __restrict__ C, int M, int K, int N) {                           \
    matmul_tc1_impl<BM_, BN_, BK_, NW_>(A, B, C, M, K, N);                         \
}

// ── NW=4 (128 threads, 2×2 inter-warp) ───────────────────────────────────────
// BM=64
MAKE_LAUNCHER( 64,  64, 16, 4) MAKE_LAUNCHER( 64,  64, 32, 4)
MAKE_LAUNCHER( 64, 128, 16, 4) MAKE_LAUNCHER( 64, 128, 32, 4)
MAKE_LAUNCHER( 64, 256, 16, 4) MAKE_LAUNCHER( 64, 256, 32, 4)
// BM=128 (BN=256 excluded: 128×256=32768 > 4096×4=16384)
MAKE_LAUNCHER(128,  64, 16, 4) MAKE_LAUNCHER(128,  64, 32, 4)
MAKE_LAUNCHER(128, 128, 16, 4) MAKE_LAUNCHER(128, 128, 32, 4)
// BM=256 (BN>=128 excluded for NW=4)
MAKE_LAUNCHER(256,  64, 16, 4) MAKE_LAUNCHER(256,  64, 32, 4)

// ── NW=8 (256 threads, 4×2 inter-warp) ───────────────────────────────────────
// BM=64
MAKE_LAUNCHER( 64,  64, 16, 8) MAKE_LAUNCHER( 64,  64, 32, 8)
MAKE_LAUNCHER( 64, 128, 16, 8) MAKE_LAUNCHER( 64, 128, 32, 8)
MAKE_LAUNCHER( 64, 256, 16, 8) MAKE_LAUNCHER( 64, 256, 32, 8)
// BM=128
MAKE_LAUNCHER(128,  64, 16, 8) MAKE_LAUNCHER(128,  64, 32, 8)
MAKE_LAUNCHER(128, 128, 16, 8) MAKE_LAUNCHER(128, 128, 32, 8)
MAKE_LAUNCHER(128, 256, 16, 8) MAKE_LAUNCHER(128, 256, 32, 8)
// BM=256 (BN=256 excluded: 256×256=65536 > 4096×8=32768)
MAKE_LAUNCHER(256,  64, 16, 8) MAKE_LAUNCHER(256,  64, 32, 8)
MAKE_LAUNCHER(256, 128, 16, 8) MAKE_LAUNCHER(256, 128, 32, 8)

#undef MAKE_LAUNCHER
