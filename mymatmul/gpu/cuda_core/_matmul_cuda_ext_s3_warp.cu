#include <cuda_runtime.h>

/*
 * Stage 3 + explicit warp tiling
 *
 * Template parameters:
 *   BM, BN : CTA output tile dimensions
 *   BK     : K tile depth
 *   WM, WN : warp output tile dimensions
 *   TM, TN : per-thread micro-tile dimensions
 *
 * Hierarchy:
 *   CTA tile   : BM x BN
 *   warp tile  : WM x WN
 *   thread tile: TM x TN
 *
 * Constraints:
 *   THREADS       = (BM/TM) * (BN/TN)
 *   WARPS         = THREADS / 32
 *   WARP_ROWS     = BM / WM
 *   WARP_COLS     = BN / WN
 *   WTHREAD_ROWS  = WM / TM
 *   WTHREAD_COLS  = WN / TN
 *
 * Must satisfy:
 *   WARP_ROWS * WARP_COLS == WARPS
 *   WTHREAD_ROWS * WTHREAD_COLS == 32
 *
 * Example:
 *   BM=128 BN=128 TM=8 TN=8 THREADS=256 WARPS=8
 *   WM=64  WN=32  => WARP_ROWS=2 WARP_COLS=4
 *                  => WTHREAD_ROWS=8 WTHREAD_COLS=4
 */
template <
    int BM, int BN, int BK,
    int WM, int WN,
    int TM, int TN,
    int UNROLL
>
__device__ __forceinline__ void matmul_s3_warp_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    constexpr int WARPS   = THREADS / 32;

    constexpr int WARP_ROWS    = BM / WM;
    constexpr int WARP_COLS    = BN / WN;
    constexpr int WTHREAD_ROWS = WM / TM;
    constexpr int WTHREAD_COLS = WN / TN;

    constexpr int A_ITERS = (BM * BK) / THREADS;
    constexpr int B_ITERS = (BK * BN) / THREADS;

    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");
    static_assert(THREADS % 32 == 0, "THREADS must be a multiple of 32");

    static_assert(BM % WM == 0, "BM must be divisible by WM");
    static_assert(BN % WN == 0, "BN must be divisible by WN");

    static_assert(WARP_ROWS * WARP_COLS == WARPS,
                  "Warp tiling must exactly cover all warps in the CTA");

    static_assert(WM % TM == 0, "WM must be divisible by TM");
    static_assert(WN % TN == 0, "WN must be divisible by TN");

    static_assert(WTHREAD_ROWS * WTHREAD_COLS == 32,
                  "Warp tile must decompose into exactly 32 thread tiles");

    static_assert((BM * BK) % THREADS == 0, "A tile load must divide evenly across threads");
    static_assert((BK * BN) % THREADS == 0, "B tile load must divide evenly across threads");

    __shared__ float A_shared[BM][BK];
    __shared__ float B_shared[BK][BN];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // Physical warp/lane ids
    const int warp_id = tid >> 5;   // tid / 32
    const int lane    = tid & 31;   // tid % 32

    // Warp location inside CTA tile
    const int warp_row = warp_id / WARP_COLS;
    const int warp_col = warp_id % WARP_COLS;

    // Thread-tile location inside warp tile
    const int lane_row = lane / WTHREAD_COLS;
    const int lane_col = lane % WTHREAD_COLS;

    // Final logical thread-tile coordinates at CTA scope
    const int lty = warp_row * WTHREAD_ROWS + lane_row;
    const int ltx = warp_col * WTHREAD_COLS + lane_col;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int row_start = block_row + lty * TM;
    const int col_start = block_col + ltx * TN;

    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Cooperatively load A tile (keep your original tid-based mapping)
        #pragma unroll UNROLL
        for (int i = 0; i < A_ITERS; i++) {
            const int idx = tid + i * THREADS;
            const int r = idx / BK;
            const int c = idx % BK;
            A_shared[r][c] = (block_row + r < M && k0 + c < K)
                ? A[(block_row + r) * K + (k0 + c)]
                : 0.0f;
        }

        // Cooperatively load B tile
        #pragma unroll UNROLL
        for (int i = 0; i < B_ITERS; i++) {
            const int idx = tid + i * THREADS;
            const int r = idx / BN;
            const int c = idx % BN;
            B_shared[r][c] = (k0 + r < K && block_col + c < N)
                ? B[(k0 + r) * N + (block_col + c)]
                : 0.0f;
        }

        __syncthreads();

        // Compute TM x TN micro-tile
        // The only real change from your original kernel is that lty/ltx
        // now come from warp tiling + lane tiling instead of flat CTA remap.
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a[TM], b[TN];

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                a[i] = (float)A_shared[lty * TM + i][kk];
            }

            #pragma unroll
            for (int j = 0; j < TN; j++) {
                b[j] = (float)B_shared[kk][ltx * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }

        __syncthreads();
    }

    // Write back TM x TN outputs
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const int gr = row_start + i;
            const int gc = col_start + j;
            if (gr < M && gc < N) {
                C[gr * N + gc] = acc[i][j];
            }
        }
    }
}

// ---- Launch wrapper macro ----
// Physical threads: x-dim=32 (warp-aligned), y-dim=THREADS/32
#define MAKE_LAUNCHER_WARP(NAME, BM, BN, BK, WM, WN, TM, TN, UNROLL)                 \
extern "C" __global__ void NAME(                                                       \
    const float* __restrict__ A, const float* __restrict__ B,                         \
    float* __restrict__ C, int M, int K, int N) {                                      \
    matmul_s3_warp_impl<BM, BN, BK, WM, WN, TM, TN, UNROLL>(A, B, C, M, K, N);       \
}

// -----------------------------------------------------------------------------
// Suggested warp-tiled variants
// -----------------------------------------------------------------------------

// Your strongest config so far, now with explicit warp tiling:
// CTA 128x128, thread tile 8x8, 256 threads = 8 warps.
// Two natural warp layouts to compare:
//
// 1) warp layout 2x4 across CTA -> warp tile 64x32
MAKE_LAUNCHER_WARP(matmul_cuda_s3_warp_tm8_tn8_bm128_bn128_bk32_wm64_wn32_u8,
                   128, 128, 32, 64, 32, 8, 8, 8)

// 2) transpose: warp layout 4x2 across CTA -> warp tile 32x64
MAKE_LAUNCHER_WARP(matmul_cuda_s3_warp_tm8_tn8_bm128_bn128_bk32_wm32_wn64_u8,
                   128, 128, 32, 32, 64, 8, 8, 8)

// For BM=128, BN=64, TM=8, TN=8:
// THREADS = (128/8)*(64/8) = 16*8 = 128 = 4 warps.
// Natural choices:
//
// 1) warp layout 2x2 across CTA -> warp tile 64x32
MAKE_LAUNCHER_WARP(matmul_cuda_s3_warp_tm8_tn8_bm128_bn64_bk32_wm64_wn32_u8,
                   128, 64, 32, 64, 32, 8, 8, 8)

// 2) warp layout 4x1 across CTA -> warp tile 32x64
MAKE_LAUNCHER_WARP(matmul_cuda_s3_warp_tm8_tn8_bm128_bn64_bk32_wm32_wn64_u8,
                   128, 64, 32, 32, 64, 8, 8, 8)

// For BM=64, BN=64, TM=8, TN=4:
// THREADS = (64/8)*(64/4) = 8*16 = 128 = 4 warps.
// One balanced warp tiling: 2x2 warp layout -> warp tile 32x32
MAKE_LAUNCHER_WARP(matmul_cuda_s3_warp_tm8_tn4_bm64_bn64_bk32_wm32_wn32_u8,
                   64, 64, 32, 32, 32, 8, 4, 8)

// For BM=64, BN=64, TM=4, TN=4:
// THREADS = (64/4)*(64/4) = 16*16 = 256 = 8 warps.
// Nice balanced choice: warp layout 2x4 -> warp tile 32x16
MAKE_LAUNCHER_WARP(matmul_cuda_s3_warp_tm4_tn4_bm64_bn64_bk32_wm32_wn16_u8,
                   64, 64, 32, 32, 16, 4, 4, 8)

// For BM=32, BN=64, TM=4, TN=4:
// THREADS = (32/4)*(64/4) = 8*16 = 128 = 4 warps.
// Nice balanced choice: warp layout 2x2 -> warp tile 16x32
MAKE_LAUNCHER_WARP(matmul_cuda_s3_warp_tm4_tn4_bm32_bn64_bk32_wm16_wn32_u8,
                   32, 64, 32, 16, 32, 4, 4, 8)

