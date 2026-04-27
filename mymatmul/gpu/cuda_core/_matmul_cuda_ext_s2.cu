#include <cuda_runtime.h>

/*
 * Stage 2: Shared Memory Tiling
 * ==============================
 * Educational kernels illustrating the tradeoff between arithmetic intensity
 * and occupancy for two tile / thread-layout configurations.
 *
 * Data type: float32 input/output and accumulation.
 * Shared memory holds float32 tiles.
 *
 * Each thread computes a small micro-tile (TM x TN output elements) with no
 * tx-remap: physical thread layout maps directly to the output tile.
 * The x-dimension covers the N dimension (columns) for coalesced writes to C.
 *
 * Global-memory arithmetic intensity = BM * BN / (BM + BN)
 * Shared-memory arithmetic intensity = TM * TN / (TM + TN)
 *
 * Case 1: BM=8,  BN=32, BK=32, threads=32x4 → TM=2, TN=1
 *   Global A.I. = 8*32/(8+32)   = 6.4
 *   Shared A.I. = 2*1/(2+1)     = 0.67
 *   SMEM: (8*32 + 32*32) * 4B   = 5120 B
 *
 * Case 2: BM=16, BN=32, BK=32, threads=32x8 → TM=2, TN=1
 *   Global A.I. = 16*32/(16+32) = 10.67
 *   Shared A.I. = 2*1/(2+1)     = 0.67  (same as case 1)
 *   SMEM: (16*32 + 32*32) * 4B  = 6144 B
 *
 * Register and shared-memory usage can be inspected by compiling with:
 *   nvcc -arch=sm_89 -O3 -Xptxas -v
 * which is enabled in setup.py for this extension.
 *
 * Occupancy notes (RTX 4090, SM 8.9, 65536 regs / 48 KB SMEM per SM):
 *   Case 1 (128 threads, 4 warps/block):
 *     5120 B SMEM → up to 9 blocks fit on SMEM budget → 36 warps
 *
 *   Case 2 (256 threads, 8 warps/block):
 *     6144 B SMEM → up to 7 blocks fit → 56 warps
 */


// ---------------------------------------------------------------------------
// Case 1 kernel
//   BM=8, BN=32, BK=32, thread block = 32 x 4  (128 threads, 4 warps)
//   TM=2, TN=1: each thread owns rows [ty*2, ty*2+1] and column [tx].
//
//   Cooperative load element counts:
//     A tile (BM x BK = 8 x 32 = 256 elems): 256 / 128 threads = 2 per thread
//     B tile (BK x BN = 32 x 32 = 1024 elems): 1024 / 128 threads = 8 per thread
// ---------------------------------------------------------------------------
extern "C" __global__ void smem_tiled_bm8_bn32_bk32_threads32x4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    constexpr int BM = 8;
    constexpr int BN = 32;
    constexpr int BK = 32;
    constexpr int THREADS = 32 * 4;   // 128

    __shared__ float A_shared[BM][BK];   // 8  * 32 * 4B = 1024 B
    __shared__ float B_shared[BK][BN];   // 32 * 32 * 4B = 4096 B
                                          // total = 5120 B

    const int tx = threadIdx.x;   // 0..31 → N dimension
    const int ty = threadIdx.y;   // 0..3  → M dimension (row pair index)
    const int tid = ty * blockDim.x + tx;   // flat id [0, 128)

    const int col      = blockIdx.x * BN + tx;
    const int row_base = blockIdx.y * BM + ty * 2;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {

        // --- Cooperative load of A tile (BM x BK = 8 x 32) ---
        // 128 threads, 2 elements each, linearised for coalescing
        for (int i = 0; i < 2; i++) {
            const int idx     = tid + i * THREADS;
            const int r       = idx / BK;
            const int c       = idx % BK;
            const int glo_row = blockIdx.y * BM + r;
            const int glo_col = k0 + c;
            A_shared[r][c] = (glo_row < M && glo_col < K)
                             ? A[glo_row * K + glo_col]
                             : 0.0f;
        }

        // --- Cooperative load of B tile (BK x BN = 32 x 32) ---
        // 128 threads, 8 elements each
        for (int i = 0; i < 8; i++) {
            const int idx     = tid + i * THREADS;
            const int r       = idx / BN;
            const int c       = idx % BN;
            const int glo_row = k0 + r;
            const int glo_col = blockIdx.x * BN + c;
            B_shared[r][c] = (glo_row < K && glo_col < N)
                             ? B[glo_row * N + glo_col]
                             : 0.0f;
        }

        __syncthreads();

        // --- Compute: TM=2 rows, TN=1 col, accumulate over BK ---
        for (int kk = 0; kk < BK; kk++) {
            const float b_val = B_shared[kk][tx];
            acc0 += A_shared[ty * 2    ][kk] * b_val;
            acc1 += A_shared[ty * 2 + 1][kk] * b_val;
        }

        __syncthreads();
    }

    // --- Write back ---
    if (col < N) {
        if (row_base     < M) C[ row_base      * N + col] = acc0;
        if (row_base + 1 < M) C[(row_base + 1) * N + col] = acc1;
    }
}



// ---------------------------------------------------------------------------
// Case 2 kernel
//   BM=16, BN=32, BK=32, thread block = 32 x 8  (256 threads, 8 warps)
//   TM=2, TN=1: each thread owns rows [ty*2, ty*2+1] and column [tx].
//
//   Cooperative load element counts:
//     A tile (BM x BK = 16 x 32 = 512 elems):  512 / 256 threads = 2 per thread
//     B tile (BK x BN = 32 x 32 = 1024 elems): 1024 / 256 threads = 4 per thread
//
//   vs Case 1: same TM/TN so same shared-memory A.I. (0.67),
//   but global A.I. rises from 6.4 → 10.67 at the cost of a larger thread block.
// ---------------------------------------------------------------------------
extern "C" __global__ void smem_tiled_bm16_bn32_bk32_threads32x8(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    constexpr int BM = 16;
    constexpr int BN = 32;
    constexpr int BK = 32;
    constexpr int THREADS = 32 * 8;   // 256

    __shared__ float A_shared[BM][BK];   // 16 * 32 * 4B = 2048 B
    __shared__ float B_shared[BK][BN];   // 32 * 32 * 4B = 4096 B
                                          // total = 6144 B

    const int tx = threadIdx.x;   // 0..31 → N dimension
    const int ty = threadIdx.y;   // 0..7  → M dimension (row pair index)
    const int tid = ty * blockDim.x + tx;   // flat id [0, 256)

    const int col      = blockIdx.x * BN + tx;
    const int row_base = blockIdx.y * BM + ty * 2;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {

        // --- Cooperative load of A tile (BM x BK = 16 x 32) ---
        // 256 threads, 2 elements each
        for (int i = 0; i < 2; i++) {
            const int idx     = tid + i * THREADS;
            const int r       = idx / BK;
            const int c       = idx % BK;
            const int glo_row = blockIdx.y * BM + r;
            const int glo_col = k0 + c;
            A_shared[r][c] = (glo_row < M && glo_col < K)
                             ? A[glo_row * K + glo_col]
                             : 0.0f;
        }

        // --- Cooperative load of B tile (BK x BN = 32 x 32) ---
        // 256 threads, 4 elements each
        for (int i = 0; i < 4; i++) {
            const int idx     = tid + i * THREADS;
            const int r       = idx / BN;
            const int c       = idx % BN;
            const int glo_row = k0 + r;
            const int glo_col = blockIdx.x * BN + c;
            B_shared[r][c] = (glo_row < K && glo_col < N)
                             ? B[glo_row * N + glo_col]
                             : 0.0f;
        }

        __syncthreads();

        // --- Compute ---
        for (int kk = 0; kk < BK; kk++) {
            const float b_val = B_shared[kk][tx];
            acc0 += A_shared[ty * 2    ][kk] * b_val;
            acc1 += A_shared[ty * 2 + 1][kk] * b_val;
        }

        __syncthreads();
    }

    // --- Write back ---
    if (col < N) {
        if (row_base     < M) C[ row_base      * N + col] = acc0;
        if (row_base + 1 < M) C[(row_base + 1) * N + col] = acc1;
    }
}


