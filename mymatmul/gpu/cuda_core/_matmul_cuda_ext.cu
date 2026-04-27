#include <cuda_runtime.h>

/* Naive 2D grid: each thread (i,j) computes the full k reduction
   Uses float32 accumulator for better numerical stability with float16 inputs */
extern "C" __global__ void matmul_naive_ijk_2d_grid(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        float sum = 0.0f;  // Accumulate in float32 for stability
        for (int k = 0; k < K; k++) {
            sum += (float)A[i*K + k] * (float)B[k*N + j];
        }
        C[i*N + j] = (float)sum;  // Convert back to input dtype
    }
}


/* Naive 2D grid with improved thread mapping: j to x (fastest-changing in row-major)
   Maps j (column, fastest-changing in C layout) to x (fastest CUDA dimension) */
extern "C" __global__ void matmul_naive_ijk_2d_grid_jx(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // x dimension (changes fastest)
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // y dimension

    if (i < M && j < N) {
        float sum = 0.0f;  // Accumulate in float32 for stability
        for (int k = 0; k < K; k++) {
            sum += (float)A[i*K + k] * (float)B[k*N + j];
        }
        C[i*N + j] = (float)sum;  // Convert back to input dtype
    }
}


/* Tiled matmul: 32x32 tiles with shared memory
   Each thread (tx, ty) computes one element C[i,j] where:
   - i = blockIdx.y * 32 + ty
   - j = blockIdx.x * 32 + tx

   For each K-tile, load 32x32 slice of A and B into shared memory,
   then accumulate 32 elements per iteration. */
extern "C" __global__ void matmul_tiled_32x32(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    const int TILE_SIZE = 32;
    __shared__ float A_shared[32][32];
    __shared__ float B_shared[32][32];

    int i = blockIdx.y * TILE_SIZE + threadIdx.y;
    int j = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;  // Accumulate in float32 for stability

    // Iterate over K dimension in 32-element tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {
        // Load A[i, k_tile:k_tile+32] into shared memory
        // Each thread loads one element: A_shared[ty][tx] = A[i, k_tile + tx]
        if (i < M && (k_tile + threadIdx.x) < K) {
            A_shared[threadIdx.y][threadIdx.x] = A[i * K + (k_tile + threadIdx.x)];
        } else {
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B[k_tile:k_tile+32, j] into shared memory
        // Each thread loads one element: B_shared[ty][tx] = B[k_tile + ty, j]
        if ((k_tile + threadIdx.y) < K && j < N) {
            B_shared[threadIdx.y][threadIdx.x] = B[(k_tile + threadIdx.y) * N + j];
        } else {
            B_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure all data is loaded
        __syncthreads();

        // Accumulate: sum over the 32-element tile
        // Each thread (i,j) accumulates: A[i, k_tile:k_tile+32] * B[k_tile:k_tile+32, j]
        if (i < M && j < N) {
            for (int kk = 0; kk < TILE_SIZE; kk++) {
                sum += (float)A_shared[threadIdx.y][kk] * (float)B_shared[kk][threadIdx.x];
            }
        }

        // Synchronize before next iteration
        __syncthreads();
    }

    // Write result
    if (i < M && j < N) {
        C[i * N + j] = (float)sum;
    }
}


/*
16x16 threads per CTA, computing a 32x32 output tile.

Thread mapping:
  - block computes C tile [block_row : block_row+32, block_col : block_col+32]
  - each thread computes a 2x2 micro-tile:
      (row0, col0), (row0, col1),
      (row1, col0), (row1, col1)

where:
  row0 = block_row + ty
  row1 = block_row + 16 + ty
  col0 = block_col + tx
  col1 = block_col + 16 + tx

K is traversed in tiles of 32.

Shared memory:
  A_shared[32][32] holds a 32x32 tile from A
  B_shared[32][32] holds a 32x32 tile from B

Each thread loads 4 values into A_shared and 4 values into B_shared.
Each thread accumulates 4 output values in float.
*/
extern "C" __global__ void matmul_tiled_32x32_16x16_threads(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    constexpr int BM = 32;   // C tile height
    constexpr int BN = 32;   // C tile width
    constexpr int BK = 32;   // K tile depth
    constexpr int TM = 2;    // per-thread rows
    constexpr int TN = 2;    // per-thread cols

    __shared__ float A_shared[BM][BK];
    __shared__ float B_shared[BK][BN];

    const int tx = threadIdx.x;   // 0..15
    const int ty = threadIdx.y;   // 0..15

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // This thread computes a 2x2 micro-tile in C
    const int row0 = block_row + ty;
    const int row1 = block_row + 16 + ty;
    const int col0 = block_col + tx;
    const int col1 = block_col + 16 + tx;

    float acc00 = 0.0f;   // C[row0, col0]
    float acc01 = 0.0f;   // C[row0, col1]
    float acc10 = 0.0f;   // C[row1, col0]
    float acc11 = 0.0f;   // C[row1, col1]

    // Sweep over K dimension in tiles of 32
    for (int k0 = 0; k0 < K; k0 += BK) {
        // ----------------------------
        // Load A tile into shared memory
        // A_shared has shape [32][32]
        // Each thread loads 4 elements:
        //   [ty][tx], [ty][tx+16], [ty+16][tx], [ty+16][tx+16]
        // ----------------------------

        // A_shared[ty][tx] = A[row0][k0 + tx]
        if (row0 < M && (k0 + tx) < K) {
            A_shared[ty][tx] = A[row0 * K + (k0 + tx)];
        } else {
            A_shared[ty][tx] = static_cast<float>(0);
        }

        // A_shared[ty][tx+16] = A[row0][k0 + tx + 16]
        if (row0 < M && (k0 + tx + 16) < K) {
            A_shared[ty][tx + 16] = A[row0 * K + (k0 + tx + 16)];
        } else {
            A_shared[ty][tx + 16] = static_cast<float>(0);
        }

        // A_shared[ty+16][tx] = A[row1][k0 + tx]
        if (row1 < M && (k0 + tx) < K) {
            A_shared[ty + 16][tx] = A[row1 * K + (k0 + tx)];
        } else {
            A_shared[ty + 16][tx] = static_cast<float>(0);
        }

        // A_shared[ty+16][tx+16] = A[row1][k0 + tx + 16]
        if (row1 < M && (k0 + tx + 16) < K) {
            A_shared[ty + 16][tx + 16] = A[row1 * K + (k0 + tx + 16)];
        } else {
            A_shared[ty + 16][tx + 16] = static_cast<float>(0);
        }

        // ----------------------------
        // Load B tile into shared memory
        // B_shared has shape [32][32]
        // Each thread loads 4 elements:
        //   [ty][tx], [ty][tx+16], [ty+16][tx], [ty+16][tx+16]
        // Here rows correspond to K dimension, cols to N dimension.
        // ----------------------------

        // B_shared[ty][tx] = B[k0 + ty][col0]
        if ((k0 + ty) < K && col0 < N) {
            B_shared[ty][tx] = B[(k0 + ty) * N + col0];
        } else {
            B_shared[ty][tx] = static_cast<float>(0);
        }

        // B_shared[ty][tx+16] = B[k0 + ty][col1]
        if ((k0 + ty) < K && col1 < N) {
            B_shared[ty][tx + 16] = B[(k0 + ty) * N + col1];
        } else {
            B_shared[ty][tx + 16] = static_cast<float>(0);
        }

        // B_shared[ty+16][tx] = B[k0 + ty + 16][col0]
        if ((k0 + ty + 16) < K && col0 < N) {
            B_shared[ty + 16][tx] = B[(k0 + ty + 16) * N + col0];
        } else {
            B_shared[ty + 16][tx] = static_cast<float>(0);
        }

        // B_shared[ty+16][tx+16] = B[k0 + ty + 16][col1]
        if ((k0 + ty + 16) < K && col1 < N) {
            B_shared[ty + 16][tx + 16] = B[(k0 + ty + 16) * N + col1];
        } else {
            B_shared[ty + 16][tx + 16] = static_cast<float>(0);
        }

        __syncthreads();

        // ----------------------------
        // Compute on the shared tiles
        // ----------------------------
        for (int kk = 0; kk < BK; kk++) {
            float a0 = (float)A_shared[ty][kk];
            float a1 = (float)A_shared[ty + 16][kk];

            float b0 = (float)B_shared[kk][tx];
            float b1 = (float)B_shared[kk][tx + 16];

            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        __syncthreads();
    }

    // ----------------------------
    // Write back 4 outputs
    // ----------------------------
    if (row0 < M && col0 < N) {
        C[row0 * N + col0] = (float)acc00;
    }
    if (row0 < M && col1 < N) {
        C[row0 * N + col1] = (float)acc01;
    }
    if (row1 < M && col0 < N) {
        C[row1 * N + col0] = (float)acc10;
    }
    if (row1 < M && col1 < N) {
        C[row1 * N + col1] = (float)acc11;
    }
}


/* Tiled matmul: 32x32 tiles with 32x8 warp-aligned thread layout
   Each thread computes a 4x1 micro-tile, one warp owns a full row stripe.

   Thread mapping:
   - blockDim: 32x8 = 256 threads (8 warps, each with 32 threads)
   - Each thread (tx, ty) computes: C[row0, col], C[row1, col], C[row2, col], C[row3, col]
     where row0 = block_row + ty, row1 = block_row + 8+ty, etc.
     and col = block_col + tx

   Shared memory:
   - A_shared[32][32]: row-major, pitch 64 bytes (bfloat16)
   - B_shared[32][32]: row-major, pitch 64 bytes

   Benefit: One warp operates on one row stripe. When reading B_shared[kk][tx],
   all 32 threads in the warp read B_shared[kk][0..31] sequentially, mapping to
   banks 0..31. No bank conflicts!

   Compared to 16x16 layout:
   - 16x16: Warp spans 2 rows, causing B_shared bank conflicts across rows
   - 32x8: Warp stays in one row, no conflicts in B_shared reads
*/
extern "C" __global__ void matmul_tiled_32x32_32x8_threads(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    constexpr int BM = 32;   // C tile height
    constexpr int BN = 32;   // C tile width
    constexpr int BK = 32;   // K tile depth

    __shared__ float A_shared[BM][BK];
    __shared__ float B_shared[BK][BN];

    const int tx = threadIdx.x;   // 0..31
    const int ty = threadIdx.y;   // 0..7

    const int tid = ty * blockDim.x + tx;   // Linear thread ID within block

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // This thread computes a 4x1 micro-tile in C
    const int col0 = block_col + tx;

    const int row0 = block_row + ty;
    const int row1 = block_row + 8  + ty;
    const int row2 = block_row + 16 + ty;
    const int row3 = block_row + 24 + ty;

    float acc0 = 0.0f;   // C[row0, col0]
    float acc1 = 0.0f;   // C[row1, col0]
    float acc2 = 0.0f;   // C[row2, col0]
    float acc3 = 0.0f;   // C[row3, col0]

    // Sweep over K dimension in tiles of 32
    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile into shared memory: 1024 elements, 256 threads => 4 per thread
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx = tid + i * 256;   // Linearized index 0..1023
            const int r = idx / BK;          // Row in A_shared: 0..31
            const int c = idx % BK;          // Col in A_shared: 0..31

            const int global_r = block_row + r;
            const int global_c = k0 + c;

            if (global_r < M && global_c < K) {
                A_shared[r][c] = A[global_r * K + global_c];
            } else {
                A_shared[r][c] = static_cast<float>(0);
            }
        }

        // Load B tile into shared memory: 1024 elements, 256 threads => 4 per thread
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx = tid + i * 256;   // Linearized index 0..1023
            const int r = idx / BN;          // Row in B_shared (K dimension): 0..31
            const int c = idx % BN;          // Col in B_shared (N dimension): 0..31

            const int global_r = k0 + r;
            const int global_c = block_col + c;

            if (global_r < K && global_c < N) {
                B_shared[r][c] = B[global_r * N + global_c];
            } else {
                B_shared[r][c] = static_cast<float>(0);
            }
        }

        __syncthreads();

        // Compute on the shared tiles
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a0 = (float)A_shared[ty][kk];
            float a1 = (float)A_shared[ty + 8][kk];
            float a2 = (float)A_shared[ty + 16][kk];
            float a3 = (float)A_shared[ty + 24][kk];

            float b0 = (float)B_shared[kk][tx];

            acc0 += a0 * b0;
            acc1 += a1 * b0;
            acc2 += a2 * b0;
            acc3 += a3 * b0;
        }

        __syncthreads();
    }

    // Write back 4 outputs
    if (row0 < M && col0 < N) {
        C[row0 * N + col0] = (float)acc0;
    }
    if (row1 < M && col0 < N) {
        C[row1 * N + col0] = (float)acc1;
    }
    if (row2 < M && col0 < N) {
        C[row2 * N + col0] = (float)acc2;
    }
    if (row3 < M && col0 < N) {
        C[row3 * N + col0] = (float)acc3;
    }
}


/* Tiled matmul: 32x32 tiles with 32x4 warp-aligned thread layout
   Each thread computes an 8x1 micro-tile, two warps own half the output tile.

   Thread mapping:
   - blockDim: 32x4 = 128 threads (4 warps, each with 32 threads)
   - Each thread (tx, ty) computes 8 consecutive rows in a single column:
     C[row0, col], C[row1, col], ..., C[row7, col]
     where rowk = block_row + ty + 8*k

   Shared memory:
   - A_shared[32][32]: row-major, pitch 64 bytes (bfloat16)
   - B_shared[32][32]: row-major, pitch 64 bytes

   Benefit: Extreme warp efficiency - one warp owns 8 consecutive output elements.
   Bank conflict profile is identical to 32x8 (zero conflicts in B_shared).

   Trade-offs:
   - Fewer total threads → lower occupancy per SM
   - Each thread has more work (8 accumulators) → better ILP
   - Fewer syncs per thread block
   - Very high register usage per thread (8 float accumulators)
*/
extern "C" __global__ void matmul_tiled_32x32_32x4_threads(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    constexpr int BM = 32;   // C tile height
    constexpr int BN = 32;   // C tile width
    constexpr int BK = 32;   // K tile depth

    __shared__ float A_shared[BM][BK];
    __shared__ float B_shared[BK][BN];

    const int tx = threadIdx.x;   // 0..31
    const int ty = threadIdx.y;   // 0..3

    const int tid = ty * blockDim.x + tx;   // Linear thread ID within block

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // This thread computes an 8x1 micro-tile in C
    const int col0 = block_col + tx;

    const int row0 = block_row + ty;
    const int row1 = block_row + 4  + ty;
    const int row2 = block_row + 8  + ty;
    const int row3 = block_row + 12 + ty;
    const int row4 = block_row + 16 + ty;
    const int row5 = block_row + 20 + ty;
    const int row6 = block_row + 24 + ty;
    const int row7 = block_row + 28 + ty;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    // Sweep over K dimension in tiles of 32
    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile into shared memory: 1024 elements, 128 threads => 8 per thread
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = tid + i * 128;   // Linearized index 0..1023
            const int r = idx / BK;          // Row in A_shared: 0..31
            const int c = idx % BK;          // Col in A_shared: 0..31

            const int global_r = block_row + r;
            const int global_c = k0 + c;

            if (global_r < M && global_c < K) {
                A_shared[r][c] = A[global_r * K + global_c];
            } else {
                A_shared[r][c] = static_cast<float>(0);
            }
        }

        // Load B tile into shared memory: 1024 elements, 128 threads => 8 per thread
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = tid + i * 128;   // Linearized index 0..1023
            const int r = idx / BN;          // Row in B_shared (K dimension): 0..31
            const int c = idx % BN;          // Col in B_shared (N dimension): 0..31

            const int global_r = k0 + r;
            const int global_c = block_col + c;

            if (global_r < K && global_c < N) {
                B_shared[r][c] = B[global_r * N + global_c];
            } else {
                B_shared[r][c] = static_cast<float>(0);
            }
        }

        __syncthreads();

        // Compute on the shared tiles
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a0 = (float)A_shared[ty][kk];
            float a1 = (float)A_shared[ty + 4][kk];
            float a2 = (float)A_shared[ty + 8][kk];
            float a3 = (float)A_shared[ty + 12][kk];
            float a4 = (float)A_shared[ty + 16][kk];
            float a5 = (float)A_shared[ty + 20][kk];
            float a6 = (float)A_shared[ty + 24][kk];
            float a7 = (float)A_shared[ty + 28][kk];

            float b0 = (float)B_shared[kk][tx];

            acc0 += a0 * b0;
            acc1 += a1 * b0;
            acc2 += a2 * b0;
            acc3 += a3 * b0;
            acc4 += a4 * b0;
            acc5 += a5 * b0;
            acc6 += a6 * b0;
            acc7 += a7 * b0;
        }

        __syncthreads();
    }

    // Write back 8 outputs
    if (row0 < M && col0 < N) {
        C[row0 * N + col0] = (float)acc0;
    }
    if (row1 < M && col0 < N) {
        C[row1 * N + col0] = (float)acc1;
    }
    if (row2 < M && col0 < N) {
        C[row2 * N + col0] = (float)acc2;
    }
    if (row3 < M && col0 < N) {
        C[row3 * N + col0] = (float)acc3;
    }
    if (row4 < M && col0 < N) {
        C[row4 * N + col0] = (float)acc4;
    }
    if (row5 < M && col0 < N) {
        C[row5 * N + col0] = (float)acc5;
    }
    if (row6 < M && col0 < N) {
        C[row6 * N + col0] = (float)acc6;
    }
    if (row7 < M && col0 < N) {
        C[row7 * N + col0] = (float)acc7;
    }
}


/* Tiled matmul: 32x64 output tiles with 32x4 warp-aligned thread layout
   Each thread computes an 8x2 micro-tile.

   Thread mapping:
   - blockDim: 32x4 = 128 threads (4 warps)
   - CTA output tile: BMxBN = 32x64
   - threadIdx.x (0..31) maps to the innermost N dimension
   - Each thread computes:
       C[row0..row7, col0]
       C[row0..row7, col1]
     where:
       col0 = block_col + tx
       col1 = block_col + 32 + tx
       rowk = block_row + ty + 4*k, k=0..7

   Shared memory:
   - A_shared[32][32]
   - B_shared[32][64]

   Tile sizes:
   - BM = 32
   - BN = 64
   - BK = 32
*/
extern "C" __global__ void matmul_tiled_32x64_32x4_threads(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    constexpr int BM = 32;   // C tile height
    constexpr int BN = 64;   // C tile width
    constexpr int BK = 32;   // K tile depth

    __shared__ float A_shared[BM][BK];
    __shared__ float B_shared[BK][BN];

    const int tx = threadIdx.x;   // 0..31
    const int ty = threadIdx.y;   // 0..3
    const int tid = ty * blockDim.x + tx;   // 0..127

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // 8x2 micro-tile per thread
    const int col0 = block_col + tx;
    const int col1 = block_col + 32 + tx;

    const int row0 = block_row + ty;
    const int row1 = block_row + 4  + ty;
    const int row2 = block_row + 8  + ty;
    const int row3 = block_row + 12 + ty;
    const int row4 = block_row + 16 + ty;
    const int row5 = block_row + 20 + ty;
    const int row6 = block_row + 24 + ty;
    const int row7 = block_row + 28 + ty;

    float acc00 = 0.0f, acc01 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f;
    float acc40 = 0.0f, acc41 = 0.0f;
    float acc50 = 0.0f, acc51 = 0.0f;
    float acc60 = 0.0f, acc61 = 0.0f;
    float acc70 = 0.0f, acc71 = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile: 32x32 = 1024 elems, 128 threads => 8 elems/thread
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = tid + i * 128;   // 0..1023
            const int r = idx / BK;          // 0..31
            const int c = idx % BK;          // 0..31

            const int global_r = block_row + r;
            const int global_c = k0 + c;

            if (global_r < M && global_c < K) {
                A_shared[r][c] = A[global_r * K + global_c];
            } else {
                A_shared[r][c] = static_cast<float>(0);
            }
        }

        // Load B tile: 32x64 = 2048 elems, 128 threads => 16 elems/thread
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const int idx = tid + i * 128;   // 0..2047
            const int r = idx / BN;          // 0..31   (K dimension)
            const int c = idx % BN;          // 0..63   (N dimension)

            const int global_r = k0 + r;
            const int global_c = block_col + c;

            if (global_r < K && global_c < N) {
                B_shared[r][c] = B[global_r * N + global_c];
            } else {
                B_shared[r][c] = static_cast<float>(0);
            }
        }

        __syncthreads();

        // Compute on shared tiles
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a0 = (float)A_shared[ty][kk];
            float a1 = (float)A_shared[ty + 4][kk];
            float a2 = (float)A_shared[ty + 8][kk];
            float a3 = (float)A_shared[ty + 12][kk];
            float a4 = (float)A_shared[ty + 16][kk];
            float a5 = (float)A_shared[ty + 20][kk];
            float a6 = (float)A_shared[ty + 24][kk];
            float a7 = (float)A_shared[ty + 28][kk];

            float b0 = (float)B_shared[kk][tx];
            float b1 = (float)B_shared[kk][tx + 32];

            acc00 += a0 * b0;  acc01 += a0 * b1;
            acc10 += a1 * b0;  acc11 += a1 * b1;
            acc20 += a2 * b0;  acc21 += a2 * b1;
            acc30 += a3 * b0;  acc31 += a3 * b1;
            acc40 += a4 * b0;  acc41 += a4 * b1;
            acc50 += a5 * b0;  acc51 += a5 * b1;
            acc60 += a6 * b0;  acc61 += a6 * b1;
            acc70 += a7 * b0;  acc71 += a7 * b1;
        }

        __syncthreads();
    }

    // Write back 8x2 outputs
    if (row0 < M && col0 < N) C[row0 * N + col0] = (float)acc00;
    if (row0 < M && col1 < N) C[row0 * N + col1] = (float)acc01;

    if (row1 < M && col0 < N) C[row1 * N + col0] = (float)acc10;
    if (row1 < M && col1 < N) C[row1 * N + col1] = (float)acc11;

    if (row2 < M && col0 < N) C[row2 * N + col0] = (float)acc20;
    if (row2 < M && col1 < N) C[row2 * N + col1] = (float)acc21;

    if (row3 < M && col0 < N) C[row3 * N + col0] = (float)acc30;
    if (row3 < M && col1 < N) C[row3 * N + col1] = (float)acc31;

    if (row4 < M && col0 < N) C[row4 * N + col0] = (float)acc40;
    if (row4 < M && col1 < N) C[row4 * N + col1] = (float)acc41;

    if (row5 < M && col0 < N) C[row5 * N + col0] = (float)acc50;
    if (row5 < M && col1 < N) C[row5 * N + col1] = (float)acc51;

    if (row6 < M && col0 < N) C[row6 * N + col0] = (float)acc60;
    if (row6 < M && col1 < N) C[row6 * N + col1] = (float)acc61;

    if (row7 < M && col0 < N) C[row7 * N + col0] = (float)acc70;
    if (row7 < M && col1 < N) C[row7 * N + col1] = (float)acc71;
}


/* Tiled matmul: 32x64 output tiles with 32x4 warp-aligned thread layout
   Each thread computes a 4x4 micro-tile (TM=TN=4), using tid-based remapping
   to decouple physical layout from logical output assignment.

   Physical layout:    32x4  = 128 threads (same as before)
   Logical grid:        8x16 = 128 threads (BM/TM x BN/TN = 32/4 x 64/4)
   Thread tile:         TM=4, TN=4 = 16 outputs per thread (same as before)

   The key idea:
     tid = ty*32 + tx                  (physical, 0..127)
     ltx = tid % 16,  lty = tid / 16  (logical, maps tid into 8x16 grid)
     row_start = lty * TM             (which 4-row block this thread owns)
     col_start = ltx * TN             (which 4-col block this thread owns)

   Global loads still use tid directly (coalesced, 32 consecutive threads
   in the physical x-dim hit consecutive addresses — unchanged from before).
*/
extern "C" __global__ void matmul_tiled_32x64_tm4_tn4(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    constexpr int BM = 32;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int TM = 4;
    constexpr int TN = 4;

    // Logical grid dimensions (derived, not hardcoded)
    // LROWS = BM/TM = 8,  LCOLS = BN/TN = 16
    // LROWS * LCOLS = 128 = total threads ✓
    constexpr int LROWS = BM / TM;   // 8
    constexpr int LCOLS = BN / TN;   // 16

    __shared__ float A_shared[BM][BK];
    __shared__ float B_shared[BK][BN];

    // --- Physical thread indices (used only for global loads) ---
    const int tx = threadIdx.x;              // 0..31
    const int ty = threadIdx.y;              // 0..3
    const int tid = ty * blockDim.x + tx;   // 0..127

    // --- Logical thread indices (used for compute + writeback) ---
    const int ltx = tid % LCOLS;   // 0..15  (which TN-column block)
    const int lty = tid / LCOLS;   // 0..7   (which TM-row block)

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // This thread's output region (4 rows x 4 cols in the CTA tile)
    const int row_start = block_row + lty * TM;
    const int col_start = block_col + ltx * TN;

    // 4x4 accumulator (TM x TN), all in registers
    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {

        // ----------------------------------------------------------
        // Load A tile: BM*BK = 32*32 = 1024 elems, 128 threads
        // => 8 elems/thread, identical to original
        // (physical tx/ty used here, not logical ltx/lty)
        // ----------------------------------------------------------
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = tid + i * 128;
            const int r = idx / BK;
            const int c = idx % BK;
            const int global_r = block_row + r;
            const int global_c = k0 + c;
            A_shared[r][c] = (global_r < M && global_c < K)
                ? A[global_r * K + global_c]
                : static_cast<float>(0);
        }

        // ----------------------------------------------------------
        // Load B tile: BK*BN = 32*64 = 2048 elems, 128 threads
        // => 16 elems/thread, identical to original
        // ----------------------------------------------------------
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const int idx = tid + i * 128;
            const int r = idx / BN;
            const int c = idx % BN;
            const int global_r = k0 + r;
            const int global_c = block_col + c;
            B_shared[r][c] = (global_r < K && global_c < N)
                ? B[global_r * N + global_c]
                : static_cast<float>(0);
        }

        __syncthreads();

        // ----------------------------------------------------------
        // Compute: each thread does its TM x TN = 4x4 micro-tile
        // lty/ltx used here, NOT ty/tx
        // ----------------------------------------------------------
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            // Load this thread's 4 A values (one column of its row-block)
            float a[TM];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a[i] = (float)A_shared[lty * TM + i][kk];

            // Load this thread's 4 B values (one row of its col-block)
            float b[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b[j] = (float)B_shared[kk][ltx * TN + j];

            // Outer product accumulation
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a[i] * b[j];
        }

        __syncthreads();
    }

    // ----------------------------------------------------------
    // Writeback: 4x4 block of C
    // ----------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const int global_r = row_start + i;
            const int global_c = col_start + j;
            if (global_r < M && global_c < N)
                C[global_r * N + global_c] = (float)acc[i][j];
        }
    }
}


