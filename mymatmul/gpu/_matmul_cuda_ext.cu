#include <torch/extension.h>
#include <cuda_runtime.h>

/* Naive 2D grid: each thread (i,j) computes the full k reduction
   Uses float32 accumulator for better numerical stability with float16 inputs */
template <typename scalar_t>
__global__ void matmul_naive_ijk_2d_grid(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    int M, int K, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        float sum = 0.0f;  // Accumulate in float32 for stability
        for (int k = 0; k < K; k++) {
            sum += (float)A[i*K + k] * (float)B[k*N + j];
        }
        C[i*N + j] = (scalar_t)sum;  // Convert back to input dtype
    }
}

torch::Tensor matmul_cuda_naive_ijk(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == B.dtype(), "A and B must have the same dtype");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(16, 16);  // 16x16 = 256 threads per block
    dim3 blocks((M + 15) / 16, (N + 15) / 16);

    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, A.scalar_type(), "matmul_naive_ijk", [&] {
        matmul_naive_ijk_2d_grid<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N
        );
    });

    return C;
}

/* Naive 2D grid with improved thread mapping: j to x (fastest-changing in row-major)
   Maps j (column, fastest-changing in C layout) to x (fastest CUDA dimension) */
template <typename scalar_t>
__global__ void matmul_naive_ijk_2d_grid_jx(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    int M, int K, int N
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // x dimension (changes fastest)
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // y dimension

    if (i < M && j < N) {
        float sum = 0.0f;  // Accumulate in float32 for stability
        for (int k = 0; k < K; k++) {
            sum += (float)A[i*K + k] * (float)B[k*N + j];
        }
        C[i*N + j] = (scalar_t)sum;  // Convert back to input dtype
    }
}

torch::Tensor matmul_cuda_naive_ijk_jx(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == B.dtype(), "A and B must have the same dtype");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(16, 16);  // 16x16 = 256 threads per block
    dim3 blocks((N + 15) / 16, (M + 15) / 16);  // Swapped: x for N, y for M

    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, A.scalar_type(), "matmul_naive_ijk_jx", [&] {
        matmul_naive_ijk_2d_grid_jx<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N
        );
    });

    return C;
}

/* Tiled matmul: 32x32 tiles with shared memory
   Each thread (tx, ty) computes one element C[i,j] where:
   - i = blockIdx.y * 32 + ty
   - j = blockIdx.x * 32 + tx

   For each K-tile, load 32x32 slice of A and B into shared memory,
   then accumulate 32 elements per iteration. */
template <typename scalar_t>
__global__ void matmul_tiled_32x32(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    int M, int K, int N
) {
    const int TILE_SIZE = 32;
    __shared__ scalar_t A_shared[32][32];
    __shared__ scalar_t B_shared[32][32];

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
        C[i * N + j] = (scalar_t)sum;
    }
}

torch::Tensor matmul_cuda_tiled_32x32(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == B.dtype(), "A and B must have the same dtype");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(32, 32);  // 32x32 = 1024 threads per block
    dim3 blocks((N + 31) / 32, (M + 31) / 32);

    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, A.scalar_type(), "matmul_tiled_32x32", [&] {
        matmul_tiled_32x32<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N
        );
    });

    return C;
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
template <typename scalar_t>
__global__ void matmul_tiled_32x32_16x16_threads(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    int M, int K, int N
) {
    constexpr int BM = 32;   // C tile height
    constexpr int BN = 32;   // C tile width
    constexpr int BK = 32;   // K tile depth
    constexpr int TM = 2;    // per-thread rows
    constexpr int TN = 2;    // per-thread cols

    __shared__ scalar_t A_shared[BM][BK];
    __shared__ scalar_t B_shared[BK][BN];

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
            A_shared[ty][tx] = static_cast<scalar_t>(0);
        }

        // A_shared[ty][tx+16] = A[row0][k0 + tx + 16]
        if (row0 < M && (k0 + tx + 16) < K) {
            A_shared[ty][tx + 16] = A[row0 * K + (k0 + tx + 16)];
        } else {
            A_shared[ty][tx + 16] = static_cast<scalar_t>(0);
        }

        // A_shared[ty+16][tx] = A[row1][k0 + tx]
        if (row1 < M && (k0 + tx) < K) {
            A_shared[ty + 16][tx] = A[row1 * K + (k0 + tx)];
        } else {
            A_shared[ty + 16][tx] = static_cast<scalar_t>(0);
        }

        // A_shared[ty+16][tx+16] = A[row1][k0 + tx + 16]
        if (row1 < M && (k0 + tx + 16) < K) {
            A_shared[ty + 16][tx + 16] = A[row1 * K + (k0 + tx + 16)];
        } else {
            A_shared[ty + 16][tx + 16] = static_cast<scalar_t>(0);
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
            B_shared[ty][tx] = static_cast<scalar_t>(0);
        }

        // B_shared[ty][tx+16] = B[k0 + ty][col1]
        if ((k0 + ty) < K && col1 < N) {
            B_shared[ty][tx + 16] = B[(k0 + ty) * N + col1];
        } else {
            B_shared[ty][tx + 16] = static_cast<scalar_t>(0);
        }

        // B_shared[ty+16][tx] = B[k0 + ty + 16][col0]
        if ((k0 + ty + 16) < K && col0 < N) {
            B_shared[ty + 16][tx] = B[(k0 + ty + 16) * N + col0];
        } else {
            B_shared[ty + 16][tx] = static_cast<scalar_t>(0);
        }

        // B_shared[ty+16][tx+16] = B[k0 + ty + 16][col1]
        if ((k0 + ty + 16) < K && col1 < N) {
            B_shared[ty + 16][tx + 16] = B[(k0 + ty + 16) * N + col1];
        } else {
            B_shared[ty + 16][tx + 16] = static_cast<scalar_t>(0);
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
        C[row0 * N + col0] = (scalar_t)acc00;
    }
    if (row0 < M && col1 < N) {
        C[row0 * N + col1] = (scalar_t)acc01;
    }
    if (row1 < M && col0 < N) {
        C[row1 * N + col0] = (scalar_t)acc10;
    }
    if (row1 < M && col1 < N) {
        C[row1 * N + col1] = (scalar_t)acc11;
    }
}

torch::Tensor matmul_cuda_tiled_32x32_16x16_threads(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == B.dtype(), "A and B must have the same dtype");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(16, 16);                  // 256 threads
    dim3 blocks((N + 31) / 32, (M + 31) / 32);   // each CTA computes a 32x32 tile

    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf,
        torch::kBFloat16,
        A.scalar_type(),
        "matmul_tiled_32x32_16x16_threads",
        [&] {
            matmul_tiled_32x32_16x16_threads<scalar_t><<<blocks, threads>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                M, K, N
            );
        }
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda_naive_ijk", &matmul_cuda_naive_ijk, "Naive CUDA matmul: 2D grid (i->x, j->y), each thread (i,j) computes full k");
    m.def("matmul_cuda_naive_ijk_jx", &matmul_cuda_naive_ijk_jx, "Naive CUDA matmul: 2D grid (j->x, i->y), each thread (i,j) computes full k");
    m.def("matmul_cuda_tiled_32x32", &matmul_cuda_tiled_32x32, "Tiled CUDA matmul: 32x32 tiles with shared memory");
    m.def("matmul_cuda_tiled_32x32_16x16_threads", &matmul_cuda_tiled_32x32_16x16_threads, "Optimized tiled CUDA matmul: 32x32 tiles, 16x16 threads, each thread computes 2x2 micro-tile");
}
