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

/* Tiled matmul: 32x32 tiles computed by 16x16 threads
   Each thread (tx, ty) computes FOUR elements C[i,j] arranged in a 2x2 grid */
template <typename scalar_t>
__global__ void matmul_tiled_32x32_threads_16x16(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    int M, int K, int N
) {
    const int TILE_SIZE = 32;
    const int THREAD_DIM = 16;

    // Shared memory still holds the full 32x32 tiles
    __shared__ scalar_t A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t B_shared[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Top-left corner of the 32x32 tile in the global C matrix
    int row_start = blockIdx.y * TILE_SIZE;
    int col_start = blockIdx.x * TILE_SIZE;

    // Each thread computes a 2x2 sub-grid of the output.
    // We store the 4 accumulators in registers.
    float sum[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    // Iterate over K dimension in 32-element tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {

        // 1. COLLABORATIVE LOAD INTO SHARED MEMORY
        // 256 threads must load 1024 elements. Each thread loads a 2x2 grid.
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                // Calculate global indices for A and B
                int a_row = row_start + ty + i * THREAD_DIM;
                int a_col = k_tile + tx + j * THREAD_DIM;

                if (a_row < M && a_col < K) {
                    A_shared[ty + i * THREAD_DIM][tx + j * THREAD_DIM] = A[a_row * K + a_col];
                } else {
                    A_shared[ty + i * THREAD_DIM][tx + j * THREAD_DIM] = 0.0f;
                }

                int b_row = k_tile + ty + i * THREAD_DIM;
                int b_col = col_start + tx + j * THREAD_DIM;

                if (b_row < K && b_col < N) {
                    B_shared[ty + i * THREAD_DIM][tx + j * THREAD_DIM] = B[b_row * N + b_col];
                } else {
                    B_shared[ty + i * THREAD_DIM][tx + j * THREAD_DIM] = 0.0f;
                }
            }
        }

        __syncthreads(); // Wait for all 1024 elements to be loaded

        // 2. COMPUTE MULTIPLY-ADDS
        #pragma unroll
        for (int kk = 0; kk < TILE_SIZE; ++kk) {
            // Load from shared memory into local registers for reuse
            float reg_A[2];
            float reg_B[2];

            reg_A[0] = (float)A_shared[ty][kk];
            reg_A[1] = (float)A_shared[ty + THREAD_DIM][kk];

            reg_B[0] = (float)B_shared[kk][tx];
            reg_B[1] = (float)B_shared[kk][tx + THREAD_DIM];

            // Perform 4 FMAs (Fused Multiply-Adds) using just 4 register loads
            sum[0][0] += reg_A[0] * reg_B[0]; // Top-Left
            sum[0][1] += reg_A[0] * reg_B[1]; // Top-Right
            sum[1][0] += reg_A[1] * reg_B[0]; // Bottom-Left
            sum[1][1] += reg_A[1] * reg_B[1]; // Bottom-Right
        }

        __syncthreads(); // Wait for compute to finish before overwriting shared memory
    }

    // 3. WRITE RESULTS TO GLOBAL MEMORY
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int c_row = row_start + ty + i * THREAD_DIM;
            int c_col = col_start + tx + j * THREAD_DIM;

            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = (scalar_t)sum[i][j];
            }
        }
    }
}

torch::Tensor matmul_cuda_tiled_32x32_opt(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == B.dtype(), "A and B must have the same dtype");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // 16x16 threads per block (256 threads total)
    dim3 threads(16, 16);

    // Grid size stays the same because each block still processes a 32x32 tile of C!
    dim3 blocks((N + 31) / 32, (M + 31) / 32);

    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, A.scalar_type(), "matmul_tiled_32x32_opt", [&] {
        matmul_tiled_32x32_threads_16x16<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N
        );
    });

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda_naive_ijk", &matmul_cuda_naive_ijk, "Naive CUDA matmul: 2D grid (i->x, j->y), each thread (i,j) computes full k");
    m.def("matmul_cuda_naive_ijk_jx", &matmul_cuda_naive_ijk_jx, "Naive CUDA matmul: 2D grid (j->x, i->y), each thread (i,j) computes full k");
    m.def("matmul_cuda_tiled_32x32", &matmul_cuda_tiled_32x32, "Tiled CUDA matmul: 32x32 tiles with shared memory");
    m.def("matmul_cuda_tiled_32x32_opt", &matmul_cuda_tiled_32x32_opt, "Optimized tiled CUDA matmul: 32x32 tiles, 16x16 threads, each thread computes 4 elements");
}
