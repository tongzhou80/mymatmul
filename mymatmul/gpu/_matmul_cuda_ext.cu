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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "matmul_naive_ijk", [&] {
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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "matmul_naive_ijk_jx", [&] {
        matmul_naive_ijk_2d_grid_jx<scalar_t><<<blocks, threads>>>(
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
}
