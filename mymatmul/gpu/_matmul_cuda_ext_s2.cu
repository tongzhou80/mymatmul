#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

/*
 * Stage 2: Shared Memory Tiling
 * ==============================
 * Educational kernels illustrating the tradeoff between arithmetic intensity
 * and occupancy for two tile / thread-layout configurations.
 *
 * Data type: bfloat16 input/output, float32 accumulation.
 * Shared memory holds bf16 tiles — half the footprint of float32 tiles —
 * which matters for the occupancy calculation.
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
 *   SMEM: (8*32 + 32*32) * 2B   = 2560 B
 *
 * Case 2: BM=16, BN=32, BK=32, threads=32x8 → TM=2, TN=1
 *   Global A.I. = 16*32/(16+32) = 10.67
 *   Shared A.I. = 2*1/(2+1)     = 0.67  (same as case 1)
 *   SMEM: (16*32 + 32*32) * 2B  = 3072 B
 *
 * Register and shared-memory usage can be inspected by compiling with:
 *   nvcc -arch=sm_89 -O3 -Xptxas -v
 * which is enabled in setup.py for this extension.
 *
 * Occupancy notes (RTX 4090, SM 8.9, 65536 regs / 48 KB SMEM per SM):
 *   Case 1 (128 threads, 4 warps/block):
 *     2560 B SMEM → up to 18 blocks fit on SMEM budget → 72 warps
 *
 *   Case 2 (256 threads, 8 warps/block):
 *     3072 B SMEM → up to 15 blocks fit → 120 warps
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
__global__ void smem_tiled_bm8_bn32_bk32_threads32x4(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    constexpr int BM = 8;
    constexpr int BN = 32;
    constexpr int BK = 32;
    constexpr int THREADS = 32 * 4;   // 128

    // bf16 shared memory: 2 B per element
    __shared__ __nv_bfloat16 A_shared[BM][BK];   // 8  * 32 * 2B = 512 B
    __shared__ __nv_bfloat16 B_shared[BK][BN];   // 32 * 32 * 2B = 2048 B
                                                   // total = 2560 B

    const int tx = threadIdx.x;   // 0..31 → N dimension
    const int ty = threadIdx.y;   // 0..3  → M dimension (row pair index)
    const int tid = ty * blockDim.x + tx;   // flat id [0, 128)

    const int col      = blockIdx.x * BN + tx;
    const int row_base = blockIdx.y * BM + ty * 2;

    // float32 accumulators for numerical stability
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
                             : __float2bfloat16(0.0f);
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
                             : __float2bfloat16(0.0f);
        }

        __syncthreads();

        // --- Compute: TM=2 rows, TN=1 col, accumulate over BK ---
        // Cast bf16 → float before FMA to keep accumulators in float32
        for (int kk = 0; kk < BK; kk++) {
            const float b_val = __bfloat162float(B_shared[kk][tx]);
            acc0 += __bfloat162float(A_shared[ty * 2    ][kk]) * b_val;
            acc1 += __bfloat162float(A_shared[ty * 2 + 1][kk]) * b_val;
        }

        __syncthreads();
    }

    // --- Write back: float32 → bf16 ---
    if (col < N) {
        if (row_base     < M) C[ row_base      * N + col] = __float2bfloat16(acc0);
        if (row_base + 1 < M) C[(row_base + 1) * N + col] = __float2bfloat16(acc1);
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
__global__ void smem_tiled_bm16_bn32_bk32_threads32x8(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    constexpr int BM = 16;
    constexpr int BN = 32;
    constexpr int BK = 32;
    constexpr int THREADS = 32 * 8;   // 256

    __shared__ __nv_bfloat16 A_shared[BM][BK];   // 16 * 32 * 2B = 1024 B
    __shared__ __nv_bfloat16 B_shared[BK][BN];   // 32 * 32 * 2B = 2048 B
                                                   // total = 3072 B

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
                             : __float2bfloat16(0.0f);
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
                             : __float2bfloat16(0.0f);
        }

        __syncthreads();

        // --- Compute ---
        for (int kk = 0; kk < BK; kk++) {
            const float b_val = __bfloat162float(B_shared[kk][tx]);
            acc0 += __bfloat162float(A_shared[ty * 2    ][kk]) * b_val;
            acc1 += __bfloat162float(A_shared[ty * 2 + 1][kk]) * b_val;
        }

        __syncthreads();
    }

    // --- Write back ---
    if (col < N) {
        if (row_base     < M) C[ row_base      * N + col] = __float2bfloat16(acc0);
        if (row_base + 1 < M) C[(row_base + 1) * N + col] = __float2bfloat16(acc1);
    }
}


// ---------------------------------------------------------------------------
// PyTorch dispatch wrappers
// ---------------------------------------------------------------------------

torch::Tensor matmul_s2_bm8_bn32_bk32_threads32x4(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "Only bfloat16 is supported in Stage 2 kernels");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    constexpr int BM = 8, BN = 32;
    dim3 threads(32, 4);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    smem_tiled_bm8_bn32_bk32_threads32x4<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, K, N
    );

    return C;
}

torch::Tensor matmul_s2_bm16_bn32_bk32_threads32x8(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "Only bfloat16 is supported in Stage 2 kernels");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    constexpr int BM = 16, BN = 32;
    dim3 threads(32, 8);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    smem_tiled_bm16_bn32_bk32_threads32x8<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, K, N
    );

    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Stage 2: Shared Memory Tiling — occupancy vs arithmetic intensity study";

    m.def("matmul_s2_bm8_bn32_bk32_threads32x4",
          &matmul_s2_bm8_bn32_bk32_threads32x4,
          "Case 1: BM=8,BN=32,BK=32, 32x4 threads, TM=2 TN=1 — global AI=6.4, smem AI=0.67");

    m.def("matmul_s2_bm16_bn32_bk32_threads32x8",
          &matmul_s2_bm16_bn32_bk32_threads32x8,
          "Case 2: BM=16,BN=32,BK=32, 32x8 threads, TM=2 TN=1 — global AI=10.67, smem AI=0.67");
}
