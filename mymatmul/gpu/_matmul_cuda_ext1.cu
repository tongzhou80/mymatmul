#include <torch/extension.h>
#include <cuda_runtime.h>

// Dispatch for float, half, bfloat16 only — excludes double to avoid SMEM
// overflow when instantiated for large tiles (e.g. BM=BN=128 at fp64 needs
// 64KB shared memory which exceeds the 48KB SM limit).
#define AT_DISPATCH_FLOAT_HALF_BF16(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME,                   \
        AT_DISPATCH_CASE(at::ScalarType::Float,      __VA_ARGS__) \
        AT_DISPATCH_CASE(at::ScalarType::Half,       __VA_ARGS__) \
        AT_DISPATCH_CASE(at::ScalarType::BFloat16,   __VA_ARGS__) \
    )

/*
 * Stage 3: Square Tile Sizes via Tx-Remap — single templated kernel
 *
 * Template parameters:
 *   BM, BN : CTA output tile dimensions
 *   BK     : K-tile depth (sweep variable; try 16 and 32)
 *   TM, TN : per-thread micro-tile dimensions
 *
 * Physical thread layout is always 32 x (BM*BN / (TM*TN) / 32), i.e. warp-aligned x-dim.
 * Logical layout LROWS=BM/TM, LCOLS=BN/TN is decoupled via tid-based remap.
 *
 * Tile loading loops use #pragma unroll 8 for good ILP without excessive register pressure.
 * The compute loop (BK iters) is fully unrolled.
 */
template <typename scalar_t, int BM, int BN, int BK, int TM, int TN, int UNROLL>
__global__ void matmul_s3(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int K, int N
) {
    constexpr int THREADS = (BM / TM) * (BN / TN);   // total threads per CTA
    constexpr int LROWS   = BM / TM;
    constexpr int LCOLS   = BN / TN;
    constexpr int A_ITERS = BM * BK / THREADS;        // A tile elems per thread
    constexpr int B_ITERS = BK * BN / THREADS;        // B tile elems per thread

    __shared__ scalar_t A_shared[BM][BK];
    __shared__ scalar_t B_shared[BK][BN];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // Logical remap: decouple output assignment from physical layout
    const int ltx = tid % LCOLS;
    const int lty = tid / LCOLS;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int row_start = block_row + lty * TM;
    const int col_start = block_col + ltx * TN;

    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Cooperatively load A tile (physical tid used for coalescing)
        #pragma unroll UNROLL
        for (int i = 0; i < A_ITERS; i++) {
            const int idx = tid + i * THREADS;
            const int r = idx / BK, c = idx % BK;
            A_shared[r][c] = (block_row + r < M && k0 + c < K)
                ? A[(block_row + r) * K + k0 + c]
                : static_cast<scalar_t>(0);
        }
        // Cooperatively load B tile
        #pragma unroll UNROLL
        for (int i = 0; i < B_ITERS; i++) {
            const int idx = tid + i * THREADS;
            const int r = idx / BN, c = idx % BN;
            B_shared[r][c] = (k0 + r < K && block_col + c < N)
                ? B[(k0 + r) * N + block_col + c]
                : static_cast<scalar_t>(0);
        }
        __syncthreads();

        // Compute TM x TN micro-tile (logical ltx/lty used here)
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = (float)A_shared[lty * TM + i][kk];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = (float)B_shared[kk][ltx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    // Write back TM x TN outputs
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const int gr = row_start + i, gc = col_start + j;
            if (gr < M && gc < N) C[gr * N + gc] = (scalar_t)acc[i][j];
        }
}

// ---- Launch wrapper macro ----
// Physical threads: x-dim=32 (warp-aligned), y-dim=THREADS/32
#define MAKE_LAUNCHER(NAME, BM, BN, BK, TM, TN, UNROLL)                            \
torch::Tensor NAME(torch::Tensor A, torch::Tensor B) {                              \
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");         \
    TORCH_CHECK(A.dtype() == B.dtype(), "Dtype mismatch");                          \
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");                \
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");             \
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Must be contiguous");      \
    constexpr int THREADS = (BM / TM) * (BN / TN);                                 \
    const int M = A.size(0), K = A.size(1), N = B.size(1);                         \
    auto C = torch::zeros({M, N}, A.options());                                     \
    dim3 threads(32, THREADS / 32);                                                 \
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);                            \
    AT_DISPATCH_FLOAT_HALF_BF16(A.scalar_type(), #NAME, [&] {                      \
        matmul_s3<scalar_t, BM, BN, BK, TM, TN, UNROLL><<<blocks, threads>>>(      \
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(),                         \
            C.data_ptr<scalar_t>(), M, K, N);                                       \
    });                                                                             \
    return C;                                                                       \
}

// ---- BK=32, unroll=8 (best found so far) ----
//                               NAME                               BM   BN  BK  TM  TN  UNROLL
MAKE_LAUNCHER(matmul_cuda_s3_tm4_tn4_bm32_bn64_bk32_u8,            32,  64, 32,  4,  4,  8)
MAKE_LAUNCHER(matmul_cuda_s3_tm4_tn4_bm64_bn64_bk32_u8,            64,  64, 32,  4,  4,  8)
MAKE_LAUNCHER(matmul_cuda_s3_tm8_tn4_bm64_bn64_bk32_u8,            64,  64, 32,  8,  4,  8)
MAKE_LAUNCHER(matmul_cuda_s3_tm8_tn8_bm128_bn64_bk32_u8,          128,  64, 32,  8,  8,  8)
MAKE_LAUNCHER(matmul_cuda_s3_tm8_tn8_bm128_bn128_bk32_u8,         128, 128, 32,  8,  8,  8)

// ---- BK=16, unroll=1,2,4,8 ----
MAKE_LAUNCHER(matmul_cuda_s3_tm4_tn4_bm32_bn64_bk16_u1,            32,  64, 16,  4,  4,  1)
MAKE_LAUNCHER(matmul_cuda_s3_tm4_tn4_bm64_bn64_bk16_u1,            64,  64, 16,  4,  4,  1)
MAKE_LAUNCHER(matmul_cuda_s3_tm8_tn4_bm64_bn64_bk16_u1,            64,  64, 16,  8,  4,  1)
MAKE_LAUNCHER(matmul_cuda_s3_tm8_tn8_bm128_bn64_bk16_u1,          128,  64, 16,  8,  8,  1)
MAKE_LAUNCHER(matmul_cuda_s3_tm8_tn8_bm128_bn128_bk16_u1,         128, 128, 16,  8,  8,  1)

MAKE_LAUNCHER(matmul_cuda_s3_tm4_tn4_bm32_bn64_bk16_u2,            32,  64, 16,  4,  4,  2)
MAKE_LAUNCHER(matmul_cuda_s3_tm4_tn4_bm64_bn64_bk16_u2,            64,  64, 16,  4,  4,  2)
MAKE_LAUNCHER(matmul_cuda_s3_tm8_tn4_bm64_bn64_bk16_u2,            64,  64, 16,  8,  4,  2)
MAKE_LAUNCHER(matmul_cuda_s3_tm8_tn8_bm128_bn64_bk16_u2,          128,  64, 16,  8,  8,  2)
MAKE_LAUNCHER(matmul_cuda_s3_tm8_tn8_bm128_bn128_bk16_u2,         128, 128, 16,  8,  8,  2)

MAKE_LAUNCHER(matmul_cuda_s3_tm4_tn4_bm32_bn64_bk16_u4,            32,  64, 16,  4,  4,  4)
MAKE_LAUNCHER(matmul_cuda_s3_tm4_tn4_bm64_bn64_bk16_u4,            64,  64, 16,  4,  4,  4)
MAKE_LAUNCHER(matmul_cuda_s3_tm8_tn4_bm64_bn64_bk16_u4,            64,  64, 16,  8,  4,  4)
MAKE_LAUNCHER(matmul_cuda_s3_tm8_tn8_bm128_bn64_bk16_u4,          128,  64, 16,  8,  8,  4)
MAKE_LAUNCHER(matmul_cuda_s3_tm8_tn8_bm128_bn128_bk16_u4,         128, 128, 16,  8,  8,  4)

MAKE_LAUNCHER(matmul_cuda_s3_tm4_tn4_bm32_bn64_bk16_u8,            32,  64, 16,  4,  4,  8)
MAKE_LAUNCHER(matmul_cuda_s3_tm4_tn4_bm64_bn64_bk16_u8,            64,  64, 16,  4,  4,  8)
MAKE_LAUNCHER(matmul_cuda_s3_tm8_tn4_bm64_bn64_bk16_u8,            64,  64, 16,  8,  4,  8)
MAKE_LAUNCHER(matmul_cuda_s3_tm8_tn8_bm128_bn64_bk16_u8,          128,  64, 16,  8,  8,  8)
MAKE_LAUNCHER(matmul_cuda_s3_tm8_tn8_bm128_bn128_bk16_u8,         128, 128, 16,  8,  8,  8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // BK=32, unroll=8
    m.def("matmul_cuda_s3_tm4_tn4_bm32_bn64_bk32_u8",   &matmul_cuda_s3_tm4_tn4_bm32_bn64_bk32_u8);
    m.def("matmul_cuda_s3_tm4_tn4_bm64_bn64_bk32_u8",   &matmul_cuda_s3_tm4_tn4_bm64_bn64_bk32_u8);
    m.def("matmul_cuda_s3_tm8_tn4_bm64_bn64_bk32_u8",   &matmul_cuda_s3_tm8_tn4_bm64_bn64_bk32_u8);
    m.def("matmul_cuda_s3_tm8_tn8_bm128_bn64_bk32_u8",  &matmul_cuda_s3_tm8_tn8_bm128_bn64_bk32_u8);
    m.def("matmul_cuda_s3_tm8_tn8_bm128_bn128_bk32_u8", &matmul_cuda_s3_tm8_tn8_bm128_bn128_bk32_u8);
    // BK=16, unroll=1
    m.def("matmul_cuda_s3_tm4_tn4_bm32_bn64_bk16_u1",   &matmul_cuda_s3_tm4_tn4_bm32_bn64_bk16_u1);
    m.def("matmul_cuda_s3_tm4_tn4_bm64_bn64_bk16_u1",   &matmul_cuda_s3_tm4_tn4_bm64_bn64_bk16_u1);
    m.def("matmul_cuda_s3_tm8_tn4_bm64_bn64_bk16_u1",   &matmul_cuda_s3_tm8_tn4_bm64_bn64_bk16_u1);
    m.def("matmul_cuda_s3_tm8_tn8_bm128_bn64_bk16_u1",  &matmul_cuda_s3_tm8_tn8_bm128_bn64_bk16_u1);
    m.def("matmul_cuda_s3_tm8_tn8_bm128_bn128_bk16_u1", &matmul_cuda_s3_tm8_tn8_bm128_bn128_bk16_u1);
    // BK=16, unroll=2
    m.def("matmul_cuda_s3_tm4_tn4_bm32_bn64_bk16_u2",   &matmul_cuda_s3_tm4_tn4_bm32_bn64_bk16_u2);
    m.def("matmul_cuda_s3_tm4_tn4_bm64_bn64_bk16_u2",   &matmul_cuda_s3_tm4_tn4_bm64_bn64_bk16_u2);
    m.def("matmul_cuda_s3_tm8_tn4_bm64_bn64_bk16_u2",   &matmul_cuda_s3_tm8_tn4_bm64_bn64_bk16_u2);
    m.def("matmul_cuda_s3_tm8_tn8_bm128_bn64_bk16_u2",  &matmul_cuda_s3_tm8_tn8_bm128_bn64_bk16_u2);
    m.def("matmul_cuda_s3_tm8_tn8_bm128_bn128_bk16_u2", &matmul_cuda_s3_tm8_tn8_bm128_bn128_bk16_u2);
    // BK=16, unroll=4
    m.def("matmul_cuda_s3_tm4_tn4_bm32_bn64_bk16_u4",   &matmul_cuda_s3_tm4_tn4_bm32_bn64_bk16_u4);
    m.def("matmul_cuda_s3_tm4_tn4_bm64_bn64_bk16_u4",   &matmul_cuda_s3_tm4_tn4_bm64_bn64_bk16_u4);
    m.def("matmul_cuda_s3_tm8_tn4_bm64_bn64_bk16_u4",   &matmul_cuda_s3_tm8_tn4_bm64_bn64_bk16_u4);
    m.def("matmul_cuda_s3_tm8_tn8_bm128_bn64_bk16_u4",  &matmul_cuda_s3_tm8_tn8_bm128_bn64_bk16_u4);
    m.def("matmul_cuda_s3_tm8_tn8_bm128_bn128_bk16_u4", &matmul_cuda_s3_tm8_tn8_bm128_bn128_bk16_u4);
    // BK=16, unroll=8
    m.def("matmul_cuda_s3_tm4_tn4_bm32_bn64_bk16_u8",   &matmul_cuda_s3_tm4_tn4_bm32_bn64_bk16_u8);
    m.def("matmul_cuda_s3_tm4_tn4_bm64_bn64_bk16_u8",   &matmul_cuda_s3_tm4_tn4_bm64_bn64_bk16_u8);
    m.def("matmul_cuda_s3_tm8_tn4_bm64_bn64_bk16_u8",   &matmul_cuda_s3_tm8_tn4_bm64_bn64_bk16_u8);
    m.def("matmul_cuda_s3_tm8_tn8_bm128_bn64_bk16_u8",  &matmul_cuda_s3_tm8_tn8_bm128_bn64_bk16_u8);
    m.def("matmul_cuda_s3_tm8_tn8_bm128_bn128_bk16_u8", &matmul_cuda_s3_tm8_tn8_bm128_bn128_bk16_u8);
}
