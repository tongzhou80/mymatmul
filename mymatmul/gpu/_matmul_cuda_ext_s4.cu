#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

// Dispatch for float, half, bfloat16 only
#define AT_DISPATCH_FLOAT_HALF_BF16(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME,                   \
        AT_DISPATCH_CASE(at::ScalarType::Float,      __VA_ARGS__) \
        AT_DISPATCH_CASE(at::ScalarType::Half,       __VA_ARGS__) \
        AT_DISPATCH_CASE(at::ScalarType::BFloat16,   __VA_ARGS__) \
    )

/*
 * Stage 4: Double-buffered matmul with async global→shared copies (cp.async)
 *
 * Key changes vs Stage 3:
 *   - Shared memory doubled: A_shared[2][BM][BK], B_shared[2][BK][BN]
 *   - Global→shared loads use __pipeline_memcpy_async (cp.async, SM 8.0+)
 *   - UNROLL removed: load loops are small enough that the compiler handles them
 *   - Adaptive load granularity per tile (A_LOAD_BYTES, B_LOAD_BYTES):
 *       choose largest of {4, 8, 16} bytes that divides evenly per thread
 *   - A trailing __syncthreads() after COMPUTE_TILE is required: buffer nxt
 *     at iteration k+1 is the same as buffer cur at iteration k, so we must
 *     ensure all threads finish reading cur before the next ISSUE_TILE writes to it
 *
 * Constraint: M, N, K must be multiples of BM, BN, BK respectively.
 */
template <typename scalar_t, int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_s4(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int K, int N
) {
    constexpr int THREADS  = (BM / TM) * (BN / TN);
    constexpr int LCOLS    = BN / TN;

    // Adaptive load size: largest of {4, 8, 16} bytes that evenly divides per-thread load.
    // For a tile of total_bytes = BM*BK*sizeof(scalar_t), each thread loads
    // total_bytes/THREADS bytes. We pick the largest power-of-two load <= that and <= 16.
    constexpr int A_THREAD_BYTES = BM * BK * (int)sizeof(scalar_t) / THREADS;
    constexpr int A_LOAD_BYTES   = (A_THREAD_BYTES >= 16) ? 16 : (A_THREAD_BYTES >= 8) ? 8 : 4;
    constexpr int A_ELEM         = A_LOAD_BYTES / (int)sizeof(scalar_t);   // elements per copy
    constexpr int A_GROUPS       = BM * BK / A_ELEM / THREADS;             // copies per thread

    constexpr int B_THREAD_BYTES = BK * BN * (int)sizeof(scalar_t) / THREADS;
    constexpr int B_LOAD_BYTES   = (B_THREAD_BYTES >= 16) ? 16 : (B_THREAD_BYTES >= 8) ? 8 : 4;
    constexpr int B_ELEM         = B_LOAD_BYTES / (int)sizeof(scalar_t);
    constexpr int B_GROUPS       = BK * BN / B_ELEM / THREADS;

    // Double-buffered shared memory
    __shared__ scalar_t A_shared[2][BM][BK];
    __shared__ scalar_t B_shared[2][BK][BN];

    const int tx  = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // Logical remap for output assignment
    const int ltx = tid % LCOLS, lty = tid / LCOLS;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int row_start = block_row + lty * TM;
    const int col_start = block_col + ltx * TN;

    float acc[TM][TN] = {};

    // ---- Async tile loader and compute macros ----
    // ISSUE_TILE: each thread issues A_GROUPS cp.async for A and B_GROUPS for B,
    //   then commits the group. Load size is A_LOAD_BYTES / B_LOAD_BYTES bytes each.
    //   The source/destination addresses are derived from the per-thread group index.
    //
    //   A tile (BM×BK, row-major): group g → elems [g*A_ELEM .. g*A_ELEM+A_ELEM-1]
    //     r = (g * A_ELEM) / BK,  c = (g * A_ELEM) % BK
    //   B tile (BK×BN, row-major): group g → elems [g*B_ELEM .. g*B_ELEM+B_ELEM-1]
    //     r = (g * B_ELEM) / BN,  c = (g * B_ELEM) % BN
    //
    // Alignment: A_ELEM divides BK (power-of-2 tile dims), B_ELEM divides BN,
    //   and block_row/col/k0 are multiples of BM/BN/BK → always aligned. ✓

#define ISSUE_TILE(k0_, buf_)                                                           \
    do {                                                                                \
        _Pragma("unroll")                                                               \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                        \
            const int _g = tid + _i * THREADS;                                         \
            const int _r = (_g * A_ELEM) / BK, _c = (_g * A_ELEM) % BK;              \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_c],                         \
                                    &A[(block_row + _r) * K + (k0_) + _c],             \
                                    A_LOAD_BYTES);                                      \
        }                                                                               \
        _Pragma("unroll")                                                               \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                        \
            const int _g = tid + _i * THREADS;                                         \
            const int _r = (_g * B_ELEM) / BN, _c = (_g * B_ELEM) % BN;              \
            __pipeline_memcpy_async(&B_shared[(buf_)][_r][_c],                         \
                                    &B[((k0_) + _r) * N + block_col + _c],             \
                                    B_LOAD_BYTES);                                      \
        }                                                                               \
        __pipeline_commit();                                                            \
    } while (0)

#define COMPUTE_TILE(buf_)                                                              \
    do {                                                                                \
        _Pragma("unroll")                                                               \
        for (int _kk = 0; _kk < BK; _kk++) {                                           \
            float _a[TM], _b[TN];                                                       \
            _Pragma("unroll")                                                           \
            for (int _i = 0; _i < TM; _i++)                                            \
                _a[_i] = (float)A_shared[(buf_)][lty * TM + _i][_kk];                  \
            _Pragma("unroll")                                                           \
            for (int _j = 0; _j < TN; _j++)                                            \
                _b[_j] = (float)B_shared[(buf_)][_kk][ltx * TN + _j];                  \
            _Pragma("unroll")                                                           \
            for (int _i = 0; _i < TM; _i++)                                            \
                _Pragma("unroll")                                                       \
                for (int _j = 0; _j < TN; _j++)                                        \
                    acc[_i][_j] += _a[_i] * _b[_j];                                    \
        }                                                                               \
    } while (0)

    const int num_tiles = K / BK;

    // Prefetch tile 0 → buffer 0
    ISSUE_TILE(0, 0);

    // Main loop: prefetch tile k+1 into nxt, wait for tile k in cur, compute tile k.
    // Trailing __syncthreads() after COMPUTE_TILE is required: nxt_{k+1} = cur_k,
    // so we must ensure all threads finish reading cur before the next ISSUE_TILE.
    for (int k_iter = 0; k_iter < num_tiles - 1; k_iter++) {
        const int cur = k_iter & 1;
        const int nxt = 1 - cur;
        ISSUE_TILE((k_iter + 1) * BK, nxt);
        __pipeline_wait_prior(1);   // wait for tile k (cur) to be ready
        __syncthreads();
        COMPUTE_TILE(cur);
        __syncthreads();            // protect cur: next ISSUE_TILE will overwrite it as nxt
    }

    // Last tile
    __pipeline_wait_prior(0);
    __syncthreads();
    COMPUTE_TILE((num_tiles - 1) & 1);

#undef ISSUE_TILE
#undef COMPUTE_TILE

    // Write back
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const int gr = row_start + i, gc = col_start + j;
            if (gr < M && gc < N) C[gr * N + gc] = (scalar_t)acc[i][j];
        }
}

// ---- Launch wrapper macro (no UNROLL — load loops are 1-4 iters, compiler handles) ----
#define MAKE_LAUNCHER_S4(NAME, BM, BN, BK, TM, TN)                                 \
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
        matmul_s4<scalar_t, BM, BN, BK, TM, TN><<<blocks, threads>>>(              \
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(),                         \
            C.data_ptr<scalar_t>(), M, K, N);                                       \
    });                                                                             \
    return C;                                                                       \
}

// ---- Stage 4 instantiations: BK=16, same 5 configs as Stage 3 ----
//                             NAME                              BM   BN  BK  TM  TN
MAKE_LAUNCHER_S4(matmul_cuda_s4_tm4_tn4_bm32_bn64_bk16,         32,  64, 16,  4,  4)
MAKE_LAUNCHER_S4(matmul_cuda_s4_tm4_tn4_bm64_bn64_bk16,         64,  64, 16,  4,  4)
MAKE_LAUNCHER_S4(matmul_cuda_s4_tm8_tn4_bm64_bn64_bk16,         64,  64, 16,  8,  4)
MAKE_LAUNCHER_S4(matmul_cuda_s4_tm8_tn8_bm128_bn64_bk16,       128,  64, 16,  8,  8)
MAKE_LAUNCHER_S4(matmul_cuda_s4_tm8_tn8_bm128_bn128_bk16,      128, 128, 16,  8,  8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda_s4_tm4_tn4_bm32_bn64_bk16",  &matmul_cuda_s4_tm4_tn4_bm32_bn64_bk16);
    m.def("matmul_cuda_s4_tm4_tn4_bm64_bn64_bk16",  &matmul_cuda_s4_tm4_tn4_bm64_bn64_bk16);
    m.def("matmul_cuda_s4_tm8_tn4_bm64_bn64_bk16",  &matmul_cuda_s4_tm8_tn4_bm64_bn64_bk16);
    m.def("matmul_cuda_s4_tm8_tn8_bm128_bn64_bk16", &matmul_cuda_s4_tm8_tn8_bm128_bn64_bk16);
    m.def("matmul_cuda_s4_tm8_tn8_bm128_bn128_bk16",&matmul_cuda_s4_tm8_tn8_bm128_bn128_bk16);
}
