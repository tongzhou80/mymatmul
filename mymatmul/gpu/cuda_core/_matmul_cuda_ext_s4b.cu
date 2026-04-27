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
 * Stage 4b: Stage 4 + A_shared bank-conflict fix.
 *
 * Change vs Stage 4:
 *   A_shared[2][BM][BK + 4]   (was [BK])
 *
 * Why BK+4 (not BK+1):
 *   cp.async requires the destination address to be aligned to the copy size.
 *   With BF16 (2 bytes/elem), row stride = (BK+P)*2 bytes.
 *   For 8-byte cp.async: need (BK+P)*2 % 8 = 0 → (BK+P) % 4 = 0.
 *   For bank-conflict fix: need 4*(BK+P) % 32 ≠ 0 → (BK+P) % 8 ≠ 0.
 *   These two together require (BK+P) ≡ 4 (mod 8) → minimum P=4.
 *   With P=4: row stride = 40 bytes (8-byte aligned ✓), bank_diff = 16 (≠ 0 ✓).
 *
 * Consequence: A_LOAD_BYTES is capped at 8 (was 16). A_GROUPS doubles (2 instead of 1).
 *   The padding column is never read or written; it only widens the physical row stride.
 *
 * Why this fixes the conflict:
 *   Without padding, row stride 32 bytes = 8 banks. Rows TM=8 apart shift by
 *   8×8=64 banks ≡ 0 (mod 32): they land on the same bank.
 *   With P=4, row stride 40 bytes = 10 banks. Rows 8 apart shift by 8×10=80 ≡ 16 (mod 32):
 *   always a different bank.
 */
template <typename scalar_t, int BM, int BN, int BK, int TM, int TN, int UNROLL>
__global__ void matmul_s4b(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int K, int N
) {
    constexpr int THREADS  = (BM / TM) * (BN / TN);
    constexpr int LCOLS    = BN / TN;

    // A: cap at 8 bytes to maintain 8-byte alignment with BK+4 row stride.
    constexpr int A_THREAD_BYTES = BM * BK * (int)sizeof(scalar_t) / THREADS;
    constexpr int A_LOAD_BYTES   = (A_THREAD_BYTES >= 8) ? 8 : 4;
    constexpr int A_ELEM         = A_LOAD_BYTES / (int)sizeof(scalar_t);
    constexpr int A_GROUPS       = BM * BK / A_ELEM / THREADS;

    // B: unchanged adaptive load size.
    constexpr int B_THREAD_BYTES = BK * BN * (int)sizeof(scalar_t) / THREADS;
    constexpr int B_LOAD_BYTES   = (B_THREAD_BYTES >= 16) ? 16 : (B_THREAD_BYTES >= 8) ? 8 : 4;
    constexpr int B_ELEM         = B_LOAD_BYTES / (int)sizeof(scalar_t);
    constexpr int B_GROUPS       = BK * BN / B_ELEM / THREADS;

    // BK+4 padding on A eliminates bank conflicts (see file header).
    __shared__ scalar_t A_shared[2][BM][BK + 4];
    __shared__ scalar_t B_shared[2][BK][BN];

    const int tx  = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int ltx = tid % LCOLS, lty = tid / LCOLS;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int row_start = block_row + lty * TM;
    const int col_start = block_col + ltx * TN;

    float acc[TM][TN] = {};

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
        _Pragma("unroll UNROLL")                                                        \
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

    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const int gr = row_start + i, gc = col_start + j;
            if (gr < M && gc < N) C[gr * N + gc] = (scalar_t)acc[i][j];
        }
}

#define MAKE_LAUNCHER_S4B(NAME, BM, BN, BK, TM, TN, UNROLL)                        \
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
        matmul_s4b<scalar_t, BM, BN, BK, TM, TN, UNROLL><<<blocks, threads>>>(     \
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(),                         \
            C.data_ptr<scalar_t>(), M, K, N);                                       \
    });                                                                             \
    return C;                                                                       \
}

// BN=128 only: P=4 fully eliminates the 2-way A conflict (lty=0 vs lty=1 per warp).
// BN=64 is omitted: it has 4-way A conflicts requiring P=2 + 4-byte cp.async (more overhead).
//                             NAME                                  BM   BN  BK  TM  TN  UNROLL
MAKE_LAUNCHER_S4B(matmul_cuda_s4b_tm8_tn8_bm128_bn128_bk16_u8,     128, 128, 16,  8,  8,   8)
MAKE_LAUNCHER_S4B(matmul_cuda_s4b_tm8_tn8_bm128_bn128_bk16_u16,    128, 128, 16,  8,  8,  16)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda_s4b_tm8_tn8_bm128_bn128_bk16_u8",   &matmul_cuda_s4b_tm8_tn8_bm128_bn128_bk16_u8);
    m.def("matmul_cuda_s4b_tm8_tn8_bm128_bn128_bk16_u16",  &matmul_cuda_s4b_tm8_tn8_bm128_bn128_bk16_u16);
}
