#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

#define AT_DISPATCH_FLOAT_HALF_BF16(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME,                   \
        AT_DISPATCH_CASE(at::ScalarType::Float,      __VA_ARGS__) \
        AT_DISPATCH_CASE(at::ScalarType::Half,       __VA_ARGS__) \
        AT_DISPATCH_CASE(at::ScalarType::BFloat16,   __VA_ARGS__) \
    )

/*
 * Stage 4 + A-swizzle: eliminates shared-memory bank conflicts on the A tile.
 *
 * Key change vs Stage 4:
 *   A_LOAD_BYTES is fixed to 8 (4 elements for BF16/FP16, 2 for FP32).
 *   This gives A_ELEM = 4, meaning each thread's cp.async covers a 4-element group.
 *   BK=16 contains exactly 4 such groups per row (columns 0–3, 4–7, 8–11, 12–15).
 *
 *   XOR swizzle: element stored at shared-memory row r, logical column c is placed at
 *     column  c_sw = c ^ (((r >> 3) & 3) << 2)
 *   where c is always a multiple of 4 (group-aligned). The XOR only touches bits 2–3,
 *   so within-group alignment is preserved and the mapping is a bijection on each row.
 *
 *   On the read side (COMPUTE_TILE), thread with lty reads logical column _kk from
 *     column  _kk ^ ((lty & 3) << 2)
 *   Consistency: shared row = lty*TM+i, so (row>>3)&3 = (lty*TM+i)>>3 & 3 = lty & 3
 *   (valid for TM=8, i in 0..7).
 *
 *   For BM=128 (LCOLS=16): each warp spans lty=0..1 → 2-way conflict eliminated.
 *   For BM=64  (LCOLS=8) : each warp spans lty=0..3 → 4-way conflict eliminated.
 */
template <typename scalar_t, int BM, int BN, int BK, int TM, int TN, int UNROLL>
__global__ void matmul_s4sw(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int K, int N
) {
    constexpr int THREADS  = (BM / TM) * (BN / TN);
    constexpr int LCOLS    = BN / TN;

    // Force A_LOAD_BYTES=8 so A_ELEM=4 groups align with the XOR swizzle stride of 4.
    constexpr int A_LOAD_BYTES = 8;
    constexpr int A_ELEM       = A_LOAD_BYTES / (int)sizeof(scalar_t);
    constexpr int A_GROUPS     = BM * BK / A_ELEM / THREADS;

    constexpr int B_THREAD_BYTES = BK * BN * (int)sizeof(scalar_t) / THREADS;
    constexpr int B_LOAD_BYTES   = (B_THREAD_BYTES >= 16) ? 16 : (B_THREAD_BYTES >= 8) ? 8 : 4;
    constexpr int B_ELEM         = B_LOAD_BYTES / (int)sizeof(scalar_t);
    constexpr int B_GROUPS       = BK * BN / B_ELEM / THREADS;

    __shared__ scalar_t A_shared[2][BM][BK];
    __shared__ scalar_t B_shared[2][BK][BN];

    const int tx  = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int ltx = tid % LCOLS, lty = tid / LCOLS;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int row_start = block_row + lty * TM;
    const int col_start = block_col + ltx * TN;

    float acc[TM][TN] = {};

    // ISSUE_TILE: copy A with XOR-swizzle on destination column.
    //   _c is always a multiple of A_ELEM=4, so _c_sw = _c ^ (4-aligned) is also 4-aligned.
    //   B is copied without swizzle (no bank conflicts on B side for the configs used here).
#define ISSUE_TILE(k0_, buf_)                                                           \
    do {                                                                                \
        _Pragma("unroll")                                                               \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                        \
            const int _g = tid + _i * THREADS;                                         \
            const int _r = (_g * A_ELEM) / BK, _c = (_g * A_ELEM) % BK;              \
            const int _c_sw = _c ^ (((_r >> 3) & 3) << 2);                            \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_c_sw],                      \
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

    // COMPUTE_TILE: un-swizzle A reads with the same XOR formula.
    //   _swizzle = (lty & 3) << 2 is constant per thread; hoist outside the loops.
    //   XOR only affects bits 2-3, so within-group offset (_kk & 3) is preserved.
#define COMPUTE_TILE(buf_)                                                              \
    do {                                                                                \
        const int _swizzle = (lty & 3) << 2;                                           \
        _Pragma("unroll UNROLL")                                                        \
        for (int _kk = 0; _kk < BK; _kk++) {                                           \
            float _a[TM], _b[TN];                                                       \
            _Pragma("unroll")                                                           \
            for (int _i = 0; _i < TM; _i++)                                            \
                _a[_i] = (float)A_shared[(buf_)][lty * TM + _i][_kk ^ _swizzle];      \
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

#define MAKE_LAUNCHER_S4SW(NAME, BM, BN, BK, TM, TN, UNROLL)                        \
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
        matmul_s4sw<scalar_t, BM, BN, BK, TM, TN, UNROLL><<<blocks, threads>>>(    \
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(),                         \
            C.data_ptr<scalar_t>(), M, K, N);                                       \
    });                                                                             \
    return C;                                                                       \
}

//                               NAME                                  BM   BN  BK  TM  TN  UNROLL
MAKE_LAUNCHER_S4SW(matmul_cuda_s4sw_tm8_tn8_bm128_bn128_bk16_u1,    128, 128, 16,  8,  8,   1)
MAKE_LAUNCHER_S4SW(matmul_cuda_s4sw_tm8_tn8_bm128_bn128_bk16_u2,    128, 128, 16,  8,  8,   2)
MAKE_LAUNCHER_S4SW(matmul_cuda_s4sw_tm8_tn8_bm128_bn128_bk16_u4,    128, 128, 16,  8,  8,   4)
MAKE_LAUNCHER_S4SW(matmul_cuda_s4sw_tm8_tn8_bm128_bn128_bk16_u8,    128, 128, 16,  8,  8,   8)
MAKE_LAUNCHER_S4SW(matmul_cuda_s4sw_tm8_tn8_bm128_bn128_bk16_u16,   128, 128, 16,  8,  8,  16)

MAKE_LAUNCHER_S4SW(matmul_cuda_s4sw_tm8_tn8_bm128_bn64_bk16_u1,     128,  64, 16,  8,  8,   1)
MAKE_LAUNCHER_S4SW(matmul_cuda_s4sw_tm8_tn8_bm128_bn64_bk16_u2,     128,  64, 16,  8,  8,   2)
MAKE_LAUNCHER_S4SW(matmul_cuda_s4sw_tm8_tn8_bm128_bn64_bk16_u4,     128,  64, 16,  8,  8,   4)
MAKE_LAUNCHER_S4SW(matmul_cuda_s4sw_tm8_tn8_bm128_bn64_bk16_u8,     128,  64, 16,  8,  8,   8)
MAKE_LAUNCHER_S4SW(matmul_cuda_s4sw_tm8_tn8_bm128_bn64_bk16_u16,    128,  64, 16,  8,  8,  16)

MAKE_LAUNCHER_S4SW(matmul_cuda_s4sw_tm8_tn8_bm64_bn64_bk16_u1,       64,  64, 16,  8,  8,   1)
MAKE_LAUNCHER_S4SW(matmul_cuda_s4sw_tm8_tn8_bm64_bn64_bk16_u2,       64,  64, 16,  8,  8,   2)
MAKE_LAUNCHER_S4SW(matmul_cuda_s4sw_tm8_tn8_bm64_bn64_bk16_u4,       64,  64, 16,  8,  8,   4)
MAKE_LAUNCHER_S4SW(matmul_cuda_s4sw_tm8_tn8_bm64_bn64_bk16_u8,       64,  64, 16,  8,  8,   8)
MAKE_LAUNCHER_S4SW(matmul_cuda_s4sw_tm8_tn8_bm64_bn64_bk16_u16,      64,  64, 16,  8,  8,  16)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda_s4sw_tm8_tn8_bm128_bn128_bk16_u1",   &matmul_cuda_s4sw_tm8_tn8_bm128_bn128_bk16_u1);
    m.def("matmul_cuda_s4sw_tm8_tn8_bm128_bn128_bk16_u2",   &matmul_cuda_s4sw_tm8_tn8_bm128_bn128_bk16_u2);
    m.def("matmul_cuda_s4sw_tm8_tn8_bm128_bn128_bk16_u4",   &matmul_cuda_s4sw_tm8_tn8_bm128_bn128_bk16_u4);
    m.def("matmul_cuda_s4sw_tm8_tn8_bm128_bn128_bk16_u8",   &matmul_cuda_s4sw_tm8_tn8_bm128_bn128_bk16_u8);
    m.def("matmul_cuda_s4sw_tm8_tn8_bm128_bn128_bk16_u16",  &matmul_cuda_s4sw_tm8_tn8_bm128_bn128_bk16_u16);
    m.def("matmul_cuda_s4sw_tm8_tn8_bm128_bn64_bk16_u1",    &matmul_cuda_s4sw_tm8_tn8_bm128_bn64_bk16_u1);
    m.def("matmul_cuda_s4sw_tm8_tn8_bm128_bn64_bk16_u2",    &matmul_cuda_s4sw_tm8_tn8_bm128_bn64_bk16_u2);
    m.def("matmul_cuda_s4sw_tm8_tn8_bm128_bn64_bk16_u4",    &matmul_cuda_s4sw_tm8_tn8_bm128_bn64_bk16_u4);
    m.def("matmul_cuda_s4sw_tm8_tn8_bm128_bn64_bk16_u8",    &matmul_cuda_s4sw_tm8_tn8_bm128_bn64_bk16_u8);
    m.def("matmul_cuda_s4sw_tm8_tn8_bm128_bn64_bk16_u16",   &matmul_cuda_s4sw_tm8_tn8_bm128_bn64_bk16_u16);
    m.def("matmul_cuda_s4sw_tm8_tn8_bm64_bn64_bk16_u1",     &matmul_cuda_s4sw_tm8_tn8_bm64_bn64_bk16_u1);
    m.def("matmul_cuda_s4sw_tm8_tn8_bm64_bn64_bk16_u2",     &matmul_cuda_s4sw_tm8_tn8_bm64_bn64_bk16_u2);
    m.def("matmul_cuda_s4sw_tm8_tn8_bm64_bn64_bk16_u4",     &matmul_cuda_s4sw_tm8_tn8_bm64_bn64_bk16_u4);
    m.def("matmul_cuda_s4sw_tm8_tn8_bm64_bn64_bk16_u8",     &matmul_cuda_s4sw_tm8_tn8_bm64_bn64_bk16_u8);
    m.def("matmul_cuda_s4sw_tm8_tn8_bm64_bn64_bk16_u16",    &matmul_cuda_s4sw_tm8_tn8_bm64_bn64_bk16_u16);
}
