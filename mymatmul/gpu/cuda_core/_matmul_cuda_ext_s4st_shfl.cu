#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

// s4st with intra-warp shuffle to reduce smem reads.
//
// Warp layout (bm128_bn128, 256 threads = 8 warps):
//   Each warp spans 2 lty-rows × 16 ltx-cols.
//   lane 0..15:  lty = warp_id*2,     ltx = 0..15
//   lane 16..31: lty = warp_id*2 + 1, ltx = 0..15
//
// Per-kk smem reads vs s4st:
//   A: 256 → 16  (only lanes 0 and 16 load; broadcast to half-warp)
//   B: 256 → 128 (only lanes 0..15 load; lanes 16..31 shfl from partner)
//   Total: 512 → 144 per kk per warp

extern "C" __global__ void matmul_cuda_s4st_shfl_tm8_tn8_bm128_bn128_bk16(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int TM = 8,  TN = 8;
    constexpr int LROWS = BM / TM;   // 16 — stride between rows a thread owns
    constexpr int LCOLS = BN / TN;   // 16 — stride between cols a thread owns
    constexpr int THREADS = 256;

    constexpr int A_ELEM   = 4;                          // 16-byte cp.async
    constexpr int B_ELEM   = 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS; // 2
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS; // 2

    __shared__ float A_shared[2][BM][BK];
    __shared__ float B_shared[2][BK][BN];

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;

    // Logical thread position in the 16×16 grid (s4st strided layout)
    const int lty_in_warp = lane >> 4;         // 0 or 1
    const int ltx         = lane & 15;          // 0..15
    const int lty         = warp_id * 2 + lty_in_warp;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // Shuffle sources (compile-time constants per lane)
    // A: broadcast from lane 0 (lty_in_warp=0) or lane 16 (lty_in_warp=1)
    const int a_src = lane & 16;   // 0 for lanes 0..15, 16 for lanes 16..31
    // B: lanes 16..31 read from their ltx-matched partner in lanes 0..15
    const int b_src = lane & 15;   // lane % 16

    float acc[TM][TN] = {};

#define ISSUE_TILE(k0_, buf_)                                                       \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                    \
            const int _g = tid + _i * THREADS;                                     \
            const int _r = (_g * A_ELEM) / BK, _c = (_g * A_ELEM) % BK;           \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_c],                     \
                                    &A[(block_row + _r) * K + (k0_) + _c], 16);   \
        }                                                                           \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                    \
            const int _g = tid + _i * THREADS;                                     \
            const int _r = (_g * B_ELEM) / BN, _c = (_g * B_ELEM) % BN;           \
            __pipeline_memcpy_async(&B_shared[(buf_)][_r][_c],                     \
                                    &B[((k0_) + _r) * N + block_col + _c], 16);   \
        }                                                                           \
        __pipeline_commit();                                                        \
    } while (0)

#define COMPUTE_TILE(buf_)                                                          \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _kk = 0; _kk < BK; _kk++) {                                       \
            float a_frag[TM] = {};                                                  \
            float b_frag[TN] = {};                                                  \
                                                                                    \
            /* A: only lanes 0 and 16 load (one per lty-row in the warp) */        \
            if (lane == 0 || lane == 16) {                                          \
                _Pragma("unroll")                                                   \
                for (int _i = 0; _i < TM; _i++)                                    \
                    a_frag[_i] = A_shared[(buf_)][lty + _i * LROWS][_kk];          \
            }                                                                       \
            /* B: only lanes 0..15 load (one per ltx value) */                     \
            if (lane < 16) {                                                        \
                _Pragma("unroll")                                                   \
                for (int _j = 0; _j < TN; _j++)                                    \
                    b_frag[_j] = B_shared[(buf_)][_kk][ltx + _j * LCOLS];          \
            }                                                                       \
                                                                                    \
            /* Broadcast A within each half-warp */                                 \
            _Pragma("unroll")                                                       \
            for (int _i = 0; _i < TM; _i++)                                        \
                a_frag[_i] = __shfl_sync(0xffffffff, a_frag[_i], a_src);           \
                                                                                    \
            /* Lanes 16..31 pick up B from their lane%16 partner */                 \
            _Pragma("unroll")                                                       \
            for (int _j = 0; _j < TN; _j++)                                        \
                b_frag[_j] = __shfl_sync(0xffffffff, b_frag[_j], b_src);           \
                                                                                    \
            _Pragma("unroll")                                                       \
            for (int _i = 0; _i < TM; _i++)                                        \
                _Pragma("unroll")                                                   \
                for (int _j = 0; _j < TN; _j++)                                    \
                    acc[_i][_j] += a_frag[_i] * b_frag[_j];                        \
        }                                                                           \
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

    // Writeback — same strided pattern as s4st
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const int gr = block_row + lty + i * LROWS;
            const int gc = block_col + ltx + j * LCOLS;
            if (gr < M && gc < N)
                C[gr * N + gc] = acc[i][j];
        }
    }
}
