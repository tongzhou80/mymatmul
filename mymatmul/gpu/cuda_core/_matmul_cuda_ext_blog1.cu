#include <cuda_runtime.h>

/*
 * Blog series article 1 kernel (lpha-z.hatenablog.com/entry/2024/08/18/231500).
 *
 * Adapted from the original fixed-N=4096 C++ to take runtime M, K, N.
 * Parameters kept identical to the blog's best config:
 *
 *   ThreadsPerBlockI (TI) = 4         threadIdx.y range
 *   ThreadsPerBlockJ (TJ) = 32        threadIdx.x range
 *   SharedMemBlockK  (BK) = 16
 *   RegisterBlockI   (RI) = 16        per-thread M output elements
 *   RegisterBlockJ   (RJ) = 8         per-thread N output elements
 *   Block tile: BM = RI*TI = 64,  BN = RJ*TJ = 256
 *   128 threads per block.
 *
 * Key design choices from the blog:
 *   - Accumulator sum[RJ][RI] = sum[8][16] (J-major) matches register
 *     .reuse annotation pattern for better register bank utilization.
 *   - float4 loads for both A and B tiles.
 *   - Over-allocated shared memory (2× each dim) to constrain occupancy to
 *     2 blocks/SM, letting the compiler allocate registers more aggressively.
 *   - Single-buffer (no cp.async); standard load -> syncthreads -> compute.
 *   - Writeback at end: "accumulate-last" pattern (C initialized to 0 by caller).
 *   - No #pragma unroll: compiler unrolls automatically from constexpr bounds.
 *   - Compiled for sm_86 (author's target arch).
 */

extern "C" __global__
void matmul_cuda_blog1(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    constexpr int TI      = 4;
    constexpr int TJ      = 32;
    constexpr int BK      = 16;
    constexpr int RI      = 16;
    constexpr int RJ      = 8;
    constexpr int BM      = RI * TI;             // 64
    constexpr int BN      = RJ * TJ;             // 256
    constexpr int THREADS = TI * TJ;             // 128
    constexpr int LOOP_A  = BM * BK / THREADS;   // 8   (in floats, /4 for float4 iters = 2)
    constexpr int LOOP_BY = BK / TI;             // 4
    constexpr int LOOP_BZ = BN / TJ;             // 8   (/4 for float4 iters = 2)

    const int i0  = blockIdx.y * BM;
    const int j0  = blockIdx.x * BN;
    const int i2  = threadIdx.y;
    const int j2  = threadIdx.x;
    const int tid = i2 * TJ + j2;

    float sum[RJ][RI] = {};

    // Over-allocated: [BM*2][BK] and [BK*2][BN] to limit blocks/SM.
    __align__(16) __shared__ float local_a[BM * 2][BK];
    __align__(16) __shared__ float local_b[BK * 2][BN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        __syncthreads();

        // Load A tile: BM rows x BK cols via float4 (2 float4s per thread).
        for (int x = 0; x < LOOP_A / 4; ++x) {
            int il = (x * THREADS + tid) / (BK / 4);
            int kl = (x * THREADS + tid) % (BK / 4) * 4;
            *reinterpret_cast<float4*>(&local_a[il][kl]) =
                *reinterpret_cast<const float4*>(&A[(i0 + il) * K + (k0 + kl)]);
        }

        // Load B tile: BK rows x BN cols via float4 (8 float4s per thread).
        for (int z = 0; z < LOOP_BZ / 4; ++z)
        for (int y = 0; y < LOOP_BY; ++y) {
            int kl = y * TI + i2;
            int jl = (z * TJ + j2) * 4;
            *reinterpret_cast<float4*>(&local_b[kl][jl]) =
                *reinterpret_cast<const float4*>(&B[(k0 + kl) * N + (j0 + jl)]);
        }

        __syncthreads();

        // Compute: loop order k1, i1, j1, j3 — matches blog's register reuse pattern.
        // k1 kept as a real loop (not unrolled) to keep icache footprint under ~2 KB.
        // Fully unrolling all 4 loops into 2048 flat FMAs overflows the 32 KB icache.
        #pragma unroll 1
        for (int k1 = 0; k1 < BK; ++k1)
        for (int i1 = 0; i1 < RI; ++i1)
        for (int j1 = 0; j1 < RJ / 4; ++j1)
        for (int j3 = 0; j3 < 4; ++j3) {
            int il = i1 * TI + i2;
            int jl = (j1 * TJ + j2) * 4 + j3;
            sum[j1 * 4 + j3][i1] = fma(local_a[il][k1], local_b[k1][jl],
                                        sum[j1 * 4 + j3][i1]);
        }
    }

    // Writeback (C must be zeroed by caller).
    for (int i1 = 0; i1 < RI; ++i1)
    for (int j1 = 0; j1 < RJ / 4; ++j1)
    for (int j3 = 0; j3 < 4; ++j3) {
        int i = i0 + i1 * TI + i2;
        int j = j0 + (j1 * TJ + j2) * 4 + j3;
        if (i < M && j < N)
            C[i * N + j] += sum[j1 * 4 + j3][i1];
    }
}
