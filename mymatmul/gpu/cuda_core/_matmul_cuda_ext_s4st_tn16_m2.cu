#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * s4st_tn16_m2: hand-crafted 2-way smem software pipelining.
 *
 * Same layout as s4st_tn16 (TM=8, TN=16, BM=128, BN=256, BK=16, THREADS=256)
 * but COMPUTE_TILE is manually interleaved to hide shared-memory latency:
 *
 *   per outer iteration (handles kk=2i and kk=2i+1):
 *     issue 8A + 16B loads  for kk=2i    (24 ld.shared)
 *     issue 8A + 16B loads  for kk=2i+1  (24 ld.shared)  ← covers ~20-cycle smem latency
 *     128 FMAs for kk=2i    (values ready by now)
 *     128 FMAs for kk=2i+1
 *
 * Efficiency target: 128 FMAs / (128 FMAs + 24 loads) ≈ 84% FMA throughput.
 *
 * Register budget:
 *   acc[8][16] = 128  (always live)
 *   a0..a7  + b0..b15  = 24  (iter kk+0)
 *   c0..c7  + d0..d15  = 24  (iter kk+1)
 *   A_addr + B_addr    =  2
 *   overhead           = ~42
 *   Total ≈ 220  (< 255, no spill expected)
 *
 * A_shared byte offsets from A_addr = &A_shared[buf][lty][0]:
 *   kk+0, row i:  i*LROWS*BK*4 + 0 = i*1024
 *   kk+1, row i:  i*LROWS*BK*4 + 4 = i*1024 + 4
 *
 * B_shared byte offsets from B_addr = &B_shared[buf][0][ltx]:
 *   kk+0, col j:  j*LCOLS*4        = j*64
 *   kk+1, col j:  BN*4 + j*LCOLS*4 = 1024 + j*64
 */

#define LD_S(reg, base, byte_off) \
    asm("ld.shared.f32 %0, [%1+" #byte_off "];" : "=f"(reg) : "r"(base))

template <int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void matmul_s4st_tn16_m2_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    static_assert(BK % 2 == 0, "BK must be even for 2-way manual unroll");

    constexpr int THREADS = (BM / TM) * (BN / TN);   // 256
    constexpr int LCOLS   = BN / TN;                  // 16
    constexpr int LROWS   = BM / TM;                  // 16

    constexpr int A_ELEM   = 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;   // 2
    constexpr int B_ELEM   = 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;   // 4

    __shared__ float A_shared[2][BM][BK];
    __shared__ float B_shared[2][BK][BN];

    const int tx  = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int ltx = tid % LCOLS;
    const int lty = tid / LCOLS;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};

#define ISSUE_TILE(k0_, buf_)                                                               \
    do {                                                                                    \
        _Pragma("unroll")                                                                   \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                            \
            const int _g = tid + _i * THREADS;                                             \
            const int _r = (_g * A_ELEM) / BK, _c = (_g * A_ELEM) % BK;                  \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_c],                             \
                                    &A[(block_row + _r) * K + (k0_) + _c],                 \
                                    A_ELEM * (int)sizeof(float));                           \
        }                                                                                   \
        _Pragma("unroll")                                                                   \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                            \
            const int _g = tid + _i * THREADS;                                             \
            const int _r = (_g * B_ELEM) / BN, _c = (_g * B_ELEM) % BN;                  \
            __pipeline_memcpy_async(&B_shared[(buf_)][_r][_c],                             \
                                    &B[((k0_) + _r) * N + block_col + _c],                 \
                                    B_ELEM * (int)sizeof(float));                           \
        }                                                                                   \
        __pipeline_commit();                                                                \
    } while (0)

    // COMPUTE_TILE: manually interleaved 2-way software pipeline.
    // Outer loop: BK/2 iterations, each handling kk and kk+1.
    // Interleaving:
    //   loads(kk+0) → loads(kk+1) → FMAs(kk+0) → FMAs(kk+1)
    // The 24 loads for kk+1 act as a latency cover for kk+0's smem reads.
#define COMPUTE_TILE(buf_)                                                                  \
    do {                                                                                    \
        uint32_t A_addr = __cvta_generic_to_shared(&A_shared[(buf_)][lty][0]);             \
        uint32_t B_addr = __cvta_generic_to_shared(&B_shared[(buf_)][0][ltx]);             \
        _Pragma("unroll 1")                                                                 \
        for (int _kk = 0; _kk < BK; _kk += 2) {                                           \
            /* --- load kk+0: 8A + 16B --- */                                              \
            float a0, a1, a2, a3, a4, a5, a6, a7;                                          \
            LD_S(a0, A_addr,    0);  LD_S(a1, A_addr, 1024);                               \
            LD_S(a2, A_addr, 2048);  LD_S(a3, A_addr, 3072);                               \
            LD_S(a4, A_addr, 4096);  LD_S(a5, A_addr, 5120);                               \
            LD_S(a6, A_addr, 6144);  LD_S(a7, A_addr, 7168);                               \
            float b0,  b1,  b2,  b3,  b4,  b5,  b6,  b7;                                  \
            float b8,  b9,  b10, b11, b12, b13, b14, b15;                                  \
            LD_S(b0,  B_addr,   0);  LD_S(b1,  B_addr,   64);                              \
            LD_S(b2,  B_addr, 128);  LD_S(b3,  B_addr,  192);                              \
            LD_S(b4,  B_addr, 256);  LD_S(b5,  B_addr,  320);                              \
            LD_S(b6,  B_addr, 384);  LD_S(b7,  B_addr,  448);                              \
            LD_S(b8,  B_addr, 512);  LD_S(b9,  B_addr,  576);                              \
            LD_S(b10, B_addr, 640);  LD_S(b11, B_addr,  704);                              \
            LD_S(b12, B_addr, 768);  LD_S(b13, B_addr,  832);                              \
            LD_S(b14, B_addr, 896);  LD_S(b15, B_addr,  960);                              \
            /* --- load kk+1: 8A + 16B (latency cover for kk+0) --- */                    \
            float c0, c1, c2, c3, c4, c5, c6, c7;                                          \
            LD_S(c0, A_addr,    4);  LD_S(c1, A_addr, 1028);                               \
            LD_S(c2, A_addr, 2052);  LD_S(c3, A_addr, 3076);                               \
            LD_S(c4, A_addr, 4100);  LD_S(c5, A_addr, 5124);                               \
            LD_S(c6, A_addr, 6148);  LD_S(c7, A_addr, 7172);                               \
            float d0,  d1,  d2,  d3,  d4,  d5,  d6,  d7;                                  \
            float d8,  d9,  d10, d11, d12, d13, d14, d15;                                  \
            LD_S(d0,  B_addr, 1024);  LD_S(d1,  B_addr, 1088);                             \
            LD_S(d2,  B_addr, 1152);  LD_S(d3,  B_addr, 1216);                             \
            LD_S(d4,  B_addr, 1280);  LD_S(d5,  B_addr, 1344);                             \
            LD_S(d6,  B_addr, 1408);  LD_S(d7,  B_addr, 1472);                             \
            LD_S(d8,  B_addr, 1536);  LD_S(d9,  B_addr, 1600);                             \
            LD_S(d10, B_addr, 1664);  LD_S(d11, B_addr, 1728);                             \
            LD_S(d12, B_addr, 1792);  LD_S(d13, B_addr, 1856);                             \
            LD_S(d14, B_addr, 1920);  LD_S(d15, B_addr, 1984);                             \
            A_addr += 8;     /* advance 2 columns (4 bytes × 2) */                         \
            B_addr += 2048;  /* advance 2 rows   (1024 bytes × 2) */                       \
            /* --- FMAs for kk+0 (128 ops) --- */                                          \
            acc[0][ 0]+=a0*b0;  acc[0][ 1]+=a0*b1;  acc[0][ 2]+=a0*b2;  acc[0][ 3]+=a0*b3;   \
            acc[0][ 4]+=a0*b4;  acc[0][ 5]+=a0*b5;  acc[0][ 6]+=a0*b6;  acc[0][ 7]+=a0*b7;   \
            acc[0][ 8]+=a0*b8;  acc[0][ 9]+=a0*b9;  acc[0][10]+=a0*b10; acc[0][11]+=a0*b11;   \
            acc[0][12]+=a0*b12; acc[0][13]+=a0*b13; acc[0][14]+=a0*b14; acc[0][15]+=a0*b15;   \
            acc[1][ 0]+=a1*b0;  acc[1][ 1]+=a1*b1;  acc[1][ 2]+=a1*b2;  acc[1][ 3]+=a1*b3;   \
            acc[1][ 4]+=a1*b4;  acc[1][ 5]+=a1*b5;  acc[1][ 6]+=a1*b6;  acc[1][ 7]+=a1*b7;   \
            acc[1][ 8]+=a1*b8;  acc[1][ 9]+=a1*b9;  acc[1][10]+=a1*b10; acc[1][11]+=a1*b11;   \
            acc[1][12]+=a1*b12; acc[1][13]+=a1*b13; acc[1][14]+=a1*b14; acc[1][15]+=a1*b15;   \
            acc[2][ 0]+=a2*b0;  acc[2][ 1]+=a2*b1;  acc[2][ 2]+=a2*b2;  acc[2][ 3]+=a2*b3;   \
            acc[2][ 4]+=a2*b4;  acc[2][ 5]+=a2*b5;  acc[2][ 6]+=a2*b6;  acc[2][ 7]+=a2*b7;   \
            acc[2][ 8]+=a2*b8;  acc[2][ 9]+=a2*b9;  acc[2][10]+=a2*b10; acc[2][11]+=a2*b11;   \
            acc[2][12]+=a2*b12; acc[2][13]+=a2*b13; acc[2][14]+=a2*b14; acc[2][15]+=a2*b15;   \
            acc[3][ 0]+=a3*b0;  acc[3][ 1]+=a3*b1;  acc[3][ 2]+=a3*b2;  acc[3][ 3]+=a3*b3;   \
            acc[3][ 4]+=a3*b4;  acc[3][ 5]+=a3*b5;  acc[3][ 6]+=a3*b6;  acc[3][ 7]+=a3*b7;   \
            acc[3][ 8]+=a3*b8;  acc[3][ 9]+=a3*b9;  acc[3][10]+=a3*b10; acc[3][11]+=a3*b11;   \
            acc[3][12]+=a3*b12; acc[3][13]+=a3*b13; acc[3][14]+=a3*b14; acc[3][15]+=a3*b15;   \
            acc[4][ 0]+=a4*b0;  acc[4][ 1]+=a4*b1;  acc[4][ 2]+=a4*b2;  acc[4][ 3]+=a4*b3;   \
            acc[4][ 4]+=a4*b4;  acc[4][ 5]+=a4*b5;  acc[4][ 6]+=a4*b6;  acc[4][ 7]+=a4*b7;   \
            acc[4][ 8]+=a4*b8;  acc[4][ 9]+=a4*b9;  acc[4][10]+=a4*b10; acc[4][11]+=a4*b11;   \
            acc[4][12]+=a4*b12; acc[4][13]+=a4*b13; acc[4][14]+=a4*b14; acc[4][15]+=a4*b15;   \
            acc[5][ 0]+=a5*b0;  acc[5][ 1]+=a5*b1;  acc[5][ 2]+=a5*b2;  acc[5][ 3]+=a5*b3;   \
            acc[5][ 4]+=a5*b4;  acc[5][ 5]+=a5*b5;  acc[5][ 6]+=a5*b6;  acc[5][ 7]+=a5*b7;   \
            acc[5][ 8]+=a5*b8;  acc[5][ 9]+=a5*b9;  acc[5][10]+=a5*b10; acc[5][11]+=a5*b11;   \
            acc[5][12]+=a5*b12; acc[5][13]+=a5*b13; acc[5][14]+=a5*b14; acc[5][15]+=a5*b15;   \
            acc[6][ 0]+=a6*b0;  acc[6][ 1]+=a6*b1;  acc[6][ 2]+=a6*b2;  acc[6][ 3]+=a6*b3;   \
            acc[6][ 4]+=a6*b4;  acc[6][ 5]+=a6*b5;  acc[6][ 6]+=a6*b6;  acc[6][ 7]+=a6*b7;   \
            acc[6][ 8]+=a6*b8;  acc[6][ 9]+=a6*b9;  acc[6][10]+=a6*b10; acc[6][11]+=a6*b11;   \
            acc[6][12]+=a6*b12; acc[6][13]+=a6*b13; acc[6][14]+=a6*b14; acc[6][15]+=a6*b15;   \
            acc[7][ 0]+=a7*b0;  acc[7][ 1]+=a7*b1;  acc[7][ 2]+=a7*b2;  acc[7][ 3]+=a7*b3;   \
            acc[7][ 4]+=a7*b4;  acc[7][ 5]+=a7*b5;  acc[7][ 6]+=a7*b6;  acc[7][ 7]+=a7*b7;   \
            acc[7][ 8]+=a7*b8;  acc[7][ 9]+=a7*b9;  acc[7][10]+=a7*b10; acc[7][11]+=a7*b11;   \
            acc[7][12]+=a7*b12; acc[7][13]+=a7*b13; acc[7][14]+=a7*b14; acc[7][15]+=a7*b15;   \
            /* --- FMAs for kk+1 (128 ops) --- */                                          \
            acc[0][ 0]+=c0*d0;  acc[0][ 1]+=c0*d1;  acc[0][ 2]+=c0*d2;  acc[0][ 3]+=c0*d3;   \
            acc[0][ 4]+=c0*d4;  acc[0][ 5]+=c0*d5;  acc[0][ 6]+=c0*d6;  acc[0][ 7]+=c0*d7;   \
            acc[0][ 8]+=c0*d8;  acc[0][ 9]+=c0*d9;  acc[0][10]+=c0*d10; acc[0][11]+=c0*d11;   \
            acc[0][12]+=c0*d12; acc[0][13]+=c0*d13; acc[0][14]+=c0*d14; acc[0][15]+=c0*d15;   \
            acc[1][ 0]+=c1*d0;  acc[1][ 1]+=c1*d1;  acc[1][ 2]+=c1*d2;  acc[1][ 3]+=c1*d3;   \
            acc[1][ 4]+=c1*d4;  acc[1][ 5]+=c1*d5;  acc[1][ 6]+=c1*d6;  acc[1][ 7]+=c1*d7;   \
            acc[1][ 8]+=c1*d8;  acc[1][ 9]+=c1*d9;  acc[1][10]+=c1*d10; acc[1][11]+=c1*d11;   \
            acc[1][12]+=c1*d12; acc[1][13]+=c1*d13; acc[1][14]+=c1*d14; acc[1][15]+=c1*d15;   \
            acc[2][ 0]+=c2*d0;  acc[2][ 1]+=c2*d1;  acc[2][ 2]+=c2*d2;  acc[2][ 3]+=c2*d3;   \
            acc[2][ 4]+=c2*d4;  acc[2][ 5]+=c2*d5;  acc[2][ 6]+=c2*d6;  acc[2][ 7]+=c2*d7;   \
            acc[2][ 8]+=c2*d8;  acc[2][ 9]+=c2*d9;  acc[2][10]+=c2*d10; acc[2][11]+=c2*d11;   \
            acc[2][12]+=c2*d12; acc[2][13]+=c2*d13; acc[2][14]+=c2*d14; acc[2][15]+=c2*d15;   \
            acc[3][ 0]+=c3*d0;  acc[3][ 1]+=c3*d1;  acc[3][ 2]+=c3*d2;  acc[3][ 3]+=c3*d3;   \
            acc[3][ 4]+=c3*d4;  acc[3][ 5]+=c3*d5;  acc[3][ 6]+=c3*d6;  acc[3][ 7]+=c3*d7;   \
            acc[3][ 8]+=c3*d8;  acc[3][ 9]+=c3*d9;  acc[3][10]+=c3*d10; acc[3][11]+=c3*d11;   \
            acc[3][12]+=c3*d12; acc[3][13]+=c3*d13; acc[3][14]+=c3*d14; acc[3][15]+=c3*d15;   \
            acc[4][ 0]+=c4*d0;  acc[4][ 1]+=c4*d1;  acc[4][ 2]+=c4*d2;  acc[4][ 3]+=c4*d3;   \
            acc[4][ 4]+=c4*d4;  acc[4][ 5]+=c4*d5;  acc[4][ 6]+=c4*d6;  acc[4][ 7]+=c4*d7;   \
            acc[4][ 8]+=c4*d8;  acc[4][ 9]+=c4*d9;  acc[4][10]+=c4*d10; acc[4][11]+=c4*d11;   \
            acc[4][12]+=c4*d12; acc[4][13]+=c4*d13; acc[4][14]+=c4*d14; acc[4][15]+=c4*d15;   \
            acc[5][ 0]+=c5*d0;  acc[5][ 1]+=c5*d1;  acc[5][ 2]+=c5*d2;  acc[5][ 3]+=c5*d3;   \
            acc[5][ 4]+=c5*d4;  acc[5][ 5]+=c5*d5;  acc[5][ 6]+=c5*d6;  acc[5][ 7]+=c5*d7;   \
            acc[5][ 8]+=c5*d8;  acc[5][ 9]+=c5*d9;  acc[5][10]+=c5*d10; acc[5][11]+=c5*d11;   \
            acc[5][12]+=c5*d12; acc[5][13]+=c5*d13; acc[5][14]+=c5*d14; acc[5][15]+=c5*d15;   \
            acc[6][ 0]+=c6*d0;  acc[6][ 1]+=c6*d1;  acc[6][ 2]+=c6*d2;  acc[6][ 3]+=c6*d3;   \
            acc[6][ 4]+=c6*d4;  acc[6][ 5]+=c6*d5;  acc[6][ 6]+=c6*d6;  acc[6][ 7]+=c6*d7;   \
            acc[6][ 8]+=c6*d8;  acc[6][ 9]+=c6*d9;  acc[6][10]+=c6*d10; acc[6][11]+=c6*d11;   \
            acc[6][12]+=c6*d12; acc[6][13]+=c6*d13; acc[6][14]+=c6*d14; acc[6][15]+=c6*d15;   \
            acc[7][ 0]+=c7*d0;  acc[7][ 1]+=c7*d1;  acc[7][ 2]+=c7*d2;  acc[7][ 3]+=c7*d3;   \
            acc[7][ 4]+=c7*d4;  acc[7][ 5]+=c7*d5;  acc[7][ 6]+=c7*d6;  acc[7][ 7]+=c7*d7;   \
            acc[7][ 8]+=c7*d8;  acc[7][ 9]+=c7*d9;  acc[7][10]+=c7*d10; acc[7][11]+=c7*d11;   \
            acc[7][12]+=c7*d12; acc[7][13]+=c7*d13; acc[7][14]+=c7*d14; acc[7][15]+=c7*d15;   \
        }                                                                                   \
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
            const int gr = block_row + lty + i * LROWS;
            const int gc = block_col + ltx + j * LCOLS;
            if (gr < M && gc < N)
                C[gr * N + gc] = acc[i][j];
        }
}

#undef LD_S

#define MAKE_LAUNCHER(NAME, BM, BN, BK, TM, TN)                           \
extern "C" __global__ void NAME(                                           \
    const float* __restrict__ A, const float* __restrict__ B,             \
    float* __restrict__ C, int M, int K, int N) {                          \
    matmul_s4st_tn16_m2_impl<BM, BN, BK, TM, TN>(A, B, C, M, K, N);     \
}

//         NAME                                            BM   BN  BK TM  TN
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_m2_tm8_tn16_bm128_bn256_bk16, 128,256,16, 8,16)

#undef MAKE_LAUNCHER
