#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * s4st_tn16_p1: register-prefetch software pipeline, depth 1.
 *
 * Same layout as s4st_tn16 (TM=8, TN=16, BM=128, BN=256, BK=16, THREADS=256).
 *
 * COMPUTE_TILE inner loop structure (no unrolling of kk):
 *
 *   prefetch kk=0 → a[8], b[16]
 *   for kk = 1 .. BK-1:
 *       issue loads  for kk   → na[8], nb[16]   (next, in-flight)
 *       128 FMAs using a[8], b[16]               (previous kk, already in regs)
 *       a, b ← na, nb                            (zero-cost SSA rename)
 *   128 FMAs using last a[8], b[16]              (epilogue)
 *
 * The 24 smem loads for kk+1 are issued BEFORE the 128 FMAs for kk.
 * By the time the next iteration needs na/nb as current, 128 FMA-cycles
 * have elapsed — far beyond the ~20-cycle smem latency. Zero stall.
 *
 * Register budget (unroll 1, no compiler duplication):
 *   acc[8][16]        = 128  (always live)
 *   current  a[8]+b[16] =  24  (live during FMAs)
 *   next    na[8]+nb[16] =  24  (live during loads, renamed to current)
 *   A_addr + B_addr   =   2
 *   overhead          =  ~20
 *   Total             = ~198   (below 255 cliff, close to u1's 194)
 */

#define LD_S(reg, base, byte_off) \
    asm("ld.shared.f32 %0, [%1+" #byte_off "];" : "=f"(reg) : "r"(base))

template <int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void matmul_s4st_tn16_p1_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
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

#define COMPUTE_TILE(buf_)                                                                  \
    do {                                                                                    \
        uint32_t A_addr = __cvta_generic_to_shared(&A_shared[(buf_)][lty][0]);             \
        uint32_t B_addr = __cvta_generic_to_shared(&B_shared[(buf_)][0][ltx]);             \
        /* prefetch kk=0 into registers */                                                  \
        float a0, a1, a2, a3, a4, a5, a6, a7;                                              \
        LD_S(a0, A_addr,    0);  LD_S(a1, A_addr, 1024);                                   \
        LD_S(a2, A_addr, 2048);  LD_S(a3, A_addr, 3072);                                   \
        LD_S(a4, A_addr, 4096);  LD_S(a5, A_addr, 5120);                                   \
        LD_S(a6, A_addr, 6144);  LD_S(a7, A_addr, 7168);                                   \
        float b0,  b1,  b2,  b3,  b4,  b5,  b6,  b7;                                      \
        float b8,  b9,  b10, b11, b12, b13, b14, b15;                                      \
        LD_S(b0,  B_addr,   0);  LD_S(b1,  B_addr,   64);                                  \
        LD_S(b2,  B_addr, 128);  LD_S(b3,  B_addr,  192);                                  \
        LD_S(b4,  B_addr, 256);  LD_S(b5,  B_addr,  320);                                  \
        LD_S(b6,  B_addr, 384);  LD_S(b7,  B_addr,  448);                                  \
        LD_S(b8,  B_addr, 512);  LD_S(b9,  B_addr,  576);                                  \
        LD_S(b10, B_addr, 640);  LD_S(b11, B_addr,  704);                                  \
        LD_S(b12, B_addr, 768);  LD_S(b13, B_addr,  832);                                  \
        LD_S(b14, B_addr, 896);  LD_S(b15, B_addr,  960);                                  \
        A_addr += 4;                                                                        \
        B_addr += BN * 4;                                                                   \
        /* pipeline: issue next kk's loads, then FMA on current kk */                      \
        _Pragma("unroll 1")                                                                 \
        for (int _kk = 1; _kk < BK; _kk++) {                                               \
            float na0, na1, na2, na3, na4, na5, na6, na7;                                  \
            LD_S(na0, A_addr,    0);  LD_S(na1, A_addr, 1024);                             \
            LD_S(na2, A_addr, 2048);  LD_S(na3, A_addr, 3072);                             \
            LD_S(na4, A_addr, 4096);  LD_S(na5, A_addr, 5120);                             \
            LD_S(na6, A_addr, 6144);  LD_S(na7, A_addr, 7168);                             \
            float nb0,  nb1,  nb2,  nb3,  nb4,  nb5,  nb6,  nb7;                          \
            float nb8,  nb9,  nb10, nb11, nb12, nb13, nb14, nb15;                          \
            LD_S(nb0,  B_addr,   0);  LD_S(nb1,  B_addr,   64);                            \
            LD_S(nb2,  B_addr, 128);  LD_S(nb3,  B_addr,  192);                            \
            LD_S(nb4,  B_addr, 256);  LD_S(nb5,  B_addr,  320);                            \
            LD_S(nb6,  B_addr, 384);  LD_S(nb7,  B_addr,  448);                            \
            LD_S(nb8,  B_addr, 512);  LD_S(nb9,  B_addr,  576);                            \
            LD_S(nb10, B_addr, 640);  LD_S(nb11, B_addr,  704);                            \
            LD_S(nb12, B_addr, 768);  LD_S(nb13, B_addr,  832);                            \
            LD_S(nb14, B_addr, 896);  LD_S(nb15, B_addr,  960);                            \
            A_addr += 4;                                                                    \
            B_addr += BN * 4;                                                               \
            /* FMAs on previous kk (a,b ready; na,nb in-flight but unused here) */         \
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
            /* rename: next → current (compiler SSA, zero-cost) */                         \
            a0=na0; a1=na1; a2=na2; a3=na3; a4=na4; a5=na5; a6=na6; a7=na7;              \
            b0=nb0;  b1=nb1;  b2=nb2;  b3=nb3;  b4=nb4;  b5=nb5;  b6=nb6;  b7=nb7;      \
            b8=nb8;  b9=nb9;  b10=nb10; b11=nb11; b12=nb12; b13=nb13; b14=nb14; b15=nb15; \
        }                                                                                   \
        /* epilogue: FMAs for last kk */                                                    \
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
    matmul_s4st_tn16_p1_impl<BM, BN, BK, TM, TN>(A, B, C, M, K, N);     \
}

MAKE_LAUNCHER(matmul_cuda_s4st_tn16_p1_tm8_tn16_bm128_bn256_bk16, 128, 256, 16, 8, 16)

#undef MAKE_LAUNCHER
