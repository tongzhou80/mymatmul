#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * s4st_tn16: TM=8, TN=16, BM=128, BN=256, BK=16.
 *
 * Same s4st strided output layout as s4st bk16, but TN doubled (8→16) with BN doubled
 * (128→256) so THREADS stays at 256, LCOLS=LROWS=16 unchanged.
 *
 * Per kk iteration: 8 A scalar loads + 16 B scalar loads + 128 FMAs.
 * FMA:load ratio = 128/24 = 5.33  (vs s4st tn8: 64/16 = 4.0).
 *
 * Register budget (255 max per thread, 1 block/SM at 256 threads):
 *   acc[8][16] = 128 (always live)
 *   UNROLL copies of a[8]+b[16] = UNROLL×24
 *   A_addr + B_addr = 2 (loop-carried, advance by constants per kk)
 *   other ≈ 15
 *   UNROLL=1: 169  UNROLL=2: 193  UNROLL=4: 241  UNROLL=8: 337(spill)
 *   → max viable UNROLL=4.
 *
 * Smem: A_shared[2][128][16]=16KB, B_shared[2][16][256]=32KB → 48KB static ✓
 *
 * COMPUTE_TILE key insight:
 *   A_addr = smem ptr to A_shared[buf][lty][0], advances +4 bytes per kk (one column).
 *   B_addr = smem ptr to B_shared[buf][0][ltx], advances +BN*4=1024 bytes per kk (one row).
 *   All A offsets: i * LROWS * BK * 4 = i * 1024 bytes  (i=0..7, compile-time)
 *   All B offsets: j * LCOLS     * 4 = j *   64 bytes  (j=0..15, compile-time)
 *   → inline PTX ld.shared.f32 with immediate byte offsets; no address registers wasted.
 *
 * Bank conflicts (same analysis as s4st bk16):
 *   B: LCOLS=16 threads in warp × 1 float = 16 consecutive banks → zero conflicts.
 *   A: lty ∈ {0,1} per warp → banks kk and (kk+16)%32 → zero conflicts.
 */

// Inline PTX scalar smem load with compile-time immediate byte offset.
// 'base' is a uint32_t 32-bit smem address; 'byte_off' must be a literal integer.
#define LD_S(reg, base, byte_off) \
    asm("ld.shared.f32 %0, [%1+" #byte_off "];" : "=f"(reg) : "r"(base))

template <int BM, int BN, int BK, int TM, int TN, int UNROLL>
__device__ __forceinline__ void matmul_s4st_tn16_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    constexpr int THREADS = (BM / TM) * (BN / TN);   // 256
    constexpr int LCOLS   = BN / TN;                  // 16
    constexpr int LROWS   = BM / TM;                  // 16

    // Global→smem load tiling (float4 cp.async, same pattern as s4st)
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

    // ── global → smem double-buffer issue (identical to s4st) ──────────────
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

    // ── COMPUTE_TILE with PTX base-pointer + immediate offsets ──────────────
    //
    // A_addr → A_shared[buf][lty][0]         (kk=0 column, lty's row)
    // B_addr → B_shared[buf][0][ltx]         (kk=0 row,    ltx's col)
    //
    // A byte offsets from A_addr:  i * LROWS * BK * 4  =  i * 1024
    //    i=0: 0   i=1: 1024  i=2: 2048  i=3: 3072
    //    i=4: 4096  i=5: 5120  i=6: 6144  i=7: 7168
    //
    // B byte offsets from B_addr:  j * LCOLS * 4  =  j * 64
    //    j=0:0  j=1:64  j=2:128  j=3:192  j=4:256  j=5:320  j=6:384  j=7:448
    //    j=8:512 j=9:576 j=10:640 j=11:704 j=12:768 j=13:832 j=14:896 j=15:960
    //
    // After each kk: A_addr += 4 (next column), B_addr += BN*4 (next row).
    //
    // With _Pragma("unroll UNROLL"), the compiler keeps UNROLL copies of
    // {a0..a7, b0..b15} simultaneously — each copy is 24 registers.
    // The acc[8][16]=128 registers are live throughout.
#define COMPUTE_TILE(buf_)                                                                  \
    do {                                                                                    \
        uint32_t A_addr = __cvta_generic_to_shared(&A_shared[(buf_)][lty][0]);             \
        uint32_t B_addr = __cvta_generic_to_shared(&B_shared[(buf_)][0][ltx]);             \
        _Pragma("unroll UNROLL")                                                            \
        for (int _kk = 0; _kk < BK; _kk++) {                                               \
            float a0, a1, a2, a3, a4, a5, a6, a7;                                          \
            LD_S(a0, A_addr,    0);  LD_S(a1, A_addr, 1024);                               \
            LD_S(a2, A_addr, 2048);  LD_S(a3, A_addr, 3072);                               \
            LD_S(a4, A_addr, 4096);  LD_S(a5, A_addr, 5120);                               \
            LD_S(a6, A_addr, 6144);  LD_S(a7, A_addr, 7168);                               \
            float b0,  b1,  b2,  b3,  b4,  b5,  b6,  b7;                                  \
            float b8,  b9,  b10, b11, b12, b13, b14, b15;                                  \
            LD_S(b0,  B_addr,   0);  LD_S(b1,  B_addr,  64);                               \
            LD_S(b2,  B_addr, 128);  LD_S(b3,  B_addr, 192);                               \
            LD_S(b4,  B_addr, 256);  LD_S(b5,  B_addr, 320);                               \
            LD_S(b6,  B_addr, 384);  LD_S(b7,  B_addr, 448);                               \
            LD_S(b8,  B_addr, 512);  LD_S(b9,  B_addr, 576);                               \
            LD_S(b10, B_addr, 640);  LD_S(b11, B_addr, 704);                               \
            LD_S(b12, B_addr, 768);  LD_S(b13, B_addr, 832);                               \
            LD_S(b14, B_addr, 896);  LD_S(b15, B_addr, 960);                               \
            A_addr += 4;                                                                    \
            B_addr += BN * 4;                                                               \
            acc[0][ 0]+=a0*b0;  acc[0][ 1]+=a0*b1;  acc[0][ 2]+=a0*b2;  acc[0][ 3]+=a0*b3;  \
            acc[0][ 4]+=a0*b4;  acc[0][ 5]+=a0*b5;  acc[0][ 6]+=a0*b6;  acc[0][ 7]+=a0*b7;  \
            acc[0][ 8]+=a0*b8;  acc[0][ 9]+=a0*b9;  acc[0][10]+=a0*b10; acc[0][11]+=a0*b11;  \
            acc[0][12]+=a0*b12; acc[0][13]+=a0*b13; acc[0][14]+=a0*b14; acc[0][15]+=a0*b15;  \
            acc[1][ 0]+=a1*b0;  acc[1][ 1]+=a1*b1;  acc[1][ 2]+=a1*b2;  acc[1][ 3]+=a1*b3;  \
            acc[1][ 4]+=a1*b4;  acc[1][ 5]+=a1*b5;  acc[1][ 6]+=a1*b6;  acc[1][ 7]+=a1*b7;  \
            acc[1][ 8]+=a1*b8;  acc[1][ 9]+=a1*b9;  acc[1][10]+=a1*b10; acc[1][11]+=a1*b11;  \
            acc[1][12]+=a1*b12; acc[1][13]+=a1*b13; acc[1][14]+=a1*b14; acc[1][15]+=a1*b15;  \
            acc[2][ 0]+=a2*b0;  acc[2][ 1]+=a2*b1;  acc[2][ 2]+=a2*b2;  acc[2][ 3]+=a2*b3;  \
            acc[2][ 4]+=a2*b4;  acc[2][ 5]+=a2*b5;  acc[2][ 6]+=a2*b6;  acc[2][ 7]+=a2*b7;  \
            acc[2][ 8]+=a2*b8;  acc[2][ 9]+=a2*b9;  acc[2][10]+=a2*b10; acc[2][11]+=a2*b11;  \
            acc[2][12]+=a2*b12; acc[2][13]+=a2*b13; acc[2][14]+=a2*b14; acc[2][15]+=a2*b15;  \
            acc[3][ 0]+=a3*b0;  acc[3][ 1]+=a3*b1;  acc[3][ 2]+=a3*b2;  acc[3][ 3]+=a3*b3;  \
            acc[3][ 4]+=a3*b4;  acc[3][ 5]+=a3*b5;  acc[3][ 6]+=a3*b6;  acc[3][ 7]+=a3*b7;  \
            acc[3][ 8]+=a3*b8;  acc[3][ 9]+=a3*b9;  acc[3][10]+=a3*b10; acc[3][11]+=a3*b11;  \
            acc[3][12]+=a3*b12; acc[3][13]+=a3*b13; acc[3][14]+=a3*b14; acc[3][15]+=a3*b15;  \
            acc[4][ 0]+=a4*b0;  acc[4][ 1]+=a4*b1;  acc[4][ 2]+=a4*b2;  acc[4][ 3]+=a4*b3;  \
            acc[4][ 4]+=a4*b4;  acc[4][ 5]+=a4*b5;  acc[4][ 6]+=a4*b6;  acc[4][ 7]+=a4*b7;  \
            acc[4][ 8]+=a4*b8;  acc[4][ 9]+=a4*b9;  acc[4][10]+=a4*b10; acc[4][11]+=a4*b11;  \
            acc[4][12]+=a4*b12; acc[4][13]+=a4*b13; acc[4][14]+=a4*b14; acc[4][15]+=a4*b15;  \
            acc[5][ 0]+=a5*b0;  acc[5][ 1]+=a5*b1;  acc[5][ 2]+=a5*b2;  acc[5][ 3]+=a5*b3;  \
            acc[5][ 4]+=a5*b4;  acc[5][ 5]+=a5*b5;  acc[5][ 6]+=a5*b6;  acc[5][ 7]+=a5*b7;  \
            acc[5][ 8]+=a5*b8;  acc[5][ 9]+=a5*b9;  acc[5][10]+=a5*b10; acc[5][11]+=a5*b11;  \
            acc[5][12]+=a5*b12; acc[5][13]+=a5*b13; acc[5][14]+=a5*b14; acc[5][15]+=a5*b15;  \
            acc[6][ 0]+=a6*b0;  acc[6][ 1]+=a6*b1;  acc[6][ 2]+=a6*b2;  acc[6][ 3]+=a6*b3;  \
            acc[6][ 4]+=a6*b4;  acc[6][ 5]+=a6*b5;  acc[6][ 6]+=a6*b6;  acc[6][ 7]+=a6*b7;  \
            acc[6][ 8]+=a6*b8;  acc[6][ 9]+=a6*b9;  acc[6][10]+=a6*b10; acc[6][11]+=a6*b11;  \
            acc[6][12]+=a6*b12; acc[6][13]+=a6*b13; acc[6][14]+=a6*b14; acc[6][15]+=a6*b15;  \
            acc[7][ 0]+=a7*b0;  acc[7][ 1]+=a7*b1;  acc[7][ 2]+=a7*b2;  acc[7][ 3]+=a7*b3;  \
            acc[7][ 4]+=a7*b4;  acc[7][ 5]+=a7*b5;  acc[7][ 6]+=a7*b6;  acc[7][ 7]+=a7*b7;  \
            acc[7][ 8]+=a7*b8;  acc[7][ 9]+=a7*b9;  acc[7][10]+=a7*b10; acc[7][11]+=a7*b11;  \
            acc[7][12]+=a7*b12; acc[7][13]+=a7*b13; acc[7][14]+=a7*b14; acc[7][15]+=a7*b15;  \
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

    // Strided writeback — thread (lty, ltx) owns rows lty+i*LROWS, cols ltx+j*LCOLS
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

#define MAKE_LAUNCHER(NAME, BM, BN, BK, TM, TN, UNROLL)                    \
extern "C" __global__ void NAME(                                            \
    const float* __restrict__ A, const float* __restrict__ B,              \
    float* __restrict__ C, int M, int K, int N) {                           \
    matmul_s4st_tn16_impl<BM, BN, BK, TM, TN, UNROLL>(A, B, C, M, K, N); \
}

//                NAME                                            BM   BN  BK TM  TN  U
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_tm8_tn16_bm128_bn256_bk16_u1,  128,256,16, 8,16, 1)
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_tm8_tn16_bm128_bn256_bk16_u2,  128,256,16, 8,16, 2)
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_tm8_tn16_bm128_bn256_bk16_u4,  128,256,16, 8,16, 4)
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_tm8_tn16_bm128_bn256_bk16_u8,  128,256,16, 8,16, 8)
MAKE_LAUNCHER(matmul_cuda_s4st_tn16_tm8_tn16_bm128_bn256_bk16_u16, 128,256,16, 8,16,16)

#undef MAKE_LAUNCHER
