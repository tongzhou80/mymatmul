#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * Stage 5 W4: warp-tiled variant with float4 B smem loads.
 *
 * Changes from s5:
 *   - Inter-warp layout : 4×2 (M×N) — 8 warps partition the BM×BN block into
 *     4 rows × 2 cols of warp tiles, each warp owns (BM/4) × (BN/2) output.
 *   - Intra-warp layout : 4×8 (M×N) — each warp's 32 threads are arranged 4 rows
 *     × 8 cols; thread (intra_lty, intra_ltx) does strided assignment within its
 *     warp tile, stride LWARP_M=4 in M and stride LWARP_N×4=32 in N.
 *   - B smem load       : float4 (4 consecutive N elements per instruction) instead
 *     of float2, so TN/4 loads per kk instead of TN/2.
 *
 * Per-thread output tile: TM = BM/16, TN = BN/16 — identical to s5.
 * acc[TM][TN] layout:
 *   acc[i][j*4 .. j*4+3] = 4 consecutive N elements at
 *     M: block_row + warp_row*(BM/4) + intra_lty + i*4
 *     N: block_col + warp_col*(BN/2) + 4*intra_ltx + j*32,  j=0..TN/4-1
 *
 * Bank conflict analysis (BK=16):
 *   A_shared[r][kk]: bank=(r*16+kk)%32. Intra-warp rows differ by 1, giving
 *   bank offsets {0,16,0,16} for lty 0..3 → same 2-way conflict as s5.
 *   B_shared[kk][warp_col*(BN/2)+4*intra_ltx+j*32]: col divisible by 4 →
 *   float4 banks {4k,4k+1,4k+2,4k+3} for intra_ltx=k, all distinct → 0 conflicts.
 *
 * Constraints: BN must be divisible by 64 (= WARP_N*LWARP_N*4 = 2*8*4).
 *   BN ∈ {64,128,256} all satisfy this.
 */

template <int BM, int BN, int BK, int UNROLL>
__device__ __forceinline__ void matmul_s5_w4_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    constexpr int WARP_M  = 4;          // warp rows  (inter-warp M)
    constexpr int WARP_N  = 2;          // warp cols  (inter-warp N)
    constexpr int LWARP_M = 4;          // thread rows inside a warp (M)
    constexpr int LWARP_N = 8;          // thread cols inside a warp (N)

    constexpr int WARP_TILE_M = BM / WARP_M;   // BM/4
    constexpr int WARP_TILE_N = BN / WARP_N;   // BN/2

    constexpr int TM      = WARP_TILE_M / LWARP_M;   // BM/16  (same as s5)
    constexpr int TN      = WARP_TILE_N / LWARP_N;   // BN/16  (same as s5)
    constexpr int THREADS = 256;

    constexpr int A_ELEM   = 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    extern __shared__ float smem[];
    auto A_shared = reinterpret_cast<float (*)[BM][BK]>(smem);
    auto B_shared = reinterpret_cast<float (*)[BK][BN]>(smem + 2 * BM * BK);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Inter-warp position
    const int warp_id  = tid / 32;
    const int warp_row = warp_id / WARP_N;     // 0..3  (M)
    const int warp_col = warp_id % WARP_N;     // 0..1  (N)

    // Intra-warp position
    const int tiw       = tid % 32;
    const int intra_lty = tiw / LWARP_N;        // 0..3  (M within warp)
    const int intra_ltx = tiw % LWARP_N;        // 0..7  (N within warp)

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};

    // Tile load: all 256 threads cooperatively fill the full BM×BK A tile
    // and BK×BN B tile — same as s5, no warp-level splitting here.
#define ISSUE_TILE(k0_, buf_)                                                           \
    do {                                                                                \
        _Pragma("unroll")                                                               \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                        \
            const int _g = tid + _i * THREADS;                                         \
            const int _r = (_g * A_ELEM) / BK, _c = (_g * A_ELEM) % BK;              \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_c],                         \
                                    &A[(block_row + _r) * K + (k0_) + _c],             \
                                    A_ELEM * (int)sizeof(float));                       \
        }                                                                               \
        _Pragma("unroll")                                                               \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                        \
            const int _g = tid + _i * THREADS;                                         \
            const int _r = (_g * B_ELEM) / BN, _c = (_g * B_ELEM) % BN;              \
            __pipeline_memcpy_async(&B_shared[(buf_)][_r][_c],                         \
                                    &B[((k0_) + _r) * N + block_col + _c],             \
                                    B_ELEM * (int)sizeof(float));                       \
        }                                                                               \
        __pipeline_commit();                                                             \
    } while (0)

    // float4 B load: each thread reads 4 consecutive N elements per j-step.
    // TN/4 j-steps per kk, stride 4*LWARP_N=32 between steps.
#define COMPUTE_TILE(buf_)                                                              \
    do {                                                                                \
        _Pragma("unroll UNROLL")                                                        \
        for (int _kk = 0; _kk < BK; _kk++) {                                           \
            float _a[TM];                                                               \
            _Pragma("unroll")                                                           \
            for (int _i = 0; _i < TM; _i++)                                            \
                _a[_i] = A_shared[(buf_)][warp_row * WARP_TILE_M + intra_lty + _i * LWARP_M][_kk]; \
            _Pragma("unroll")                                                           \
            for (int _j = 0; _j < TN / 4; _j++) {                                      \
                float4 _bv = *reinterpret_cast<const float4*>(                          \
                    &B_shared[(buf_)][_kk]                                              \
                              [warp_col * WARP_TILE_N + 4 * intra_ltx + _j * (4 * LWARP_N)]); \
                _Pragma("unroll")                                                       \
                for (int _i = 0; _i < TM; _i++) {                                      \
                    acc[_i][_j * 4 + 0] += _a[_i] * _bv.x;                            \
                    acc[_i][_j * 4 + 1] += _a[_i] * _bv.y;                            \
                    acc[_i][_j * 4 + 2] += _a[_i] * _bv.z;                            \
                    acc[_i][_j * 4 + 3] += _a[_i] * _bv.w;                            \
                }                                                                       \
            }                                                                           \
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

    // Writeback: float4 stores where possible, scalar fallback at boundaries.
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN / 4; j++) {
            const int gr = block_row + warp_row * WARP_TILE_M + intra_lty + i * LWARP_M;
            const int gc = block_col + warp_col * WARP_TILE_N + 4 * intra_ltx + j * (4 * LWARP_N);
            if (gr < M && gc + 3 < N) {
                *reinterpret_cast<float4*>(&C[gr * N + gc]) =
                    make_float4(acc[i][j*4+0], acc[i][j*4+1], acc[i][j*4+2], acc[i][j*4+3]);
            } else if (gr < M) {
                for (int k = 0; k < 4 && gc + k < N; k++)
                    C[gr * N + gc + k] = acc[i][j * 4 + k];
            }
        }
}

#define MAKE_LAUNCHER(BM_, BN_, BK_, U_)                                            \
extern "C" __global__ __launch_bounds__(256)                                        \
void matmul_cuda_s5_w4_bm##BM_##_bn##BN_##_bk##BK_##_u##U_(                       \
    const float* __restrict__ A, const float* __restrict__ B,                      \
    float* __restrict__ C, int M, int K, int N) {                                  \
    matmul_s5_w4_impl<BM_, BN_, BK_, U_>(A, B, C, M, K, N);                       \
}

// BM=64
MAKE_LAUNCHER( 64,  64, 16,  2) MAKE_LAUNCHER( 64,  64, 16,  4)
MAKE_LAUNCHER( 64,  64, 16,  8) MAKE_LAUNCHER( 64,  64, 16, 16)
MAKE_LAUNCHER( 64,  64, 32,  2) MAKE_LAUNCHER( 64,  64, 32,  4)
MAKE_LAUNCHER( 64,  64, 32,  8) MAKE_LAUNCHER( 64,  64, 32, 16)
MAKE_LAUNCHER( 64, 128, 16,  2) MAKE_LAUNCHER( 64, 128, 16,  4)
MAKE_LAUNCHER( 64, 128, 16,  8) MAKE_LAUNCHER( 64, 128, 16, 16)
MAKE_LAUNCHER( 64, 128, 32,  2) MAKE_LAUNCHER( 64, 128, 32,  4)
MAKE_LAUNCHER( 64, 128, 32,  8) MAKE_LAUNCHER( 64, 128, 32, 16)
MAKE_LAUNCHER( 64, 256, 16,  2) MAKE_LAUNCHER( 64, 256, 16,  4)
MAKE_LAUNCHER( 64, 256, 16,  8) MAKE_LAUNCHER( 64, 256, 16, 16)
MAKE_LAUNCHER( 64, 256, 32,  2) MAKE_LAUNCHER( 64, 256, 32,  4)
MAKE_LAUNCHER( 64, 256, 32,  8) MAKE_LAUNCHER( 64, 256, 32, 16)

// BM=128
MAKE_LAUNCHER(128,  64, 16,  2) MAKE_LAUNCHER(128,  64, 16,  4)
MAKE_LAUNCHER(128,  64, 16,  8) MAKE_LAUNCHER(128,  64, 16, 16)
MAKE_LAUNCHER(128,  64, 32,  2) MAKE_LAUNCHER(128,  64, 32,  4)
MAKE_LAUNCHER(128,  64, 32,  8) MAKE_LAUNCHER(128,  64, 32, 16)
MAKE_LAUNCHER(128, 128, 16,  2) MAKE_LAUNCHER(128, 128, 16,  4)
MAKE_LAUNCHER(128, 128, 16,  8) MAKE_LAUNCHER(128, 128, 16, 16)
MAKE_LAUNCHER(128, 128, 32,  2) MAKE_LAUNCHER(128, 128, 32,  4)
MAKE_LAUNCHER(128, 128, 32,  8) MAKE_LAUNCHER(128, 128, 32, 16)
MAKE_LAUNCHER(128, 256, 16,  2) MAKE_LAUNCHER(128, 256, 16,  4)
MAKE_LAUNCHER(128, 256, 16,  8) MAKE_LAUNCHER(128, 256, 16, 16)
MAKE_LAUNCHER(128, 256, 32,  2) MAKE_LAUNCHER(128, 256, 32,  4)
MAKE_LAUNCHER(128, 256, 32,  8) MAKE_LAUNCHER(128, 256, 32, 16)

// BM=256
MAKE_LAUNCHER(256,  64, 16,  2) MAKE_LAUNCHER(256,  64, 16,  4)
MAKE_LAUNCHER(256,  64, 16,  8) MAKE_LAUNCHER(256,  64, 16, 16)
MAKE_LAUNCHER(256,  64, 32,  2) MAKE_LAUNCHER(256,  64, 32,  4)
MAKE_LAUNCHER(256,  64, 32,  8) MAKE_LAUNCHER(256,  64, 32, 16)
MAKE_LAUNCHER(256, 128, 16,  2) MAKE_LAUNCHER(256, 128, 16,  4)
MAKE_LAUNCHER(256, 128, 16,  8) MAKE_LAUNCHER(256, 128, 16, 16)
MAKE_LAUNCHER(256, 128, 32,  2) MAKE_LAUNCHER(256, 128, 32,  4)
MAKE_LAUNCHER(256, 128, 32,  8) MAKE_LAUNCHER(256, 128, 32, 16)
// BM=256, BN=256 excluded: acc[16][16]=256 regs → spill

#undef MAKE_LAUNCHER
