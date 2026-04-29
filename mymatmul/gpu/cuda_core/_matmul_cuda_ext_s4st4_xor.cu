#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

/*
 * Stage 4 Strided-4 XOR (s4st4_xor): float4 B reads + 8×4 warp layout + XOR swizzle on A.
 *
 * Why 8×4 warp layout is required for float4 B loads (zero conflicts):
 *   s4st2 uses 16×2 warp layout → 16 unique ltx per warp. float4 reads: 16×4=64 consecutive
 *   floats → wraps 32 banks twice → 2-way B conflicts. Fix: 8×4 layout → 8 unique ltx per
 *   warp. float4: 8×4=32 floats → exactly 32 banks → zero B conflicts.
 *
 * Thread mapping (THREADS=256, BM=128, BN=128, TM=8, TN=8):
 *   warp_id = tid / 32;  lane = tid % 32
 *   LCOLS_W = LCOLS/2 = 8  (unique ltx values per warp)
 *   ltx = (warp_id % 2) * LCOLS_W + lane % LCOLS_W  → 0..15
 *   lty = (warp_id / 2) * 4        + lane / LCOLS_W  → 0..15
 *
 *   Warp 0 example: ltx ∈ {0..7}, lty ∈ {0,1,2,3} (each ltx appears 4 times)
 *
 * B read layout (TN=8, TN/4=2 steps, stride = 4*LCOLS = 64):
 *   Thread ltx owns 4-element groups at cols: {4*ltx..4*ltx+3}, {4*ltx+64..4*ltx+67}
 *   Step j: float4 from B_shared[kk][4*ltx + j*4*LCOLS]
 *   Per-warp bank analysis (e.g., warp 0, ltx=0..7):
 *     j=0: cols 0,4,...,28 → 8 float4s = 32 floats, banks 0..31 → zero conflicts ✓
 *     j=1: cols 64,68,...,92 → same 32-bank pattern ✓
 *
 * A bank conflict analysis (BK=16, warp 0: lty ∈ {0,1,2,3}, rows = lty at _i=0):
 *   Without swizzle: bank(A_shared[row][kk]) = (row*16 + kk) % 32
 *     row=0: bank = kk;        row=1: bank = (kk+16)%32
 *     row=2: bank = kk ← conflict with row=0;  row=3: bank = (kk+16)%32 ← conflict with row=1
 *   → 2-way A conflict (rows {0,2} alias, {1,3} alias).
 *
 * XOR swizzle fix: physical_col = logical_col XOR ((row & 2) * 4)
 *   row=0: XOR 0;  row=1: XOR 0;  row=2: XOR 8;  row=3: XOR 8
 *   Banks with swizzle (kk=0 example):
 *     row=0: physical=0, bank=0
 *     row=1: physical=0, bank=16
 *     row=2: physical=8, bank=(32+8)%32=8   → distinct from rows 0,1 ✓
 *     row=3: physical=8, bank=(48+8)%32=24  → distinct from all ✓
 *   → Zero A conflicts. XOR swizzle by 8 (= BK/2 positions) is the standard phase pattern.
 *   16-byte alignment preserved: XOR is by 8 floats (32 bytes), all offsets remain aligned.
 */
template <int BM, int BN, int BK, int TM, int TN, int UNROLL>
__device__ __forceinline__ void matmul_s4st4_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    static_assert(TN % 4 == 0, "TN must be divisible by 4 for float4 B loads");

    constexpr int THREADS = (BM / TM) * (BN / TN);
    constexpr int LCOLS   = BN / TN;   // = 16 for bm128_bn128 tm8_tn8
    constexpr int LROWS   = BM / TM;   // = 16

    // 8×4 warp layout: LCOLS_W = LCOLS/2 = 8 (ltx per warp along N)
    // (LCOLS must be even for 8×4 split)
    constexpr int LCOLS_W = LCOLS / 2;  // = 8

    // A and B global load tiling
    constexpr int A_THREAD_BYTES = BM * BK * (int)sizeof(float) / THREADS;
    constexpr int A_LOAD_BYTES   = (A_THREAD_BYTES >= 16) ? 16 : (A_THREAD_BYTES >= 8) ? 8 : 4;
    constexpr int A_ELEM         = A_LOAD_BYTES / (int)sizeof(float);
    constexpr int A_GROUPS       = BM * BK / A_ELEM / THREADS;

    constexpr int B_THREAD_BYTES = BK * BN * (int)sizeof(float) / THREADS;
    constexpr int B_LOAD_BYTES   = (B_THREAD_BYTES >= 16) ? 16 : (B_THREAD_BYTES >= 8) ? 8 : 4;
    constexpr int B_ELEM         = B_LOAD_BYTES / (int)sizeof(float);
    constexpr int B_GROUPS       = BK * BN / B_ELEM / THREADS;

    __shared__ float A_shared[2][BM][BK];
    __shared__ float B_shared[2][BK][BN];

    const int tx  = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // 8×4 warp layout
    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int ltx = (warp_id % 2) * LCOLS_W + lane % LCOLS_W;   // 0..LCOLS-1
    const int lty = (warp_id / 2) * 4        + lane / LCOLS_W;  // 0..LROWS-1

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};

    // ISSUE_TILE: global→shared with XOR swizzle on A destination
    // physical_col = logical_col XOR ((row & 2) * 4)
#define ISSUE_TILE(k0_, buf_)                                                               \
    do {                                                                                    \
        _Pragma("unroll")                                                                   \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                            \
            const int _g = tid + _i * THREADS;                                             \
            const int _r = (_g * A_ELEM) / BK, _c = (_g * A_ELEM) % BK;                  \
            const int _cp = _c ^ ((_r & 2) * 4);                                           \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_cp],                             \
                                    &A[(block_row + _r) * K + (k0_) + _c],                 \
                                    A_LOAD_BYTES);                                          \
        }                                                                                   \
        _Pragma("unroll")                                                                   \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                            \
            const int _g = tid + _i * THREADS;                                             \
            const int _r = (_g * B_ELEM) / BN, _c = (_g * B_ELEM) % BN;                  \
            __pipeline_memcpy_async(&B_shared[(buf_)][_r][_c],                             \
                                    &B[((k0_) + _r) * N + block_col + _c],                 \
                                    B_LOAD_BYTES);                                          \
        }                                                                                   \
        __pipeline_commit();                                                                \
    } while (0)

    // COMPUTE_TILE: A load with XOR swizzle, B load as float4 (TN/4 steps)
#define COMPUTE_TILE(buf_)                                                                  \
    do {                                                                                    \
        _Pragma("unroll UNROLL")                                                            \
        for (int _kk = 0; _kk < BK; _kk++) {                                               \
            float _a[TM];                                                                   \
            _Pragma("unroll")                                                               \
            for (int _i = 0; _i < TM; _i++) {                                              \
                const int _row = lty + _i * LROWS;                                          \
                _a[_i] = A_shared[(buf_)][_row][_kk ^ ((_row & 2) * 4)];                   \
            }                                                                               \
            _Pragma("unroll")                                                               \
            for (int _j = 0; _j < TN / 4; _j++) {                                          \
                float4 _bv = *reinterpret_cast<const float4*>(                              \
                    &B_shared[(buf_)][_kk][4 * ltx + _j * 4 * LCOLS]);                     \
                _Pragma("unroll")                                                           \
                for (int _i = 0; _i < TM; _i++) {                                          \
                    acc[_i][4 * _j + 0] += _a[_i] * _bv.x;                                 \
                    acc[_i][4 * _j + 1] += _a[_i] * _bv.y;                                 \
                    acc[_i][4 * _j + 2] += _a[_i] * _bv.z;                                 \
                    acc[_i][4 * _j + 3] += _a[_i] * _bv.w;                                 \
                }                                                                           \
            }                                                                               \
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

    // 4-contiguous writeback:
    // thread (lty, ltx) owns 4-element groups at:
    //   row: lty + i*LROWS,  cols: 4*ltx + j*4*LCOLS_W  ..  4*ltx + j*4*LCOLS_W + 3
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN / 4; j++) {
            const int gr = block_row + lty + i * LROWS;
            const int gc = block_col + 4 * ltx + j * 4 * LCOLS;
            if (gr < M && gc + 3 < N) {
                *reinterpret_cast<float4*>(&C[gr * N + gc]) =
                    make_float4(acc[i][4*j], acc[i][4*j+1], acc[i][4*j+2], acc[i][4*j+3]);
            } else if (gr < M) {
                for (int k = 0; k < 4 && gc + k < N; k++)
                    C[gr * N + gc + k] = acc[i][4*j+k];
            }
        }
}

#define MAKE_LAUNCHER_S4ST4_XOR(NAME, BM, BN, BK, TM, TN, UNROLL)                   \
extern "C" __global__ void NAME(                                                     \
    const float* __restrict__ A, const float* __restrict__ B,                       \
    float* __restrict__ C, int M, int K, int N) {                                    \
    matmul_s4st4_impl<BM, BN, BK, TM, TN, UNROLL>(A, B, C, M, K, N);               \
}

//                                   NAME                                         BM   BN  BK  TM  TN  UNROLL
MAKE_LAUNCHER_S4ST4_XOR(matmul_cuda_s4st4_xor_tm8_tn8_bm128_bn128_bk16_u1,      128, 128, 16,  8,  8,   1)
MAKE_LAUNCHER_S4ST4_XOR(matmul_cuda_s4st4_xor_tm8_tn8_bm128_bn128_bk16_u4,      128, 128, 16,  8,  8,   4)
MAKE_LAUNCHER_S4ST4_XOR(matmul_cuda_s4st4_xor_tm8_tn8_bm128_bn128_bk16_u8,      128, 128, 16,  8,  8,   8)
MAKE_LAUNCHER_S4ST4_XOR(matmul_cuda_s4st4_xor_tm8_tn8_bm128_bn128_bk16_u16,     128, 128, 16,  8,  8,  16)
