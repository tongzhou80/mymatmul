#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>

/*
 * TC2c: BF16 raw-PTX mma.sync, BK=64, symmetric A+B XOR swizzle.
 *
 * At BK=64 the A-tile row stride is 64×2 = 128 bytes = 32 banks, so every
 * row wraps back to bank 0 — exactly the same pathology as the B tile when
 * BN×2 is also a multiple of 128 bytes.  Both tiles therefore use the
 * identical per-row swizzle:
 *
 *   physical_chunk = logical_chunk ^ (row % SWZ)
 *   A: SWZ = BK/8  B: SWZ = BN/8
 *
 * This is simpler than TC2b's general formula (which divides row by A_SHIFT
 * to handle the shorter conflict period at BK=16/32).  For BK=64, A_SHIFT=1
 * so the division disappears and both tiles share the same one-liner.
 *
 * Template: (BM, BN, NUM_WARPS) — BK is fixed at 64.
 * Config space: BM, BN ∈ {64,128,256}, NUM_WARPS ∈ {4,8},
 *               smem(BM,BN,64) = (2*BM*64 + 2*64*BN)*2 ≤ 100352 bytes,
 *               BM*BN ≤ 4096*NUM_WARPS.
 */

// ── PTX helpers ──────────────────────────────────────────────────────────────

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    uint32_t smem_ptr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& r0, uint32_t& r1,
    uint32_t smem_ptr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

// ── Kernel implementation ─────────────────────────────────────────────────────

template <int BM, int BN, int NUM_WARPS>
__device__ __forceinline__ void matmul_tc2c_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    constexpr int BK = 64;

    constexpr int WARP_N = 2;
    constexpr int WARP_M = NUM_WARPS / WARP_N;

    constexpr int WARP_TILE_M = BM / WARP_M;
    constexpr int WARP_TILE_N = BN / WARP_N;

    constexpr int WM_TILES = WARP_TILE_M / 16;
    constexpr int WN_TILES = WARP_TILE_N / 8;

    constexpr int THREADS = NUM_WARPS * 32;

    constexpr int A_ELEM   = (BM * BK / THREADS >= 8) ? 8 : 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = (BK * BN / THREADS >= 8) ? 8 : 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    // Both tiles: every row aliases to bank 0 → same per-row XOR swizzle.
    //   physical_chunk = logical_chunk ^ (row % SWZ)
    constexpr int A_SWZ = BK / 8;   // = 8
    constexpr int B_SWZ = BN / 8;

    extern __shared__ __nv_bfloat16 smem[];
    auto A_shared = reinterpret_cast<__nv_bfloat16 (*)[BM][BK]>(smem);
    auto B_shared = reinterpret_cast<__nv_bfloat16 (*)[BK][BN]>(smem + 2 * BM * BK);

    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid % 32;
    const int warp_row = warp_id / WARP_N;
    const int warp_col = warp_id % WARP_N;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[WM_TILES][WN_TILES][4] = {};

// ── Async load: both tiles use symmetric per-row XOR swizzle ─────────────────
//   physical_chunk = (logical_chunk) ^ (row % SWZ)
#define ISSUE_TILE(k0_, buf_)                                                       \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                    \
            const int _g  = tid + _i * THREADS;                                    \
            const int _r  = (_g * A_ELEM) / BK;                                   \
            const int _c  = (_g * A_ELEM) % BK;                                   \
            const int _sc = ((_c / 8) ^ (_r % A_SWZ)) * 8 + (_c % 8);            \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_sc],                     \
                                    &A[(block_row + _r) * K + (k0_) + _c],        \
                                    A_ELEM * (int)sizeof(__nv_bfloat16));           \
        }                                                                           \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                    \
            const int _g  = tid + _i * THREADS;                                    \
            const int _r  = (_g * B_ELEM) / BN;                                   \
            const int _c  = (_g * B_ELEM) % BN;                                   \
            const int _sc = ((_c / 8) ^ (_r % B_SWZ)) * 8 + (_c % 8);            \
            __pipeline_memcpy_async(&B_shared[(buf_)][_r][_sc],                    \
                                    &B[((k0_) + _r) * N + block_col + _c],        \
                                    B_ELEM * (int)sizeof(__nv_bfloat16));           \
        }                                                                           \
        __pipeline_commit();                                                         \
    } while (0)

// ── Compute: ldmatrix + mma.sync, outer-product order ────────────────────────
// A: logical_chunk = kk*2 + (lane/16); un-swizzle: physical = logical ^ (row % A_SWZ)
// B: same per-row formula as TC2.
#define COMPUTE_TILE(buf_)                                                          \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                 \
            uint32_t _fa[WM_TILES][4];                                              \
            _Pragma("unroll")                                                       \
            for (int _mt = 0; _mt < WM_TILES; _mt++) {                             \
                const int _ar   = warp_row * WARP_TILE_M + _mt * 16 + (lane % 16);\
                const int _lg   = _kk * 2 + (lane / 16);                           \
                const int _phys = _lg ^ (_ar % A_SWZ);                             \
                ldmatrix_x4(_fa[_mt][0], _fa[_mt][1], _fa[_mt][2], _fa[_mt][3],   \
                    __cvta_generic_to_shared(&A_shared[(buf_)][_ar][_phys * 8]));  \
            }                                                                       \
            uint32_t _fb[WN_TILES][2];                                              \
            _Pragma("unroll")                                                       \
            for (int _nt = 0; _nt < WN_TILES; _nt++) {                             \
                const int _br = _kk * 16 + (lane % 16);                            \
                const int _nc = warp_col * WARP_TILE_N + _nt * 8;                  \
                const int _sc = ((_nc / 8) ^ (_br % B_SWZ)) * 8;                  \
                ldmatrix_x2_trans(_fb[_nt][0], _fb[_nt][1],                        \
                    __cvta_generic_to_shared(&B_shared[(buf_)][_br][_sc]));         \
            }                                                                       \
            _Pragma("unroll")                                                       \
            for (int _mt = 0; _mt < WM_TILES; _mt++) {                             \
                _Pragma("unroll")                                                   \
                for (int _nt = 0; _nt < WN_TILES; _nt++) {                         \
                    mma_m16n8k16(acc[_mt][_nt][0], acc[_mt][_nt][1],               \
                                 acc[_mt][_nt][2], acc[_mt][_nt][3],               \
                                 _fa[_mt][0], _fa[_mt][1],                         \
                                 _fa[_mt][2], _fa[_mt][3],                         \
                                 _fb[_nt][0], _fb[_nt][1]);                        \
                }                                                                   \
            }                                                                       \
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

    constexpr int row_off[4] = {0, 0, 8, 8};
    constexpr int col_off[4] = {0, 1, 0, 1};
    const int base_row = lane / 4;
    const int base_col = (lane % 4) * 2;

    #pragma unroll
    for (int mt = 0; mt < WM_TILES; mt++) {
        #pragma unroll
        for (int nt = 0; nt < WN_TILES; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int gr = block_row + warp_row * WARP_TILE_M + mt * 16
                               + base_row + row_off[e];
                const int gc = block_col + warp_col * WARP_TILE_N + nt * 8
                               + base_col + col_off[e];
                if (gr < M && gc < N)
                    C[gr * N + gc] = __float2bfloat16(acc[mt][nt][e]);
            }
        }
    }
}

// BK=64 is baked in; template only over (BM, BN, NUM_WARPS).
#define MAKE_LAUNCHER(BM_, BN_, NW_)                                                \
extern "C" __global__ __launch_bounds__(NW_ * 32)                                   \
void matmul_cuda_tc2c_bm##BM_##_bn##BN_##_bk64_nw##NW_(                            \
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,       \
    __nv_bfloat16* __restrict__ C, int M, int K, int N) {                           \
    matmul_tc2c_impl<BM_, BN_, NW_>(A, B, C, M, K, N);                             \
}

// smem(BM,BN,64) = (2*BM*64 + 2*64*BN)*2 = 256*(BM+BN) bytes
// ── NW=4 (smem ≤ 100352, BM*BN ≤ 16384) ─────────────────────────────────────
MAKE_LAUNCHER( 64,  64, 4)   //  32 KB
MAKE_LAUNCHER( 64, 128, 4)   //  48 KB
MAKE_LAUNCHER( 64, 256, 4)   //  80 KB
MAKE_LAUNCHER(128,  64, 4)   //  48 KB
MAKE_LAUNCHER(128, 128, 4)   //  64 KB
MAKE_LAUNCHER(256,  64, 4)   //  80 KB

// ── NW=8 (smem ≤ 100352, BM*BN ≤ 32768) ─────────────────────────────────────
MAKE_LAUNCHER( 64,  64, 8)   //  32 KB
MAKE_LAUNCHER( 64, 128, 8)   //  48 KB
MAKE_LAUNCHER( 64, 256, 8)   //  80 KB
MAKE_LAUNCHER(128,  64, 8)   //  48 KB
MAKE_LAUNCHER(128, 128, 8)   //  64 KB
MAKE_LAUNCHER(128, 256, 8)   //  98 KB
MAKE_LAUNCHER(256,  64, 8)   //  80 KB
MAKE_LAUNCHER(256, 128, 8)   //  98 KB

#undef MAKE_LAUNCHER
