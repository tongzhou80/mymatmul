#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>

/*
 * TC2: BF16 raw-PTX mma.sync with XOR-swizzled B tile.
 *
 * Builds on TC1 (WMMA) with two changes:
 *
 * 1. Raw PTX replaces the high-level WMMA API.
 *    - A load : ldmatrix.x4.m8n8 (no-trans, row-major → A register layout).
 *    - B load : ldmatrix.x2.m8n8.trans (row-major smem → col-major register
 *               layout required by mma.sync.row.col, transposed by hardware).
 *    - Compute: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 (two
 *               calls per 16×16 warp tile: one for each 8-wide N half).
 *
 * 2. B tile XOR-swizzled in shared memory to reduce bank conflicts.
 *    Canonical swizzle on 8-bf16 chunks (= one ldmatrix row, 16 bytes):
 *      physical_col_chunk = logical_col_chunk ^ (row % (BN/8))
 *    Write (ISSUE_TILE): cp.async destination uses the swizzled address.
 *    Read (COMPUTE_TILE): ldmatrix pointer un-swizzles using the same formula.
 *    Effect: the 16 consecutive k-rows covered by one ldmatrix.x2 call each
 *    land in a different column chunk → at most 2-way bank conflict (the
 *    hardware optimum for 16 × 16-byte loads across 32 banks) vs. 16-way
 *    without swizzling.
 *
 * Accumulator layout: float acc[WM_TILES][WN_TILES][8], identical to TC1's
 * wmma fragment.  Elements [0..3] come from the first mma call (N-cols 0..7)
 * and [4..7] from the second (N-cols 8..15), matching TC1's row_off/col_off
 * writeback pattern exactly.
 *
 * Template: (BM, BN, BK, NUM_WARPS) — same search space as TC1.
 * Constraints: BK % 16 == 0, BN % 8 == 0.
 */

// ── PTX helpers ──────────────────────────────────────────────────────────────

// ldmatrix.x4 for A (16×16 bf16, row-major → 4 uint32 per thread)
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

// ldmatrix.x2.trans for B (16×8 bf16 row-major in smem → col-major registers)
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

// mma.sync m16n8k16 bf16→f32 accumulate (d += a*b; d also serves as c input)
__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    // "+f" = read-write: {%0..%3} are both C (input) and D (output).
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

// ── Kernel implementation ─────────────────────────────────────────────────────

template <int BM, int BN, int BK, int NUM_WARPS>
__device__ __forceinline__ void matmul_tc2_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    constexpr int WARP_N = 2;
    constexpr int WARP_M = NUM_WARPS / WARP_N;

    constexpr int WARP_TILE_M = BM / WARP_M;
    constexpr int WARP_TILE_N = BN / WARP_N;

    constexpr int WM_TILES = WARP_TILE_M / 16;
    constexpr int WN_TILES = WARP_TILE_N / 8;   // native mma width: one mma per nt

    constexpr int THREADS = NUM_WARPS * 32;

    // cp.async width: 8 bf16 (16 B) when the tile is large enough, else 4 bf16 (8 B).
    constexpr int A_ELEM   = (BM * BK / THREADS >= 8) ? 8 : 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = (BK * BN / THREADS >= 8) ? 8 : 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    // XOR swizzle period: number of 8-bf16 column chunks per B row.
    constexpr int B_SWZ = BN / 8;

    extern __shared__ __nv_bfloat16 smem[];
    auto A_shared = reinterpret_cast<__nv_bfloat16 (*)[BM][BK]>(smem);
    // B_shared: same [2][BK][BN] layout as TC1; data physically XOR-swizzled.
    auto B_shared = reinterpret_cast<__nv_bfloat16 (*)[BK][BN]>(smem + 2 * BM * BK);

    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid % 32;
    const int warp_row = warp_id / WARP_N;
    const int warp_col = warp_id % WARP_N;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // acc[mt][nt][4]: 4 f32 per 16x8 mma output tile.
    //   nt indexes 8-wide N chunks directly — one mma.sync per (mt, nt, kk).
    float acc[WM_TILES][WN_TILES][4] = {};

// ── Async load: A flat, B XOR-swizzled ───────────────────────────────────────
// Swizzle formula (8-bf16 granularity):
//   physical_col = (logical_col/8 ^ (row % B_SWZ)) * 8 + logical_col%8
// Works for both B_ELEM=4 and B_ELEM=8 since logical_col is always 4-aligned.
#define ISSUE_TILE(k0_, buf_)                                                       \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                    \
            const int _g = tid + _i * THREADS;                                     \
            const int _r = (_g * A_ELEM) / BK, _c = (_g * A_ELEM) % BK;          \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_c],                     \
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

// ── Compute: ldmatrix + mma.sync, outer-product order (kk outer) ─────────────
// A: ldmatrix.x4 — threads 0..15 address left half (col kk*16),
//                  threads 16..31 address right half (col kk*16+8).
// B: ldmatrix.x2.trans — threads 0..15 address rows kk*16..kk*16+15 with
//    swizzled column; threads 16..31 mirror 0..15 (unused by hardware).
// _nt indexes 8-wide N chunks directly: _nc = warp_col*WARP_TILE_N + _nt*8.
#define COMPUTE_TILE(buf_)                                                          \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                 \
            uint32_t _fa[WM_TILES][4];                                              \
            _Pragma("unroll")                                                       \
            for (int _mt = 0; _mt < WM_TILES; _mt++) {                             \
                const int _ar = warp_row * WARP_TILE_M + _mt * 16 + (lane % 16);  \
                const int _ac = _kk * 16 + (lane / 16) * 8;                       \
                ldmatrix_x4(_fa[_mt][0], _fa[_mt][1], _fa[_mt][2], _fa[_mt][3],   \
                    __cvta_generic_to_shared(&A_shared[(buf_)][_ar][_ac]));         \
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

    // Write-back: mma.sync m16n8 output layout.
    //   Thread t: base_row = t/4, base_col = 2*(t%4)
    //   e=0: row+0,col+0  e=1: row+0,col+1  e=2: row+8,col+0  e=3: row+8,col+1
    //   nt indexes 8-wide N chunks: gc base = warp_col*WARP_TILE_N + nt*8
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

#define MAKE_LAUNCHER(BM_, BN_, BK_, NW_)                                           \
extern "C" __global__ __launch_bounds__(NW_ * 32)                                   \
void matmul_cuda_tc2_bm##BM_##_bn##BN_##_bk##BK_##_nw##NW_(                        \
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,       \
    __nv_bfloat16* __restrict__ C, int M, int K, int N) {                           \
    matmul_tc2_impl<BM_, BN_, BK_, NW_>(A, B, C, M, K, N);                         \
}

// ── NW=4 (128 threads, 2×2 inter-warp) ───────────────────────────────────────
MAKE_LAUNCHER( 64,  64, 16, 4) MAKE_LAUNCHER( 64,  64, 32, 4)
MAKE_LAUNCHER( 64, 128, 16, 4) MAKE_LAUNCHER( 64, 128, 32, 4)
MAKE_LAUNCHER( 64, 256, 16, 4) MAKE_LAUNCHER( 64, 256, 32, 4)
MAKE_LAUNCHER(128,  64, 16, 4) MAKE_LAUNCHER(128,  64, 32, 4)
MAKE_LAUNCHER(128, 128, 16, 4) MAKE_LAUNCHER(128, 128, 32, 4)
MAKE_LAUNCHER(256,  64, 16, 4) MAKE_LAUNCHER(256,  64, 32, 4)

// ── NW=8 (256 threads, 4×2 inter-warp) ───────────────────────────────────────
MAKE_LAUNCHER( 64,  64, 16, 8) MAKE_LAUNCHER( 64,  64, 32, 8)
MAKE_LAUNCHER( 64, 128, 16, 8) MAKE_LAUNCHER( 64, 128, 32, 8)
MAKE_LAUNCHER( 64, 256, 16, 8) MAKE_LAUNCHER( 64, 256, 32, 8)
MAKE_LAUNCHER(128,  64, 16, 8) MAKE_LAUNCHER(128,  64, 32, 8)
MAKE_LAUNCHER(128, 128, 16, 8) MAKE_LAUNCHER(128, 128, 32, 8)
MAKE_LAUNCHER(128, 256, 16, 8) MAKE_LAUNCHER(128, 256, 32, 8)
MAKE_LAUNCHER(256,  64, 16, 8) MAKE_LAUNCHER(256,  64, 32, 8)
MAKE_LAUNCHER(256, 128, 16, 8) MAKE_LAUNCHER(256, 128, 32, 8)

#undef MAKE_LAUNCHER
