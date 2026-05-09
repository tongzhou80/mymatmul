#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>

/*
 * TC5jit_lb: TC5jit with two-arg __launch_bounds__(NW*32, LB_MIN_BLOCKS).
 *
 * Compiled with -DM_VAL=... -DK_VAL=... -DN_VAL=... -DLB_MIN_BLOCKS=...
 * per (problem size, lb) pair.  The two-arg form authoritatively sets the
 * register budget to floor(65536 / (NW*32 * LB_MIN_BLOCKS)), enabling LB
 * tuning on top of tc5jit's JIT M/K/N constants.
 */

#ifndef M_VAL
#  define M_VAL 4096
#  define K_VAL 4096
#  define N_VAL 4096
#endif

#ifndef LB_MIN_BLOCKS
#  define LB_MIN_BLOCKS 2
#endif

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

template <int BM, int BN, int BK, int NUM_WARPS>
__device__ __forceinline__ void matmul_tc5jit_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C
) {
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

    constexpr int A_SWZ   = BK / 8;
    constexpr int A_SHIFT = 64 / BK;

    constexpr int B_SWZ = BN / 8;

    constexpr int num_tiles = K_VAL / BK;   // compile-time loop trip count

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

#define ISSUE_TILE(k0_, buf_)                                                       \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                    \
            const int _g  = tid + _i * THREADS;                                    \
            const int _r  = (_g * A_ELEM) / BK;                                   \
            const int _c  = (_g * A_ELEM) % BK;                                   \
            const int _sc = ((_c / 8) ^ ((_r / A_SHIFT) % A_SWZ)) * 8 + (_c % 8); \
            __pipeline_memcpy_async(&A_shared[(buf_)][_r][_sc],                     \
                                    &A[(block_row + _r) * K_VAL + (k0_) + _c],    \
                                    A_ELEM * (int)sizeof(__nv_bfloat16));           \
        }                                                                           \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                    \
            const int _g  = tid + _i * THREADS;                                    \
            const int _r  = (_g * B_ELEM) / BN;                                   \
            const int _c  = (_g * B_ELEM) % BN;                                   \
            const int _sc = ((_c / 8) ^ (_r % B_SWZ)) * 8 + (_c % 8);            \
            __pipeline_memcpy_async(&B_shared[(buf_)][_r][_sc],                    \
                                    &B[((k0_) + _r) * N_VAL + block_col + _c],    \
                                    B_ELEM * (int)sizeof(__nv_bfloat16));           \
        }                                                                           \
        __pipeline_commit();                                                         \
    } while (0)

#define COMPUTE_TILE(buf_)                                                          \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                 \
            uint32_t _fa[WM_TILES][4];                                              \
            _Pragma("unroll")                                                       \
            for (int _mt = 0; _mt < WM_TILES; _mt++) {                             \
                const int _ar  = warp_row * WARP_TILE_M + _mt * 16 + (lane % 16); \
                const int _lg  = _kk * 2 + (lane / 16);                            \
                const int _phys = _lg ^ ((_ar / A_SHIFT) % A_SWZ);                \
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

    // Vectorized write-back: M and N are compile-time multiples of BM/BN,
    // so all tile writes are in-bounds — no guards needed.
    const int base_row = lane / 4;
    const int base_col = (lane % 4) * 2;

    #pragma unroll
    for (int mt = 0; mt < WM_TILES; mt++) {
        #pragma unroll
        for (int nt = 0; nt < WN_TILES; nt++) {
            const int gc  = block_col + warp_col * WARP_TILE_N + nt * 8 + base_col;
            const int gr0 = block_row + warp_row * WARP_TILE_M + mt * 16 + base_row;
            *reinterpret_cast<__nv_bfloat162*>(&C[gr0 * N_VAL + gc]) =
                __floats2bfloat162_rn(acc[mt][nt][0], acc[mt][nt][1]);
            const int gr8 = gr0 + 8;
            *reinterpret_cast<__nv_bfloat162*>(&C[gr8 * N_VAL + gc]) =
                __floats2bfloat162_rn(acc[mt][nt][2], acc[mt][nt][3]);
        }
    }
}

#define MAKE_LAUNCHER(BM_, BN_, BK_, NW_)                                           \
extern "C" __global__ __launch_bounds__(NW_ * 32, LB_MIN_BLOCKS)                    \
void matmul_cuda_tc5jit_bm##BM_##_bn##BN_##_bk##BK_##_nw##NW_(                      \
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,       \
    __nv_bfloat16* __restrict__ C) {                                                 \
    matmul_tc5jit_impl<BM_, BN_, BK_, NW_>(A, B, C);                               \
}

// ── NW=4 ─────────────────────────────────────────────────────────────────────
MAKE_LAUNCHER( 64,  64, 16, 4) MAKE_LAUNCHER( 64,  64, 32, 4) MAKE_LAUNCHER( 64,  64, 64, 4)
MAKE_LAUNCHER( 64, 128, 16, 4) MAKE_LAUNCHER( 64, 128, 32, 4) MAKE_LAUNCHER( 64, 128, 64, 4)
MAKE_LAUNCHER( 64, 256, 16, 4) MAKE_LAUNCHER( 64, 256, 32, 4) MAKE_LAUNCHER( 64, 256, 64, 4)
MAKE_LAUNCHER(128,  64, 16, 4) MAKE_LAUNCHER(128,  64, 32, 4) MAKE_LAUNCHER(128,  64, 64, 4)
MAKE_LAUNCHER(128, 128, 16, 4) MAKE_LAUNCHER(128, 128, 32, 4) MAKE_LAUNCHER(128, 128, 64, 4)
MAKE_LAUNCHER(256,  64, 16, 4) MAKE_LAUNCHER(256,  64, 32, 4) MAKE_LAUNCHER(256,  64, 64, 4)

// ── NW=8 ─────────────────────────────────────────────────────────────────────
MAKE_LAUNCHER( 64,  64, 16, 8) MAKE_LAUNCHER( 64,  64, 32, 8) MAKE_LAUNCHER( 64,  64, 64, 8)
MAKE_LAUNCHER( 64, 128, 16, 8) MAKE_LAUNCHER( 64, 128, 32, 8) MAKE_LAUNCHER( 64, 128, 64, 8)
MAKE_LAUNCHER( 64, 256, 16, 8) MAKE_LAUNCHER( 64, 256, 32, 8) MAKE_LAUNCHER( 64, 256, 64, 8)
MAKE_LAUNCHER(128,  64, 16, 8) MAKE_LAUNCHER(128,  64, 32, 8) MAKE_LAUNCHER(128,  64, 64, 8)
MAKE_LAUNCHER(128, 128, 16, 8) MAKE_LAUNCHER(128, 128, 32, 8) MAKE_LAUNCHER(128, 128, 64, 8)
MAKE_LAUNCHER(128, 256, 16, 8) MAKE_LAUNCHER(128, 256, 32, 8) MAKE_LAUNCHER(128, 256, 64, 8)
MAKE_LAUNCHER(256,  64, 16, 8) MAKE_LAUNCHER(256,  64, 32, 8) MAKE_LAUNCHER(256,  64, 64, 8)
MAKE_LAUNCHER(256, 128, 16, 8) MAKE_LAUNCHER(256, 128, 32, 8) MAKE_LAUNCHER(256, 128, 64, 8)

#undef MAKE_LAUNCHER
