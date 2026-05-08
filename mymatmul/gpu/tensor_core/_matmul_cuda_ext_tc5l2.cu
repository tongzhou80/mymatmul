#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>

/*
 * TC5l2: TC5 with cp.async.cg.L2::128B prefetch hint on tile loads.
 *
 * Identical to TC5 except ISSUE_TILE uses cp.async.cg + L2::128B eviction
 * hint (bypass L1, prefetch/retain the full 128-byte cache line in L2).
 * This helps when adjacent CTAs touch overlapping cache lines — the line
 * is already warm in L2 when the next CTA's DMA arrives.
 *
 *   .cg       = cache at global (L2) level, bypass L1
 *   L2::128B  = eviction hint: prefetch full 128-byte line into L2
 *
 * L2::128B requires 16-byte copies; for configs where A_ELEM or B_ELEM is
 * 4 (8 bytes) we fall back to cp.async.ca without the hint.
 *
 * Pipeline management switches from __pipeline_* to the PTX
 * cp.async.commit_group / cp.async.wait_group builtins that underlie them.
 *
 * Everything else (smem layout, XOR swizzle, COMPUTE_TILE, vectorized
 * bfloat162 write-back) is identical to TC5.
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

// 16-byte async copy with L2::128B hint (bypass L1, prefetch full cache line).
__device__ __forceinline__ void cp_async_cg_l2(void* dst, const void* src) {
    uint32_t smem_ptr = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(smem_ptr), "l"((uint64_t)src)
    );
}

// 8-byte async copy (L2::128B requires 16-byte; fall back to .ca for 8-byte).
__device__ __forceinline__ void cp_async_ca_8(void* dst, const void* src) {
    uint32_t smem_ptr = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8;\n"
        :: "r"(smem_ptr), "l"((uint64_t)src)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" :::);
}

__device__ __forceinline__ void cp_async_wait1() {
    asm volatile("cp.async.wait_group 1;\n" :::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" :::);
}

// ── Kernel implementation ─────────────────────────────────────────────────────

template <int BM, int BN, int BK, int NUM_WARPS>
__device__ __forceinline__ void matmul_tc5l2_impl(
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
    constexpr int WN_TILES = WARP_TILE_N / 8;

    constexpr int THREADS = NUM_WARPS * 32;

    constexpr int A_ELEM   = (BM * BK / THREADS >= 8) ? 8 : 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = (BK * BN / THREADS >= 8) ? 8 : 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    constexpr int A_SWZ   = BK / 8;
    constexpr int A_SHIFT = 64 / BK;
    constexpr int B_SWZ   = BN / 8;

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
            if constexpr (A_ELEM == 8) {                                            \
                cp_async_cg_l2(&A_shared[(buf_)][_r][_sc],                          \
                               &A[(block_row + _r) * K + (k0_) + _c]);             \
            } else {                                                                \
                cp_async_ca_8(&A_shared[(buf_)][_r][_sc],                           \
                              &A[(block_row + _r) * K + (k0_) + _c]);              \
            }                                                                       \
        }                                                                           \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                    \
            const int _g  = tid + _i * THREADS;                                    \
            const int _r  = (_g * B_ELEM) / BN;                                   \
            const int _c  = (_g * B_ELEM) % BN;                                   \
            const int _sc = ((_c / 8) ^ (_r % B_SWZ)) * 8 + (_c % 8);            \
            if constexpr (B_ELEM == 8) {                                            \
                cp_async_cg_l2(&B_shared[(buf_)][_r][_sc],                          \
                               &B[((k0_) + _r) * N + block_col + _c]);             \
            } else {                                                                \
                cp_async_ca_8(&B_shared[(buf_)][_r][_sc],                           \
                              &B[((k0_) + _r) * N + block_col + _c]);              \
            }                                                                       \
        }                                                                           \
        cp_async_commit();                                                           \
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

    const int num_tiles = K / BK;

    ISSUE_TILE(0, 0);

    for (int k_iter = 0; k_iter < num_tiles - 1; k_iter++) {
        const int cur = k_iter & 1;
        const int nxt = 1 - cur;
        ISSUE_TILE((k_iter + 1) * BK, nxt);
        cp_async_wait1();
        __syncthreads();
        COMPUTE_TILE(cur);
        __syncthreads();
    }

    cp_async_wait_all();
    __syncthreads();
    COMPUTE_TILE((num_tiles - 1) & 1);

#undef ISSUE_TILE
#undef COMPUTE_TILE

    const int base_row = lane / 4;
    const int base_col = (lane % 4) * 2;

    #pragma unroll
    for (int mt = 0; mt < WM_TILES; mt++) {
        #pragma unroll
        for (int nt = 0; nt < WN_TILES; nt++) {
            const int gc  = block_col + warp_col * WARP_TILE_N + nt * 8 + base_col;
            const int gr0 = block_row + warp_row * WARP_TILE_M + mt * 16 + base_row;
            if (gr0 < M && gc < N)
                *reinterpret_cast<__nv_bfloat162*>(&C[gr0 * N + gc]) =
                    __floats2bfloat162_rn(acc[mt][nt][0], acc[mt][nt][1]);
            const int gr8 = gr0 + 8;
            if (gr8 < M && gc < N)
                *reinterpret_cast<__nv_bfloat162*>(&C[gr8 * N + gc]) =
                    __floats2bfloat162_rn(acc[mt][nt][2], acc[mt][nt][3]);
        }
    }
}

#define MAKE_LAUNCHER(BM_, BN_, BK_, NW_)                                           \
extern "C" __global__ __launch_bounds__(NW_ * 32)                                   \
void matmul_cuda_tc5l2_bm##BM_##_bn##BN_##_bk##BK_##_nw##NW_(                      \
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,       \
    __nv_bfloat16* __restrict__ C, int M, int K, int N) {                           \
    matmul_tc5l2_impl<BM_, BN_, BK_, NW_>(A, B, C, M, K, N);                       \
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
