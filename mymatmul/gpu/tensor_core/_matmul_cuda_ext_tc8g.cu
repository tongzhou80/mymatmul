/*
 * TC8g: Triton-style multi-stage BF16 tensor-core matmul.
 *
 * Template parameters: BM, BN, BK, NUM_WARPS, STAGES.
 *   STAGES controls ring-buffer depth: smem = STAGES*(BM*BK+BK*BN)*2 bytes.
 *   STAGES=2 is equivalent to the TC5 double-buffer pipeline.
 *   STAGES=5 matches Triton's autotuned config for large N.
 *
 * Pipeline: issue-before-compute.
 *   Prologue: issue STAGES-1 tiles.
 *   Main loop: wait_group(STAGES-2) → sync → issue_next → commit → compute → sync.
 *   Epilogue: wait_group(0) → sync → compute remaining STAGES-1 tiles.
 *
 * Grid: 1-D flat with GROUP_M=8 CTA swizzle (matches Triton's block ordering).
 * Loads: cp.async.cg.L2::128B (bypass L1).
 * Write-back: vectorized __nv_bfloat162 (2 stores per mma output pair).
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ── PTX helpers ──────────────────────────────────────────────────────────────

__device__ __forceinline__ void cp_async_cg(uint32_t smem_addr, const void* gmem_addr) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_addr));
}

__device__ __forceinline__ void cp_async_cg4(uint32_t smem_addr, const void* gmem_addr) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8;\n"
        :: "r"(smem_addr), "l"(gmem_addr));
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_ptr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& r0, uint32_t& r1, uint32_t smem_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1) : "r"(smem_ptr));
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

// ── Kernel ────────────────────────────────────────────────────────────────────

template <int BM, int BN, int BK, int NUM_WARPS, int STAGES>
__device__ __forceinline__ void matmul_tc8g_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N)
{
    constexpr int WARP_N      = 2;
    constexpr int WARP_M      = NUM_WARPS / WARP_N;
    constexpr int WARP_TILE_M = BM / WARP_M;
    constexpr int WARP_TILE_N = BN / WARP_N;
    constexpr int WM_TILES    = WARP_TILE_M / 16;
    constexpr int WN_TILES    = WARP_TILE_N / 8;
    constexpr int THREADS      = NUM_WARPS * 32;

    constexpr int A_ELEM   = (BM * BK / THREADS >= 8) ? 8 : 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = (BK * BN / THREADS >= 8) ? 8 : 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    constexpr int A_SWZ   = BK / 8;
    constexpr int A_SHIFT = 64 / BK;
    constexpr int B_SWZ   = BN / 8;

    // smem layout: A_shared[STAGES][BM][BK], B_shared[STAGES][BK][BN]
    extern __shared__ __nv_bfloat16 smem[];
    auto A_shared = reinterpret_cast<__nv_bfloat16 (*)[BM][BK]>(smem);
    auto B_shared = reinterpret_cast<__nv_bfloat16 (*)[BK][BN]>(smem + STAGES * BM * BK);

    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid % 32;
    const int warp_row = warp_id / WARP_N;
    const int warp_col = warp_id % WARP_N;

    // GROUP_M=8 CTA swizzle: Triton-style 1-D block ordering for L2 B-tile reuse
    constexpr int GROUP_M = 8;
    const int pid         = blockIdx.x;
    const int grid_m      = (M + BM - 1) / BM;
    const int grid_n      = (N + BN - 1) / BN;
    const int group_id    = pid / (GROUP_M * grid_n);
    const int first_pid_m = group_id * GROUP_M;
    const int group_size_m = (first_pid_m + GROUP_M <= grid_m) ? GROUP_M : (grid_m - first_pid_m);
    const int pid_m       = first_pid_m + (pid % group_size_m);
    const int pid_n       = (pid % (GROUP_M * grid_n)) / group_size_m;
    const int block_row   = pid_m * BM;
    const int block_col   = pid_n * BN;

    float acc[WM_TILES][WN_TILES][4] = {};

    // Macro: issue one A+B tile (k-offset k0_) into smem slot buf_.
    // Uses cp.async.cg for 16-byte copies (A_ELEM/B_ELEM=8) or .ca for 8-byte.
#define ISSUE_TILE(k0_, buf_)                                                         \
    do {                                                                               \
        _Pragma("unroll")                                                              \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                       \
            const int _g  = tid + _i * THREADS;                                       \
            const int _r  = (_g * A_ELEM) / BK;                                       \
            const int _c  = (_g * A_ELEM) % BK;                                       \
            const int _sc = ((_c / 8) ^ ((_r / A_SHIFT) % A_SWZ)) * 8 + (_c % 8);   \
            uint32_t _da  = __cvta_generic_to_shared(&A_shared[(buf_)][_r][_sc]);     \
            if constexpr (A_ELEM == 8)                                                 \
                cp_async_cg(_da, &A[(block_row + _r) * K + (k0_) + _c]);             \
            else                                                                        \
                cp_async_cg4(_da, &A[(block_row + _r) * K + (k0_) + _c]);            \
        }                                                                              \
        _Pragma("unroll")                                                              \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                       \
            const int _g  = tid + _i * THREADS;                                       \
            const int _r  = (_g * B_ELEM) / BN;                                       \
            const int _c  = (_g * B_ELEM) % BN;                                       \
            const int _sc = ((_c / 8) ^ (_r % B_SWZ)) * 8 + (_c % 8);               \
            uint32_t _db  = __cvta_generic_to_shared(&B_shared[(buf_)][_r][_sc]);     \
            if constexpr (B_ELEM == 8)                                                 \
                cp_async_cg(_db, &B[((k0_) + _r) * N + block_col + _c]);             \
            else                                                                        \
                cp_async_cg4(_db, &B[((k0_) + _r) * N + block_col + _c]);            \
        }                                                                              \
    } while (0)

    // Macro: compute one tile from smem slot buf_.
#define COMPUTE_TILE(buf_)                                                            \
    do {                                                                               \
        _Pragma("unroll")                                                              \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                    \
            uint32_t _fa[WM_TILES][4];                                                 \
            _Pragma("unroll")                                                          \
            for (int _mt = 0; _mt < WM_TILES; _mt++) {                                \
                const int _ar   = warp_row * WARP_TILE_M + _mt * 16 + (lane % 16);   \
                const int _lg   = _kk * 2 + (lane / 16);                              \
                const int _phys = _lg ^ ((_ar / A_SHIFT) % A_SWZ);                   \
                ldmatrix_x4(_fa[_mt][0],_fa[_mt][1],_fa[_mt][2],_fa[_mt][3],         \
                    __cvta_generic_to_shared(&A_shared[(buf_)][_ar][_phys * 8]));     \
            }                                                                          \
            uint32_t _fb[WN_TILES][2];                                                 \
            _Pragma("unroll")                                                          \
            for (int _nt = 0; _nt < WN_TILES; _nt++) {                                \
                const int _br = _kk * 16 + (lane % 16);                               \
                const int _nc = warp_col * WARP_TILE_N + _nt * 8;                     \
                const int _sc = ((_nc / 8) ^ (_br % B_SWZ)) * 8;                     \
                ldmatrix_x2_trans(_fb[_nt][0],_fb[_nt][1],                            \
                    __cvta_generic_to_shared(&B_shared[(buf_)][_br][_sc]));           \
            }                                                                          \
            _Pragma("unroll")                                                          \
            for (int _mt = 0; _mt < WM_TILES; _mt++) {                                \
                _Pragma("unroll")                                                      \
                for (int _nt = 0; _nt < WN_TILES; _nt++) {                            \
                    mma_m16n8k16(acc[_mt][_nt][0],acc[_mt][_nt][1],                   \
                                 acc[_mt][_nt][2],acc[_mt][_nt][3],                   \
                                 _fa[_mt][0],_fa[_mt][1],_fa[_mt][2],_fa[_mt][3],    \
                                 _fb[_nt][0],_fb[_nt][1]);                            \
                }                                                                      \
            }                                                                          \
        }                                                                              \
    } while (0)

    const int num_tiles = K / BK;

    // Prologue: issue STAGES-1 tiles
    #pragma unroll
    for (int i = 0; i < STAGES - 1; i++) {
        ISSUE_TILE(i * BK, i);
        asm volatile("cp.async.commit_group;\n");
    }

    // Main loop: wait → sync → issue_next → commit → compute → sync
    for (int k_iter = 0; k_iter < num_tiles - (STAGES - 1); k_iter++) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
        __syncthreads();

        const int compute_idx = k_iter % STAGES;
        const int fetch_idx   = (k_iter + STAGES - 1) % STAGES;

        ISSUE_TILE((k_iter + STAGES - 1) * BK, fetch_idx);
        asm volatile("cp.async.commit_group;\n");

        COMPUTE_TILE(compute_idx);
        __syncthreads();
    }

    // Epilogue: wait for all remaining groups, then compute the last STAGES-1 tiles
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    const int d = num_tiles - (STAGES - 1);
    #pragma unroll
    for (int i = 0; i < STAGES - 1; i++) {
        COMPUTE_TILE((d + i) % STAGES);
    }

#undef ISSUE_TILE
#undef COMPUTE_TILE

    // Vectorized BF16 write-back: pack consecutive column pairs into bfloat162
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

// ── Launchers ─────────────────────────────────────────────────────────────────

#define MAKE_LAUNCHER(BM_, BN_, BK_, NW_, NS_)                                       \
extern "C" __global__ __launch_bounds__(NW_ * 32)                                    \
void matmul_cuda_tc8g_bm##BM_##_bn##BN_##_bk##BK_##_nw##NW_##_ns##NS_(             \
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,        \
    __nv_bfloat16* __restrict__ C, int M, int K, int N) {                            \
    matmul_tc8g_impl<BM_, BN_, BK_, NW_, NS_>(A, B, C, M, K, N);                   \
}

// NW=4
MAKE_LAUNCHER( 64,  64, 16, 4, 2) MAKE_LAUNCHER( 64,  64, 16, 4, 3) MAKE_LAUNCHER( 64,  64, 16, 4, 4) MAKE_LAUNCHER( 64,  64, 16, 4, 5)
MAKE_LAUNCHER( 64,  64, 32, 4, 2) MAKE_LAUNCHER( 64,  64, 32, 4, 3) MAKE_LAUNCHER( 64,  64, 32, 4, 4) MAKE_LAUNCHER( 64,  64, 32, 4, 5)
MAKE_LAUNCHER( 64,  64, 64, 4, 2) MAKE_LAUNCHER( 64,  64, 64, 4, 3) MAKE_LAUNCHER( 64,  64, 64, 4, 4) MAKE_LAUNCHER( 64,  64, 64, 4, 5)
MAKE_LAUNCHER( 64, 128, 16, 4, 2) MAKE_LAUNCHER( 64, 128, 16, 4, 3) MAKE_LAUNCHER( 64, 128, 16, 4, 4) MAKE_LAUNCHER( 64, 128, 16, 4, 5)
MAKE_LAUNCHER( 64, 128, 32, 4, 2) MAKE_LAUNCHER( 64, 128, 32, 4, 3) MAKE_LAUNCHER( 64, 128, 32, 4, 4) MAKE_LAUNCHER( 64, 128, 32, 4, 5)
MAKE_LAUNCHER( 64, 128, 64, 4, 2) MAKE_LAUNCHER( 64, 128, 64, 4, 3) MAKE_LAUNCHER( 64, 128, 64, 4, 4) MAKE_LAUNCHER( 64, 128, 64, 4, 5)
MAKE_LAUNCHER( 64, 256, 16, 4, 2) MAKE_LAUNCHER( 64, 256, 16, 4, 3) MAKE_LAUNCHER( 64, 256, 16, 4, 4) MAKE_LAUNCHER( 64, 256, 16, 4, 5)
MAKE_LAUNCHER( 64, 256, 32, 4, 2) MAKE_LAUNCHER( 64, 256, 32, 4, 3) MAKE_LAUNCHER( 64, 256, 32, 4, 4) MAKE_LAUNCHER( 64, 256, 32, 4, 5)
MAKE_LAUNCHER( 64, 256, 64, 4, 2) MAKE_LAUNCHER( 64, 256, 64, 4, 3) MAKE_LAUNCHER( 64, 256, 64, 4, 4)
MAKE_LAUNCHER(128,  64, 16, 4, 2) MAKE_LAUNCHER(128,  64, 16, 4, 3) MAKE_LAUNCHER(128,  64, 16, 4, 4) MAKE_LAUNCHER(128,  64, 16, 4, 5)
MAKE_LAUNCHER(128,  64, 32, 4, 2) MAKE_LAUNCHER(128,  64, 32, 4, 3) MAKE_LAUNCHER(128,  64, 32, 4, 4) MAKE_LAUNCHER(128,  64, 32, 4, 5)
MAKE_LAUNCHER(128,  64, 64, 4, 2) MAKE_LAUNCHER(128,  64, 64, 4, 3) MAKE_LAUNCHER(128,  64, 64, 4, 4) MAKE_LAUNCHER(128,  64, 64, 4, 5)
MAKE_LAUNCHER(128, 128, 16, 4, 2) MAKE_LAUNCHER(128, 128, 16, 4, 3) MAKE_LAUNCHER(128, 128, 16, 4, 4) MAKE_LAUNCHER(128, 128, 16, 4, 5)
MAKE_LAUNCHER(128, 128, 32, 4, 2) MAKE_LAUNCHER(128, 128, 32, 4, 3) MAKE_LAUNCHER(128, 128, 32, 4, 4) MAKE_LAUNCHER(128, 128, 32, 4, 5)
MAKE_LAUNCHER(128, 128, 64, 4, 2) MAKE_LAUNCHER(128, 128, 64, 4, 3) MAKE_LAUNCHER(128, 128, 64, 4, 4)
MAKE_LAUNCHER(128, 256, 16, 4, 2) MAKE_LAUNCHER(128, 256, 16, 4, 3) MAKE_LAUNCHER(128, 256, 16, 4, 4) MAKE_LAUNCHER(128, 256, 16, 4, 5)
MAKE_LAUNCHER(128, 256, 32, 4, 2) MAKE_LAUNCHER(128, 256, 32, 4, 3) MAKE_LAUNCHER(128, 256, 32, 4, 4) MAKE_LAUNCHER(128, 256, 32, 4, 5)
MAKE_LAUNCHER(128, 256, 64, 4, 2) MAKE_LAUNCHER(128, 256, 64, 4, 3)
MAKE_LAUNCHER(256,  64, 16, 4, 2) MAKE_LAUNCHER(256,  64, 16, 4, 3) MAKE_LAUNCHER(256,  64, 16, 4, 4) MAKE_LAUNCHER(256,  64, 16, 4, 5)
MAKE_LAUNCHER(256,  64, 32, 4, 2) MAKE_LAUNCHER(256,  64, 32, 4, 3) MAKE_LAUNCHER(256,  64, 32, 4, 4) MAKE_LAUNCHER(256,  64, 32, 4, 5)
MAKE_LAUNCHER(256,  64, 64, 4, 2) MAKE_LAUNCHER(256,  64, 64, 4, 3) MAKE_LAUNCHER(256,  64, 64, 4, 4)
MAKE_LAUNCHER(256, 128, 16, 4, 2) MAKE_LAUNCHER(256, 128, 16, 4, 3) MAKE_LAUNCHER(256, 128, 16, 4, 4) MAKE_LAUNCHER(256, 128, 16, 4, 5)
MAKE_LAUNCHER(256, 128, 32, 4, 2) MAKE_LAUNCHER(256, 128, 32, 4, 3) MAKE_LAUNCHER(256, 128, 32, 4, 4)
MAKE_LAUNCHER(256, 128, 64, 4, 2)

// NW=8
MAKE_LAUNCHER( 64,  64, 16, 8, 2) MAKE_LAUNCHER( 64,  64, 16, 8, 3) MAKE_LAUNCHER( 64,  64, 16, 8, 4) MAKE_LAUNCHER( 64,  64, 16, 8, 5)
MAKE_LAUNCHER( 64,  64, 32, 8, 2) MAKE_LAUNCHER( 64,  64, 32, 8, 3) MAKE_LAUNCHER( 64,  64, 32, 8, 4) MAKE_LAUNCHER( 64,  64, 32, 8, 5)
MAKE_LAUNCHER( 64,  64, 64, 8, 2) MAKE_LAUNCHER( 64,  64, 64, 8, 3) MAKE_LAUNCHER( 64,  64, 64, 8, 4) MAKE_LAUNCHER( 64,  64, 64, 8, 5)
MAKE_LAUNCHER( 64, 128, 16, 8, 2) MAKE_LAUNCHER( 64, 128, 16, 8, 3) MAKE_LAUNCHER( 64, 128, 16, 8, 4) MAKE_LAUNCHER( 64, 128, 16, 8, 5)
MAKE_LAUNCHER( 64, 128, 32, 8, 2) MAKE_LAUNCHER( 64, 128, 32, 8, 3) MAKE_LAUNCHER( 64, 128, 32, 8, 4) MAKE_LAUNCHER( 64, 128, 32, 8, 5)
MAKE_LAUNCHER( 64, 128, 64, 8, 2) MAKE_LAUNCHER( 64, 128, 64, 8, 3) MAKE_LAUNCHER( 64, 128, 64, 8, 4) MAKE_LAUNCHER( 64, 128, 64, 8, 5)
MAKE_LAUNCHER( 64, 256, 16, 8, 2) MAKE_LAUNCHER( 64, 256, 16, 8, 3) MAKE_LAUNCHER( 64, 256, 16, 8, 4) MAKE_LAUNCHER( 64, 256, 16, 8, 5)
MAKE_LAUNCHER( 64, 256, 32, 8, 2) MAKE_LAUNCHER( 64, 256, 32, 8, 3) MAKE_LAUNCHER( 64, 256, 32, 8, 4) MAKE_LAUNCHER( 64, 256, 32, 8, 5)
MAKE_LAUNCHER( 64, 256, 64, 8, 2) MAKE_LAUNCHER( 64, 256, 64, 8, 3)
MAKE_LAUNCHER(128,  64, 16, 8, 2) MAKE_LAUNCHER(128,  64, 16, 8, 3) MAKE_LAUNCHER(128,  64, 16, 8, 4) MAKE_LAUNCHER(128,  64, 16, 8, 5)
MAKE_LAUNCHER(128,  64, 32, 8, 2) MAKE_LAUNCHER(128,  64, 32, 8, 3) MAKE_LAUNCHER(128,  64, 32, 8, 4) MAKE_LAUNCHER(128,  64, 32, 8, 5)
MAKE_LAUNCHER(128,  64, 64, 8, 2) MAKE_LAUNCHER(128,  64, 64, 8, 3) MAKE_LAUNCHER(128,  64, 64, 8, 4) MAKE_LAUNCHER(128,  64, 64, 8, 5)
MAKE_LAUNCHER(128, 128, 16, 8, 2) MAKE_LAUNCHER(128, 128, 16, 8, 3) MAKE_LAUNCHER(128, 128, 16, 8, 4) MAKE_LAUNCHER(128, 128, 16, 8, 5)
MAKE_LAUNCHER(128, 128, 32, 8, 2) MAKE_LAUNCHER(128, 128, 32, 8, 3) MAKE_LAUNCHER(128, 128, 32, 8, 4) MAKE_LAUNCHER(128, 128, 32, 8, 5)
MAKE_LAUNCHER(128, 128, 64, 8, 2) MAKE_LAUNCHER(128, 128, 64, 8, 3) MAKE_LAUNCHER(128, 128, 64, 8, 4)
MAKE_LAUNCHER(128, 256, 16, 8, 2) MAKE_LAUNCHER(128, 256, 16, 8, 3) MAKE_LAUNCHER(128, 256, 16, 8, 4) MAKE_LAUNCHER(128, 256, 16, 8, 5)
MAKE_LAUNCHER(128, 256, 32, 8, 2) MAKE_LAUNCHER(128, 256, 32, 8, 3) MAKE_LAUNCHER(128, 256, 32, 8, 4)
MAKE_LAUNCHER(128, 256, 64, 8, 2)
MAKE_LAUNCHER(256,  64, 16, 8, 2) MAKE_LAUNCHER(256,  64, 16, 8, 3) MAKE_LAUNCHER(256,  64, 16, 8, 4) MAKE_LAUNCHER(256,  64, 16, 8, 5)
MAKE_LAUNCHER(256,  64, 32, 8, 2) MAKE_LAUNCHER(256,  64, 32, 8, 3) MAKE_LAUNCHER(256,  64, 32, 8, 4) MAKE_LAUNCHER(256,  64, 32, 8, 5)
MAKE_LAUNCHER(256,  64, 64, 8, 2) MAKE_LAUNCHER(256,  64, 64, 8, 3) MAKE_LAUNCHER(256,  64, 64, 8, 4)
MAKE_LAUNCHER(256, 128, 16, 8, 2) MAKE_LAUNCHER(256, 128, 16, 8, 3) MAKE_LAUNCHER(256, 128, 16, 8, 4) MAKE_LAUNCHER(256, 128, 16, 8, 5)
MAKE_LAUNCHER(256, 128, 32, 8, 2) MAKE_LAUNCHER(256, 128, 32, 8, 3) MAKE_LAUNCHER(256, 128, 32, 8, 4)
MAKE_LAUNCHER(256, 128, 64, 8, 2)

#undef MAKE_LAUNCHER
