#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>

// ── PTX helpers ──────────────────────────────────────────────────────────────

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& r0, uint32_t& r1, uint32_t smem_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

// Emulate Triton's cp.async.cg.shared.global (Cache-Global, 16-byte bypasses L1)
__device__ __forceinline__ void cp_async_cg_16(uint32_t smem_addr, const void* global_ptr) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(global_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N > 0 ? N : 0));
}

// ── Kernel implementation ─────────────────────────────────────────────────────

template <int BM, int BN, int BK, int NUM_WARPS, int STAGES>
__device__ __forceinline__ void matmul_tc5_triton_emulated(
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

    constexpr int A_ELEM   = 8; // Triton forces 16-byte (8 bf16) vectorization here
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = 8; 
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    constexpr int A_SWZ   = BK / 8;
    constexpr int A_SHIFT = 64 / BK;
    constexpr int B_SWZ   = BN / 8;

    // 5-Stage Circular Shared Memory
    extern __shared__ __nv_bfloat16 smem[];
    auto A_shared = reinterpret_cast<__nv_bfloat16 (*)[STAGES][BM][BK]>(smem);
    auto B_shared = reinterpret_cast<__nv_bfloat16 (*)[STAGES][BK][BN]>(smem + STAGES * BM * BK);

    const int tid      = threadIdx.x; // We use a 1D block now
    const int warp_id  = tid / 32;
    const int lane     = tid % 32;
    const int warp_row = warp_id / WARP_N;
    const int warp_col = warp_id % WARP_N;

    // ──────────────────────────────────────────────────────────────────────────
    // Triton Grid Swizzling (GROUP_M = 8)
    // ──────────────────────────────────────────────────────────────────────────
    const int pid = blockIdx.x; // Launch grid as 1D
    const int GROUP_M = 8;
    
    const int grid_m = (M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;
    const int num_pid_in_group = GROUP_M * grid_n;
    
    const int group_id = pid / num_pid_in_group;
    const int first_pid_m = group_id * GROUP_M;
    const int group_size_m = min(grid_m - first_pid_m, GROUP_M);
    
    const int pid_m = first_pid_m + (pid % group_size_m);
    const int pid_n = (pid % num_pid_in_group) / group_size_m;

    const int block_row = pid_m * BM;
    const int block_col = pid_n * BN;
    // ──────────────────────────────────────────────────────────────────────────

    float acc[WM_TILES][WN_TILES][4] = {};

#define ISSUE_TILE(k0_, buf_)                                                       \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                    \
            const int _g  = tid + _i * THREADS;                                    \
            const int _r  = (_g * A_ELEM) / BK;                                   \
            const int _c  = (_g * A_ELEM) % BK;                                   \
            const int _sc = ((_c / 8) ^ ((_r / A_SHIFT) % A_SWZ)) * 8 + (_c % 8); \
            uint32_t smem_addr = __cvta_generic_to_shared(&A_shared[0][buf_][_r][_sc]); \
            cp_async_cg_16(smem_addr, &A[(block_row + _r) * K + (k0_) + _c]);       \
        }                                                                           \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                    \
            const int _g  = tid + _i * THREADS;                                    \
            const int _r  = (_g * B_ELEM) / BN;                                   \
            const int _c  = (_g * B_ELEM) % BN;                                   \
            const int _sc = ((_c / 8) ^ (_r % B_SWZ)) * 8 + (_c % 8);            \
            uint32_t smem_addr = __cvta_generic_to_shared(&B_shared[0][buf_][_r][_sc]); \
            cp_async_cg_16(smem_addr, &B[((k0_) + _r) * N + block_col + _c]);       \
        }                                                                           \
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
                    __cvta_generic_to_shared(&A_shared[0][buf_][_ar][_phys * 8]));  \
            }                                                                       \
            uint32_t _fb[WN_TILES][2];                                              \
            _Pragma("unroll")                                                       \
            for (int _nt = 0; _nt < WN_TILES; _nt++) {                             \
                const int _br = _kk * 16 + (lane % 16);                            \
                const int _nc = warp_col * WARP_TILE_N + _nt * 8;                  \
                const int _sc = ((_nc / 8) ^ (_br % B_SWZ)) * 8;                  \
                ldmatrix_x2_trans(_fb[_nt][0], _fb[_nt][1],                        \
                    __cvta_generic_to_shared(&B_shared[0][buf_][_br][_sc]));         \
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

    // Prologue: fill STAGES-1 buffers
    #pragma unroll
    for (int i = 0; i < STAGES - 1; i++) {
        ISSUE_TILE(i * BK, i);
        cp_async_commit();
    }

    // Main loop: wait for the oldest tile, compute it, and issue the next one
    for (int k_iter = 0; k_iter < num_tiles - (STAGES - 1); k_iter++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();

        int compute_idx = k_iter % STAGES;
        int fetch_idx = (k_iter + STAGES - 1) % STAGES;

        ISSUE_TILE((k_iter + STAGES - 1) * BK, fetch_idx);
        cp_async_commit();

        COMPUTE_TILE(compute_idx);
        __syncthreads();
    }

    // Epilogue: drain remaining STAGES-1=4 tiles.
    // No post-COMPUTE_TILE sync needed — no writes follow, different slots.
    {
        const int d = num_tiles - (STAGES - 1);
        asm volatile("cp.async.wait_group 2;\n" ::: "memory");
        __syncthreads(); COMPUTE_TILE((d + 0) % STAGES);
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        __syncthreads(); COMPUTE_TILE((d + 1) % STAGES);
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads(); COMPUTE_TILE((d + 2) % STAGES);
        COMPUTE_TILE((d + 3) % STAGES);
    }

#undef ISSUE_TILE
#undef COMPUTE_TILE

    // Vectorized write-back (Identical to your implementation)
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


// Fixed config: BM=64, BN=128, BK=16, NW=4, STAGES=5
// Smem: 5 × (BM×BK + BK×BN) × 2 = 5 × 3072 × 2 = 30720 bytes
// Grid: 1-D flat (grid_size = (M/BM) * (N/BN)), GROUP_M=8 swizzle inside kernel.

extern "C" __global__ __launch_bounds__(128)
void matmul_cuda_tc8_gemini(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N)
{
    matmul_tc5_triton_emulated<64, 128, 16, 4, 5>(A, B, C, M, K, N);
}