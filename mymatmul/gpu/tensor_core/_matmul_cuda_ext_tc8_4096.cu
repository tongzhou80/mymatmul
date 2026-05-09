/*
 * TC8_4096: BF16 tensor-core matmul — Triton-style 5-stage pipeline.
 *
 * Fixed config: BM=64, BN=128, BK=16, NW=4.
 * Pipeline: NS=5 stages, NS_SLOTS=4 ring-buffer smem slots.
 *   Prologue issues NS_SLOTS=4 tiles.
 *   Main loop: wait_group(3) → sync → compute → issue-next → commit_group.
 *   One __syncthreads() per iteration (matches Triton PTX structure).
 *
 * Load path: cp.async.cg.L2::128B (bypass L1, hint 128-byte L2 lines).
 * Smem: 4 × (BM×BK + BK×BN) × 2 = 4 × 3072 × 2 = 24576 bytes.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ── fixed config ─────────────────────────────────────────────────────────────

static constexpr int BM       = 64;
static constexpr int BN       = 128;
static constexpr int BK       = 16;
static constexpr int NW       = 4;
static constexpr int NS_SLOTS = 4;      // ring-buffer slots = NS - 1

static constexpr int THREADS  = NW * 32;                // 128
static constexpr int WARP_N   = 2;
static constexpr int WARP_M   = NW / WARP_N;            // 2
static constexpr int WARP_TILE_M = BM / WARP_M;         // 32
static constexpr int WARP_TILE_N = BN / WARP_N;         // 64
static constexpr int WM_TILES = WARP_TILE_M / 16;       // 2
static constexpr int WN_TILES = WARP_TILE_N / 8;        // 8

static constexpr int A_ELEM   = 8;                      // 16 bytes / 2
static constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS; // 1
static constexpr int B_ELEM   = 8;
static constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS; // 2

static constexpr int A_SWZ    = BK / 8;                 // 2
static constexpr int A_SHIFT  = 64 / BK;                // 4
static constexpr int B_SWZ    = BN / 8;                 // 16

static constexpr int SLOT_A   = BM * BK;                // 1024 bf16 elements
static constexpr int SLOT_B   = BK * BN;                // 2048 bf16 elements
static constexpr int SLOT_SZ  = SLOT_A + SLOT_B;        // 3072 bf16 elements per slot

// ── PTX helpers ──────────────────────────────────────────────────────────────

__device__ __forceinline__ void cp_async_cg(uint32_t smem_addr, const void* gmem_addr) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_addr)
    );
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_ptr)
{
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_ptr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& r0, uint32_t& r1, uint32_t smem_ptr)
{
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1) : "r"(smem_ptr));
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

// ── Issue one A+B tile (k-offset k0_) into ring-buffer slot s_ ───────────────
// Ends with cp.async.commit_group, creating one async group.

#define ISSUE_TILE(k0_, s_)                                                         \
    do {                                                                             \
        __nv_bfloat16* _As = smem + (int)(s_) * SLOT_SZ;                           \
        __nv_bfloat16* _Bs = smem + (int)(s_) * SLOT_SZ + SLOT_A;                  \
        const __nv_bfloat16* _Ag = A + block_row * K + (k0_);                      \
        const __nv_bfloat16* _Bg = B + (int)(k0_) * N + block_col;                 \
        _Pragma("unroll")                                                            \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                    \
            const int _g  = tid + _i * THREADS;                                    \
            const int _r  = (_g * A_ELEM) / BK;                                    \
            const int _c  = (_g * A_ELEM) % BK;                                    \
            const int _sc = ((_c / 8) ^ ((_r / A_SHIFT) % A_SWZ)) * 8;            \
            cp_async_cg(__cvta_generic_to_shared(_As + _r * BK + _sc),             \
                        _Ag + _r * K + _c);                                         \
        }                                                                            \
        _Pragma("unroll")                                                            \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                    \
            const int _g  = tid + _i * THREADS;                                    \
            const int _r  = (_g * B_ELEM) / BN;                                    \
            const int _c  = (_g * B_ELEM) % BN;                                    \
            const int _sc = ((_c / 8) ^ (_r % B_SWZ)) * 8;                        \
            cp_async_cg(__cvta_generic_to_shared(_Bs + _r * BN + _sc),             \
                        _Bg + _r * N + _c);                                         \
        }                                                                            \
        asm volatile("cp.async.commit_group;\n");                                   \
    } while (0)

// ── Compute one tile from smem ring-buffer slot s_ ────────────────────────────

#define COMPUTE_TILE(s_)                                                            \
    do {                                                                             \
        const __nv_bfloat16* _As = smem + (int)(s_) * SLOT_SZ;                    \
        const __nv_bfloat16* _Bs = smem + (int)(s_) * SLOT_SZ + SLOT_A;           \
        _Pragma("unroll")                                                            \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                 \
            uint32_t _fa[WM_TILES][4];                                              \
            _Pragma("unroll")                                                        \
            for (int _mt = 0; _mt < WM_TILES; _mt++) {                             \
                const int _ar   = warp_row * WARP_TILE_M + _mt * 16 + (lane % 16); \
                const int _lg   = _kk * 2 + (lane / 16);                           \
                const int _phys = _lg ^ ((_ar / A_SHIFT) % A_SWZ);                \
                ldmatrix_x4(_fa[_mt][0], _fa[_mt][1], _fa[_mt][2], _fa[_mt][3],   \
                    __cvta_generic_to_shared(_As + _ar * BK + _phys * 8));         \
            }                                                                        \
            uint32_t _fb[WN_TILES][2];                                              \
            _Pragma("unroll")                                                        \
            for (int _nt = 0; _nt < WN_TILES; _nt++) {                             \
                const int _br = _kk * 16 + (lane % 16);                            \
                const int _nc = warp_col * WARP_TILE_N + _nt * 8;                  \
                const int _sc = ((_nc / 8) ^ (_br % B_SWZ)) * 8;                  \
                ldmatrix_x2_trans(_fb[_nt][0], _fb[_nt][1],                        \
                    __cvta_generic_to_shared(_Bs + _br * BN + _sc));               \
            }                                                                        \
            _Pragma("unroll")                                                        \
            for (int _mt = 0; _mt < WM_TILES; _mt++) {                             \
                _Pragma("unroll")                                                    \
                for (int _nt = 0; _nt < WN_TILES; _nt++) {                         \
                    mma_m16n8k16(acc[_mt][_nt][0], acc[_mt][_nt][1],               \
                                 acc[_mt][_nt][2], acc[_mt][_nt][3],               \
                                 _fa[_mt][0], _fa[_mt][1],                         \
                                 _fa[_mt][2], _fa[_mt][3],                         \
                                 _fb[_nt][0], _fb[_nt][1]);                        \
                }                                                                    \
            }                                                                        \
        }                                                                            \
    } while (0)

// ── Kernel ────────────────────────────────────────────────────────────────────

extern "C" __global__ __launch_bounds__(128)
void matmul_cuda_tc8_4096(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N)
{
    extern __shared__ __nv_bfloat16 smem[];

    const int tid      = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id  = tid / 32;
    const int lane     = tid % 32;
    const int warp_row = warp_id / WARP_N;
    const int warp_col = warp_id % WARP_N;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[WM_TILES][WN_TILES][4] = {};

    const int num_tiles  = K / BK;
    const int main_iters = num_tiles - NS_SLOTS;  // tiles 0..main_iters-1

    // Prologue: issue first 4 tiles into slots 0-3
    ISSUE_TILE(0 * BK, 0);
    ISSUE_TILE(1 * BK, 1);
    ISSUE_TILE(2 * BK, 2);
    ISSUE_TILE(3 * BK, 3);

    // Main loop: wait → sync → compute → sync → issue-next
    // Two syncs: first ensures tile is ready; second ensures all warps
    // finish reading slot s before any warp starts writing to slot s via cp.async.
    for (int iter = 0; iter < main_iters; iter++) {
        const int s = iter % NS_SLOTS;
        asm volatile("cp.async.wait_group 3;\n" ::: "memory");
        __syncthreads();
        COMPUTE_TILE(s);
        __syncthreads();
        ISSUE_TILE((iter + NS_SLOTS) * BK, s);
    }

    // Drain remaining 4 pending tiles
    asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    __syncthreads(); COMPUTE_TILE(main_iters % NS_SLOTS);

    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    __syncthreads(); COMPUTE_TILE((main_iters + 1) % NS_SLOTS);

    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads(); COMPUTE_TILE((main_iters + 2) % NS_SLOTS);

    // Last drain tile: all copies done, no wait needed; sync for register consistency
    __syncthreads(); COMPUTE_TILE((main_iters + 3) % NS_SLOTS);

#undef ISSUE_TILE
#undef COMPUTE_TILE

    // Vectorized BF16 write-back: pack two consecutive columns into one bfloat162 store.
    const int base_row = lane / 4;
    const int base_col = (lane % 4) * 2;  // always even → 32-bit aligned

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
