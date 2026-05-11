#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

/*
 * H2 Stage 1: TMA + mbarrier for async loads; mma.sync compute (no swizzle).
 *
 * What changed vs tc5_lb:
 *   REMOVED: __pipeline_memcpy_async per-thread cp.async loads (ISSUE_TILE)
 *   REMOVED: XOR swizzle in COMPUTE_TILE ldmatrix addresses
 *   REMOVED: __pipeline_wait_prior / __pipeline_commit
 *   ADDED:   TMA bulk 2D copy (one thread issues per tile)
 *   ADDED:   mbarrier arrive-expect-tx + wait-parity for sync
 *
 * Compute path (ldmatrix + mma.sync) is identical to tc5_lb.
 * SMEM is linear (no swizzle) → bank conflicts exist but are tolerable for S1.
 *
 * Compiled with -arch=sm_90a (required for cp.async.bulk.tensor PTX).
 * Compiled with -DLB_MIN_BLOCKS=N (same tunable launch-bounds as tc5_lb).
 */

#ifndef LB_MIN_BLOCKS
#define LB_MIN_BLOCKS 2
#endif

// ── TMA descriptor type (matches CUtensorMap; avoids needing cuda.h) ─────────

struct alignas(64) TmaDesc {
    uint64_t opaque[16];
};

// ── mbarrier PTX helpers ──────────────────────────────────────────────────────

// Initialise mbarrier with expected arrival count (call from one thread).
__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t count) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(addr), "r"(count)
        : "memory");
}

// One thread: software-arrive + declare that TMA will deliver tx_bytes.
// Decrements arrive count by 1; adds tx_bytes to the expected TX count.
// The mbarrier completes when arrive count AND tx count both reach 0.
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    // Correct form from PTX ISA 8.0: release.cta semantics, shared::cta space.
    // Returns a state token we discard via a dummy output register.
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(mbar);
    uint64_t state;
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;\n"
        : "=l"(state)
        : "r"(addr), "r"(tx_bytes)
        : "memory");
}

// All threads: spin-wait until mbarrier completes (phase matches parity).
// Use test_wait (non-blocking probe inside a loop); adequate for correctness.
// We always wait on phase=0 because we reinit before every use.
__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t phase) {
    uint32_t done = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(mbar);
    while (!done) {
        asm volatile(
            "{\n"
            ".reg .pred P;\n"
            "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2;\n"
            "selp.u32 %0, 1, 0, P;\n"
            "}\n"
            : "=r"(done)
            : "r"(addr), "r"(phase)
            : "memory");
    }
}

// ── TMA load PTX helper ───────────────────────────────────────────────────────

// Issue a 2D TMA bulk copy into smem_ptr, signalling mbar on completion.
// coord0 = innermost (column) coordinate, coord1 = outer (row) coordinate.
// Both coords are element-based (the descriptor encodes element size).
__device__ __forceinline__ void tma_load_2d(
    const TmaDesc* __restrict__ desc,
    void*          smem_ptr,
    uint64_t*      mbar,
    int32_t        coord0,
    int32_t        coord1
) {
    uint32_t dst  = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
        :
        : "r"(dst),
          "l"((unsigned long long)desc),
          "r"(mbar_addr),
          "r"(coord0), "r"(coord1)
        : "memory");
}

// ── Compute helpers (identical to tc5_lb) ────────────────────────────────────

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_ptr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_ptr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& r0, uint32_t& r1, uint32_t smem_ptr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1) : "r"(smem_ptr));
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
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ── Kernel implementation ─────────────────────────────────────────────────────

template <int BM, int BN, int BK, int NUM_WARPS>
__device__ __forceinline__ void h2s1_impl(
    const TmaDesc& tma_A,
    const TmaDesc& tma_B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    // ── Warp/thread layout (identical to tc5_lb) ──────────────────────────────
    constexpr int WARP_N      = 2;
    constexpr int WARP_M      = NUM_WARPS / WARP_N;
    constexpr int WARP_TILE_M = BM / WARP_M;
    constexpr int WARP_TILE_N = BN / WARP_N;
    constexpr int WM_TILES    = WARP_TILE_M / 16;
    constexpr int WN_TILES    = WARP_TILE_N / 8;

    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid % 32;
    const int warp_row = warp_id / WARP_N;
    const int warp_col = warp_id % WARP_N;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // ── Shared memory layout ──────────────────────────────────────────────────
    // A[2][BM][BK]  B[2][BK][BN]  mbar[2]   (all in one extern __shared__ blob)
    //
    // No swizzle: linear row-major layout.
    // Bank conflicts will occur on ldmatrix (same as tc5 before swizzle was added).
    // This is intentional for Stage 1 — we are testing TMA plumbing, not peak perf.
    extern __shared__ char smem_raw[];
    constexpr int A_BYTES  = 2 * BM * BK * 2;   // 2 stages × BM×BK × sizeof(bf16)
    constexpr int B_BYTES  = 2 * BK * BN * 2;
    constexpr int MBAR_OFF = (A_BYTES + B_BYTES + 7) & ~7; // 8-byte aligned

    auto A_shared = reinterpret_cast<__nv_bfloat16 (*)[BM][BK]>(smem_raw);
    auto B_shared = reinterpret_cast<__nv_bfloat16 (*)[BK][BN]>(smem_raw + A_BYTES);
    auto mbar     = reinterpret_cast<uint64_t*>(smem_raw + MBAR_OFF);

    // ── Accumulator ───────────────────────────────────────────────────────────
    float acc[WM_TILES][WN_TILES][4] = {};

    const int num_tiles = K / BK;

    // ── Prologue: init both mbarriers and issue tile 0 ────────────────────────
    // Only thread 0 touches mbarriers and issues TMA.
    // mbar_init(mbar, 1): one software arrive (from arrive_expect_tx); TMA is the TX.
    if (tid == 0) {
        mbar_init(&mbar[0], 1);
        mbar_init(&mbar[1], 1);
        mbar_arrive_expect_tx(&mbar[0], (BM * BK + BK * BN) * 2);
        tma_load_2d(&tma_A, &A_shared[0][0][0], &mbar[0], /*col=*/0, /*row=*/block_row);
        tma_load_2d(&tma_B, &B_shared[0][0][0], &mbar[0], /*col=*/block_col, /*row=*/0);
    }
    // Ensure mbar[0] is initialised and TMA is issued before any thread waits on it.
    __syncthreads();

    // ── Main loop ─────────────────────────────────────────────────────────────
    for (int k = 0; k < num_tiles - 1; k++) {
        const int cur = k & 1;
        const int nxt = 1 - cur;

        // Thread 0: reinit mbar[nxt] and issue next tile.
        // Safe: mbar[nxt] was completed and all threads exited its wait in the
        // previous iteration (enforced by __syncthreads at end of loop body).
        if (tid == 0) {
            mbar_init(&mbar[nxt], 1);
            mbar_arrive_expect_tx(&mbar[nxt], (BM * BK + BK * BN) * 2);
            tma_load_2d(&tma_A, &A_shared[nxt][0][0], &mbar[nxt],
                        /*col=*/(k + 1) * BK, /*row=*/block_row);
            tma_load_2d(&tma_B, &B_shared[nxt][0][0], &mbar[nxt],
                        /*col=*/block_col, /*row=*/(k + 1) * BK);
        }

        // All threads: wait for cur tile to arrive (phase=0 after fresh mbar_init).
        mbar_wait(&mbar[cur], 0);

        // ── COMPUTE_TILE (tc5_lb logic, XOR swizzle removed) ──────────────────
        // A address: A_shared[cur][row][col_group * 8], col_group = _kk*2 + lane/16
        // B address: B_shared[cur][row][col], col = warp_col*WARP_TILE_N + nt*8
        // No _phys XOR — linear access. Bank conflicts are present but accepted.
        #pragma unroll
        for (int _kk = 0; _kk < BK / 16; _kk++) {
            uint32_t _fa[WM_TILES][4];
            #pragma unroll
            for (int _mt = 0; _mt < WM_TILES; _mt++) {
                const int _ar = warp_row * WARP_TILE_M + _mt * 16 + (lane % 16);
                const int _lg = _kk * 2 + (lane / 16);
                // Linear: use _lg directly (was XOR'd with (_ar/A_SHIFT)%A_SWZ in tc5_lb)
                ldmatrix_x4(_fa[_mt][0], _fa[_mt][1], _fa[_mt][2], _fa[_mt][3],
                    __cvta_generic_to_shared(&A_shared[cur][_ar][_lg * 8]));
            }
            uint32_t _fb[WN_TILES][2];
            #pragma unroll
            for (int _nt = 0; _nt < WN_TILES; _nt++) {
                const int _br = _kk * 16 + (lane % 16);
                const int _nc = warp_col * WARP_TILE_N + _nt * 8;
                // Linear: use _nc directly (was XOR'd with _br%B_SWZ in tc5_lb)
                ldmatrix_x2_trans(_fb[_nt][0], _fb[_nt][1],
                    __cvta_generic_to_shared(&B_shared[cur][_br][_nc]));
            }
            #pragma unroll
            for (int _mt = 0; _mt < WM_TILES; _mt++)
                #pragma unroll
                for (int _nt = 0; _nt < WN_TILES; _nt++)
                    mma_m16n8k16(acc[_mt][_nt][0], acc[_mt][_nt][1],
                                 acc[_mt][_nt][2], acc[_mt][_nt][3],
                                 _fa[_mt][0], _fa[_mt][1], _fa[_mt][2], _fa[_mt][3],
                                 _fb[_nt][0], _fb[_nt][1]);
        }

        // Sync ensures:
        //  (a) all threads finished reading smem[cur] before thread 0 reuses it next iter
        //  (b) mbar[nxt] reinit (issued by thread 0 above) is visible to all threads
        //      before the next iteration's mbar_wait(&mbar[nxt], 0)
        __syncthreads();
    }

    // ── Epilogue: wait for last tile and compute ──────────────────────────────
    const int last = (num_tiles - 1) & 1;
    mbar_wait(&mbar[last], 0);

    #pragma unroll
    for (int _kk = 0; _kk < BK / 16; _kk++) {
        uint32_t _fa[WM_TILES][4];
        #pragma unroll
        for (int _mt = 0; _mt < WM_TILES; _mt++) {
            const int _ar = warp_row * WARP_TILE_M + _mt * 16 + (lane % 16);
            const int _lg = _kk * 2 + (lane / 16);
            ldmatrix_x4(_fa[_mt][0], _fa[_mt][1], _fa[_mt][2], _fa[_mt][3],
                __cvta_generic_to_shared(&A_shared[last][_ar][_lg * 8]));
        }
        uint32_t _fb[WN_TILES][2];
        #pragma unroll
        for (int _nt = 0; _nt < WN_TILES; _nt++) {
            const int _br = _kk * 16 + (lane % 16);
            const int _nc = warp_col * WARP_TILE_N + _nt * 8;
            ldmatrix_x2_trans(_fb[_nt][0], _fb[_nt][1],
                __cvta_generic_to_shared(&B_shared[last][_br][_nc]));
        }
        #pragma unroll
        for (int _mt = 0; _mt < WM_TILES; _mt++)
            #pragma unroll
            for (int _nt = 0; _nt < WN_TILES; _nt++)
                mma_m16n8k16(acc[_mt][_nt][0], acc[_mt][_nt][1],
                             acc[_mt][_nt][2], acc[_mt][_nt][3],
                             _fa[_mt][0], _fa[_mt][1], _fa[_mt][2], _fa[_mt][3],
                             _fb[_nt][0], _fb[_nt][1]);
    }

    // ── Write accumulators to C (identical to tc5_lb vectorised epilogue) ─────
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

// ── Kernel entry points ───────────────────────────────────────────────────────

#define MAKE_LAUNCHER(BM_, BN_, BK_, NW_)                                      \
extern "C" __global__ __launch_bounds__(NW_ * 32, LB_MIN_BLOCKS)              \
void matmul_h2s1_bm##BM_##_bn##BN_##_bk##BK_##_nw##NW_(                       \
    const __grid_constant__ TmaDesc tma_A,                                     \
    const __grid_constant__ TmaDesc tma_B,                                     \
    __nv_bfloat16* __restrict__ C, int M, int K, int N)                        \
{                                                                               \
    h2s1_impl<BM_, BN_, BK_, NW_>(tma_A, tma_B, C, M, K, N);                  \
}

// NW=4
MAKE_LAUNCHER( 64,  64, 16, 4) MAKE_LAUNCHER( 64,  64, 32, 4) MAKE_LAUNCHER( 64,  64, 64, 4)
MAKE_LAUNCHER( 64, 128, 16, 4) MAKE_LAUNCHER( 64, 128, 32, 4) MAKE_LAUNCHER( 64, 128, 64, 4)
MAKE_LAUNCHER( 64, 256, 16, 4) MAKE_LAUNCHER( 64, 256, 32, 4) MAKE_LAUNCHER( 64, 256, 64, 4)
MAKE_LAUNCHER(128,  64, 16, 4) MAKE_LAUNCHER(128,  64, 32, 4) MAKE_LAUNCHER(128,  64, 64, 4)
MAKE_LAUNCHER(128, 128, 16, 4) MAKE_LAUNCHER(128, 128, 32, 4) MAKE_LAUNCHER(128, 128, 64, 4)
MAKE_LAUNCHER(256,  64, 16, 4) MAKE_LAUNCHER(256,  64, 32, 4) MAKE_LAUNCHER(256,  64, 64, 4)

// NW=8
MAKE_LAUNCHER( 64,  64, 16, 8) MAKE_LAUNCHER( 64,  64, 32, 8) MAKE_LAUNCHER( 64,  64, 64, 8)
MAKE_LAUNCHER( 64, 128, 16, 8) MAKE_LAUNCHER( 64, 128, 32, 8) MAKE_LAUNCHER( 64, 128, 64, 8)
MAKE_LAUNCHER( 64, 256, 16, 8) MAKE_LAUNCHER( 64, 256, 32, 8) MAKE_LAUNCHER( 64, 256, 64, 8)
MAKE_LAUNCHER(128,  64, 16, 8) MAKE_LAUNCHER(128,  64, 32, 8) MAKE_LAUNCHER(128,  64, 64, 8)
MAKE_LAUNCHER(128, 128, 16, 8) MAKE_LAUNCHER(128, 128, 32, 8) MAKE_LAUNCHER(128, 128, 64, 8)
MAKE_LAUNCHER(128, 256, 16, 8) MAKE_LAUNCHER(128, 256, 32, 8) MAKE_LAUNCHER(128, 256, 64, 8)
MAKE_LAUNCHER(256,  64, 16, 8) MAKE_LAUNCHER(256,  64, 32, 8) MAKE_LAUNCHER(256,  64, 64, 8)
MAKE_LAUNCHER(256, 128, 16, 8) MAKE_LAUNCHER(256, 128, 32, 8) MAKE_LAUNCHER(256, 128, 64, 8)

#undef MAKE_LAUNCHER
