// b5_tma: port of gau-nernst/learn-cuda matmul_v2a.cu.
//
// Adds TMA bulk loads + 128B-swizzled SMEM on top of b3_tc05's basic tcgen05
// MMA. Single-stage (no pipeline yet); per-K-tile: TMA load (A+B) → mbarrier
// wait → tcgen05.mma loop → mbarrier wait → next K-tile.
//
// SMEM layout (TMA-produced, 128B-swizzled):
//   A: [BLOCK_K / 64][BLOCK_M][64]   K_tile outer, M-rows × 64-K-cols inner
//   B: [BLOCK_K / 64][BLOCK_N][64]   K_tile outer, N-rows × 64-K-cols inner
//
// Both A and B share the same descriptor encoding (SBO=64, version=1, swizzle=128B).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>

#ifndef LB_MIN_BLOCKS
#define LB_MIN_BLOCKS 1
#endif

constexpr int WARP_SIZE = 32;

// ── mbarrier helpers ────────────────────────────────────────────────────────

__device__ __forceinline__ void mbarrier_init(uint32_t mb, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(count));
}

__device__ __forceinline__ void mbarrier_wait(uint32_t mb, uint32_t phase) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t"
        "bra WAIT_%=;\n\t"
        "DONE_%=:\n\t"
        "}"
        :: "r"(mb), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t mb, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(mb), "r"(bytes) : "memory");
}

// `elect.sync` — one lane in the warp returns true, others false.
__device__ __forceinline__ bool elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t .reg .pred px;\n\t"
        "elect.sync _|px, %1;\n\t"
        "@px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred) : "r"(0xFFFFFFFF));
    return pred;
}

// ── TMA load (2D) ───────────────────────────────────────────────────────────

__device__ __forceinline__ void tma_2d_load(
    uint32_t smem_dst, const void* tmap, int x, int y, uint32_t mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}

// ── tcgen05 PTX wrappers ────────────────────────────────────────────────────

__device__ __forceinline__ void tcgen05_alloc(uint32_t smem_dst, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_dst), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_dealloc(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_mma(
    uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, bool enable_d
) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "setp.ne.b32 P, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, P;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
           "r"((uint32_t)enable_d) : "memory");
}
__device__ __forceinline__ void tcgen05_commit(uint32_t smem_bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(smem_bar) : "memory");
}
__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}
__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_ld_32x32b_x8(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
        : "r"(taddr));
}

// ── Matrix descriptor (verbatim from v2) ────────────────────────────────────
//   start | (SBO_enc << 32) | (1 << 46) | (2 << 61)
//   SBO = 8*128 bytes = 1024 → encoded 64; (2 << 61) = SWIZZLE_128B
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr) {
    auto enc = [](uint64_t v) { return (v >> 4) & 0x3FFFULL; };
    constexpr uint64_t SBO = 8 * 128;
    return enc((uint64_t)smem_addr)
         | (enc(SBO) << 32)
         | (1ULL << 46)
         | (2ULL << 61);
}

__device__ __forceinline__ uint32_t make_idesc_bf16(int M, int N) {
    uint32_t d = 0;
    d |= (1u << 4);                                    // c_format = F32
    d |= (1u << 7);                                    // a_format = BF16
    d |= (1u << 10);                                   // b_format = BF16
    d |= (((uint32_t)(N >> 3) & 0x3F) << 17);          // n_dim
    d |= (((uint32_t)(M >> 4) & 0x1F) << 24);          // m_dim
    return d;
}

// ── Kernel ──────────────────────────────────────────────────────────────────

template <int BLOCK_N, int BLOCK_K>
__device__ __forceinline__ void b5_tma_impl(
    const CUtensorMap* A_tmap,
    const CUtensorMap* B_tmap,
    __nv_bfloat16* C_ptr,
    int M, int N, int K
) {
    constexpr int BLOCK_M = 128;
    constexpr int MMA_K   = 16;
    constexpr int NUM_WARPS = 4;
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
    static_assert(BLOCK_K % 64 == 0, "BLOCK_K must be a multiple of 64 for 64-wide K-tiles");

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int bid     = blockIdx.x;

    const int grid_m = M / BLOCK_M;
    const int grid_n = N / BLOCK_N;
    const int bid_m  = bid / grid_n;
    const int bid_n  = bid % grid_n;
    const int off_m  = bid_m * BLOCK_M;
    const int off_n  = bid_n * BLOCK_N;

    // ── Pipeline state ──────────────────────────────────────────────────────
    //
    //   NUM_STAGES SMEM ring-buffer slots, each with its own pair of mbarriers:
    //     tile_ready_mbar[s] — fires when TMA finishes loading slot s
    //     mma_done_mbar[s]   — fires when all MMAs consuming slot s have completed
    //
    //   We currently use NUM_STAGES=1 (no pipelining): the main loop fully
    //   processes each K-tile before moving to the next.  All macros take a
    //   `slot` argument so the structure is ready for NS>1 in a future revision.
    //
    constexpr int NUM_STAGES = 1;
    constexpr int A_SLOT_BYTES = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    constexpr int B_SLOT_BYTES = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);
    constexpr int CP_BYTES_PER_TILE = A_SLOT_BYTES + B_SLOT_BYTES;

    extern __shared__ __align__(1024) char smem[];
    const uint32_t SMEM_BASE = (uint32_t)__cvta_generic_to_shared(smem);
    #define A_SMEM_SLOT(s_) (SMEM_BASE + (s_) * A_SLOT_BYTES)
    #define B_SMEM_SLOT(s_) (SMEM_BASE + NUM_STAGES * A_SLOT_BYTES + (s_) * B_SLOT_BYTES)

    __shared__ uint64_t tile_ready_mbar[NUM_STAGES];
    __shared__ uint64_t mma_done_mbar[NUM_STAGES];
    __shared__ uint32_t tmem_addr_holder[1];

    // ── One-time setup ──────────────────────────────────────────────────────
    if (warp_id == 0 && elect_sync()) {
        #pragma unroll
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init((uint32_t)__cvta_generic_to_shared(&tile_ready_mbar[s]), 1);
            mbarrier_init((uint32_t)__cvta_generic_to_shared(&mma_done_mbar[s]),   1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        tcgen05_alloc((uint32_t)__cvta_generic_to_shared(tmem_addr_holder), BLOCK_N);
    }
    __syncthreads();
    const uint32_t taddr = tmem_addr_holder[0];

    // Per-slot parity counters (in registers): which phase to wait for next.
    uint32_t tile_ready_phase[NUM_STAGES] = {};
    uint32_t mma_done_phase  [NUM_STAGES] = {};
    const uint32_t idesc = make_idesc_bf16(BLOCK_M, BLOCK_N);

    // ── Pipeline event macros ───────────────────────────────────────────────
    // Each macro is one event; the K-loop below reads as the dependency graph.
    //
    //   LOAD_TILE(slot, k0)
    //       Issue async TMA load of K-tile k0 into SMEM slot `slot`.
    //       Signals tile_ready_mbar[slot] when bytes have landed in SMEM.
    //
    //   WAIT_ON_TILE_READY(slot)
    //       Block all threads until LOAD_TILE(slot, _) has completed.
    //       Ensures SMEM data is visible to tcgen05.mma (issues the required
    //       async-proxy fence).
    //
    //   MMA_ASYNC(slot, first_iter)
    //       Issue all K_MMAS tcgen05.mma's consuming SMEM slot `slot`,
    //       accumulating into D (TMEM).  On the very first MMA of the kernel
    //       (first_iter && first_mma), overwrite D instead of accumulating.
    //       Signals mma_done_mbar[slot] when all MMAs have completed.
    //
    //   WAIT_ON_MMA(slot)
    //       Block all threads until MMA_ASYNC(slot, _) has completed.
    //       After this, the SMEM slot is free to be re-loaded.

#define LOAD_TILE(slot_, k0_)                                                          \
    do {                                                                                \
        const int _slot = (slot_);                                                      \
        if (warp_id == 0 && elect_sync()) {                                             \
            const uint32_t _mb = (uint32_t)__cvta_generic_to_shared(&tile_ready_mbar[_slot]); \
            _Pragma("unroll")                                                           \
            for (int _k = 0; _k < BLOCK_K / 64; _k++) {                                 \
                const int _off_k = (k0_) + _k * 64;                                     \
                tma_2d_load(A_SMEM_SLOT(_slot) + _k * BLOCK_M * 128,                    \
                            A_tmap, _off_k, off_m, _mb);                                \
                tma_2d_load(B_SMEM_SLOT(_slot) + _k * BLOCK_N * 128,                    \
                            B_tmap, _off_k, off_n, _mb);                                \
            }                                                                           \
            mbarrier_arrive_expect_tx(_mb, CP_BYTES_PER_TILE);                          \
        }                                                                               \
    } while (0)

#define WAIT_ON_TILE_READY(slot_)                                                       \
    do {                                                                                \
        const int _slot = (slot_);                                                      \
        const uint32_t _mb = (uint32_t)__cvta_generic_to_shared(&tile_ready_mbar[_slot]); \
        mbarrier_wait(_mb, tile_ready_phase[_slot]);                                    \
        tile_ready_phase[_slot] ^= 1;                                                   \
        /* async-proxy → tcgen05 visibility: required between TMA and MMA */            \
        asm volatile("tcgen05.fence::after_thread_sync;");                              \
    } while (0)

#define MMA_ASYNC(slot_, first_iter_)                                                   \
    do {                                                                                \
        const int _slot = (slot_);                                                      \
        if (warp_id == 0 && elect_sync()) {                                             \
            _Pragma("unroll")                                                           \
            for (int _k1 = 0; _k1 < BLOCK_K / 64; _k1++) {                              \
                _Pragma("unroll")                                                       \
                for (int _k2 = 0; _k2 < 64 / MMA_K; _k2++) {                            \
                    const uint64_t _a = make_desc(                                      \
                        A_SMEM_SLOT(_slot) + _k1 * BLOCK_M * 128 + _k2 * 32);           \
                    const uint64_t _b = make_desc(                                      \
                        B_SMEM_SLOT(_slot) + _k1 * BLOCK_N * 128 + _k2 * 32);           \
                    const bool _accumulate = !(first_iter_) || _k1 > 0 || _k2 > 0;      \
                    tcgen05_mma(taddr, _a, _b, idesc, _accumulate);                     \
                }                                                                       \
            }                                                                           \
            tcgen05_commit(                                                             \
                (uint32_t)__cvta_generic_to_shared(&mma_done_mbar[_slot]));             \
        }                                                                               \
    } while (0)

#define WAIT_ON_MMA(slot_)                                                              \
    do {                                                                                \
        const int _slot = (slot_);                                                      \
        const uint32_t _mb = (uint32_t)__cvta_generic_to_shared(&mma_done_mbar[_slot]); \
        mbarrier_wait(_mb, mma_done_phase[_slot]);                                      \
        mma_done_phase[_slot] ^= 1;                                                     \
    } while (0)

    // ── Main K-iter loop ────────────────────────────────────────────────────
    //
    //   Single-stage (no pipelining): each iteration runs the full event chain
    //   LOAD → WAIT → MMA → WAIT before starting the next K-tile.
    //
    //   Dependency chain per iter:
    //       LOAD_TILE(0, k0)
    //           │
    //           ▼
    //       WAIT_ON_TILE_READY(0)        ← SMEM slot 0 is populated
    //           │
    //           ▼
    //       MMA_ASYNC(0)                 ← issues all K_MMAS to read slot 0
    //           │
    //           ▼
    //       WAIT_ON_MMA(0)               ← SMEM slot 0 is now free for reuse
    //
    const int num_iters = K / BLOCK_K;
    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
        const int k0 = iter_k * BLOCK_K;

        LOAD_TILE(0, k0);
        WAIT_ON_TILE_READY(0);
        MMA_ASYNC(0, /*first_iter=*/(iter_k == 0));
        WAIT_ON_MMA(0);
    }

#undef LOAD_TILE
#undef WAIT_ON_TILE_READY
#undef MMA_ASYNC
#undef WAIT_ON_MMA

    // ── Epilogue: tcgen05.ld.32x32b.x8 ──
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int lane = tid % WARP_SIZE;
    int gr = off_m + warp_id * 32 + lane;
    #pragma unroll
    for (int n = 0; n < BLOCK_N; n += 8) {
        float tmp[8];
        uint32_t addr = taddr + ((warp_id * 32) << 16) + n;
        tcgen05_ld_32x32b_x8(addr, tmp);
        tcgen05_wait_ld();

        if (gr < M) {
            __nv_bfloat162 packed[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                packed[i] = __floats2bfloat162_rn(tmp[2*i], tmp[2*i+1]);
            }
            int gc = off_n + n;
            if (gc + 7 < N) {
                *reinterpret_cast<int4*>(&C_ptr[gr * N + gc]) =
                    *reinterpret_cast<int4*>(packed);
            } else {
                #pragma unroll
                for (int i = 0; i < 8 && gc + i < N; i++) {
                    C_ptr[gr * N + gc + i] = __float2bfloat16(tmp[i]);
                }
            }
        }
    }

    __syncthreads();
    if (warp_id == 0) {
        tcgen05_dealloc(taddr, BLOCK_N);
    }
}

// ── Launchers ───────────────────────────────────────────────────────────────

// __grid_constant__ pass-by-value matches what gau-nernst v2 does. pycuda's
// arg-packer copies a 128-byte numpy ndarray into the kernel param area as
// the literal CUtensorMap contents.
#define MAKE_LAUNCHER(BN_, BK_)                                                    \
extern "C" __global__ __launch_bounds__(128, LB_MIN_BLOCKS)                        \
void matmul_b5_tma_bm128_bn##BN_##_bk##BK_(                                        \
    const __grid_constant__ CUtensorMap A_tmap,                                    \
    const __grid_constant__ CUtensorMap B_tmap,                                    \
    __nv_bfloat16* C_ptr, int M, int N, int K)                                     \
{                                                                                   \
    b5_tma_impl<BN_, BK_>(&A_tmap, &B_tmap, C_ptr, M, N, K);                       \
}

MAKE_LAUNCHER(64, 64)   MAKE_LAUNCHER(64, 128)  MAKE_LAUNCHER(64, 256)
MAKE_LAUNCHER(128, 64)  MAKE_LAUNCHER(128, 128) MAKE_LAUNCHER(128, 256)
MAKE_LAUNCHER(256, 64)  MAKE_LAUNCHER(256, 128) MAKE_LAUNCHER(256, 256)

#undef MAKE_LAUNCHER
