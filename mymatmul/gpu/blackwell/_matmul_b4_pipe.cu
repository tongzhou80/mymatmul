// b4_pipe: tcgen05.mma with NS-deep cp.async pipeline + multi-MMA per K-iter
//
// Differences from b3_tc05:
//   1. Tunable BK ∈ {16, 32, 64, 128} — K_MMAS = BK/16 MMAs back-to-back per K-iter.
//   2. NS-deep cp.async ring buffer (NUM_STAGES) for SMEM tiles, prologue / main / drain
//      loop structure (same recipe as b2_ms).
//   3. One mbarrier wait per K-iter, not per MMA — async MMA-MMA overlap within an iter.
//   4. LB (__launch_bounds__) configurable via -DLB_MIN_BLOCKS for occupancy tuning.
//
// SMEM layout per stage (3D-tiled, matches descriptor LBO=height*16, SBO=128):
//   A_stage[K_TILES][BM][K_INNER=8]    K_TILES = BK/8
//   B_stage[K_TILES][BN][K_INNER=8]    (transposed)
//
// Total SMEM = NS × 2 × BM × BK × 2 bytes + tmem_holder + NS mbarriers (one per stage).

#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>

#ifndef LB_MIN_BLOCKS
#define LB_MIN_BLOCKS 1
#endif

// ── Descriptor builders ─────────────────────────────────────────────────────

// K-major operand: SMEM tile [K_TILES][height][K_INNER=8], 8 K-elements inner.
//   LBO = height * 16 bytes  (= bytes per K_TILE slab)
//   SBO = 8 * 16 = 128 bytes (= bytes per 8-row M block)
__device__ __forceinline__ uint64_t make_smem_desc(const void* smem_ptr, int height) {
    uint32_t saddr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    auto enc = [](uint64_t bytes) -> uint64_t { return (bytes >> 4) & ((1ULL << 14) - 1); };
    return enc((uint64_t)saddr)
         | (enc((uint64_t)height * 16) << 16)
         | (enc((uint64_t)8 * 16)      << 32);
}

__device__ __forceinline__ uint32_t make_idesc_bf16(int M, int N) {
    uint32_t d = 0;
    d |= (1u << 4);                                   // c_format = F32
    d |= (1u << 7);                                   // a_format = BF16
    d |= (1u << 10);                                  // b_format = BF16
    // a_major = 0, b_major = 0 — both operands K-major (K is inner SMEM dim)
    d |= (((uint32_t)(N >> 3) & 0x3F) << 17);         // n_dim = N>>3
    d |= (((uint32_t)(M >> 4) & 0x1F) << 24);         // m_dim = M>>4
    return d;
}

// ── tcgen05 PTX wrappers ────────────────────────────────────────────────────

__device__ __forceinline__ void tcgen05_alloc(uint32_t* smem_dst, uint32_t n_cols) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(s), "r"(n_cols) : "memory");
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
__device__ __forceinline__ void tcgen05_commit(uint64_t* smem_bar) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(s) : "memory");
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

// ── Kernel ──────────────────────────────────────────────────────────────────

template <int BM, int BN, int BK, int NUM_WARPS, int NUM_STAGES>
__device__ __forceinline__ void b4_pipe_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    static_assert(BM == 128 && BN == 128, "b4 fixed at 128x128 tile");
    static_assert(BK % 16 == 0, "BK must be multiple of 16");
    static_assert(NUM_WARPS == 4, "epilogue assumes 4 warps drain 4*32=128 rows");
    constexpr int THREADS = NUM_WARPS * 32;
    constexpr int K_INNER = 8;
    constexpr int K_TILES = BK / K_INNER;   // 2, 4, 8, 16 for BK = 16, 32, 64, 128
    constexpr int K_MMAS  = BK / 16;        // 1, 2, 4, 8
    constexpr int NS      = NUM_STAGES;

    // ── SMEM layout: both A and B in K-major 3D-tiled form.
    //   A[NS][K_TILES][BM][K_INNER=8]
    //   B[NS][K_TILES][BN][K_INNER=8]   (transposed: N outer, K inner)
    extern __shared__ uint8_t smem_raw[];
    constexpr int A_STAGE_B = K_TILES * BM * K_INNER * 2;
    constexpr int B_STAGE_B = K_TILES * BN * K_INNER * 2;
    auto smem_A = reinterpret_cast<__nv_bfloat16 (*)[K_TILES][BM][K_INNER]>(smem_raw);
    auto smem_B = reinterpret_cast<__nv_bfloat16 (*)[K_TILES][BN][K_INNER]>(
        smem_raw + NS * A_STAGE_B);
    auto tmem_holder = reinterpret_cast<uint32_t*>(
        smem_raw + NS * (A_STAGE_B + B_STAGE_B));
    auto mbar = reinterpret_cast<uint64_t*>(
        smem_raw + NS * (A_STAGE_B + B_STAGE_B) + 16);

    const int tid       = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id   = tid / 32;
    const int lane      = tid % 32;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // ── Setup: TMEM alloc + mbar init ──
    if (warp_id == 0) {
        tcgen05_alloc(tmem_holder, BN);
        tcgen05_relinquish();
    }
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NS; s++) {
            uint32_t mb = (uint32_t)__cvta_generic_to_shared(&mbar[s]);
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb) : "memory");
        }
    }
    __syncthreads();
    const uint32_t tmem_d = tmem_holder[0];

    // ── ISSUE_TILE: cp.async one BK-stripe into stage s ──
    auto issue_tile = [&](int stage, int k0) {
        // Load A: K_TILES × BM × K_INNER bf16 = K_TILES * BM * 16 bytes
        // 128 threads × 16 bytes/chunk = enough for BM*K_TILES = 128*K_TILES chunks
        // when K_TILES <= 2; for larger K_TILES we iterate.
        constexpr int A_CHUNKS = K_TILES * BM;           // total 16-byte chunks for A
        constexpr int A_PER_THR = (A_CHUNKS + THREADS - 1) / THREADS;
        #pragma unroll
        for (int i = 0; i < A_PER_THR; i++) {
            int g = tid + i * THREADS;
            if (g < A_CHUNKS) {
                int kt_idx = g / BM;
                int m      = g % BM;
                const __nv_bfloat16* gptr =
                    &A[(block_row + m) * K + k0 + kt_idx * K_INNER];
                __pipeline_memcpy_async(&smem_A[stage][kt_idx][m][0], gptr, 16);
            }
        }
        // Load B transposed scalar: 1 BF16 per access (K-inner SMEM doesn't admit
        // 16-byte cp.async from N-coalesced GMEM). TODO(b5): vectorize via N-major
        // SMEM + MN-major descriptor, OR via SMEM-to-SMEM transpose scratch.
        for (int n = tid; n < BN; n += THREADS) {
            #pragma unroll
            for (int kt_idx = 0; kt_idx < K_TILES; kt_idx++) {
                #pragma unroll
                for (int ki = 0; ki < K_INNER; ki++) {
                    smem_B[stage][kt_idx][n][ki] =
                        B[(k0 + kt_idx * K_INNER + ki) * N + block_col + n];
                }
            }
        }
        __pipeline_commit();
    };

    // ── COMPUTE_TILE: issue K_MMAS MMAs from one stage + commit to its mbar ──
    //   Both descriptors advance by 2 K_INNER tiles per MMA (= 16 K elements).
    auto compute_tile = [&](int stage, bool first_iter) {
        if (tid == 0) {
            uint32_t idesc = make_idesc_bf16(BM, BN);
            #pragma unroll
            for (int kk = 0; kk < K_MMAS; kk++) {
                uint64_t descA = make_smem_desc(&smem_A[stage][kk * 2][0][0], BM);
                uint64_t descB = make_smem_desc(&smem_B[stage][kk * 2][0][0], BN);
                bool enable_d = !first_iter || kk > 0;
                tcgen05_mma(tmem_d, descA, descB, idesc, enable_d);
            }
            tcgen05_commit(&mbar[stage]);
        }
    };

    // ── mbarrier wait helper ──
    auto wait_mbar = [&](int stage, uint32_t phase) {
        uint32_t mb = (uint32_t)__cvta_generic_to_shared(&mbar[stage]);
        asm volatile(
            "{\n\t .reg .pred P;\n\t"
            "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@P bra DONE_%=;\n\t"
            "bra WAIT_%=;\n\t"
            "DONE_%=:\n\t"
            "}"
            :: "r"(mb), "r"(phase) : "memory");
    };

    const int num_tiles = K / BK;
    uint32_t phases = 0;   // bit s = parity for mbar[s]

    // ── Prologue: fill pipeline with NS-1 stages worth of loads ──
    #pragma unroll
    for (int s = 0; s < NS - 1; s++) {
        issue_tile(s, s * BK);
    }

    // ── Main loop: tiles 0 .. num_tiles - NS, always issues a new load ──
    for (int kt = 0; kt < num_tiles - (NS - 1); kt++) {
        int load_stage    = (kt + NS - 1) % NS;
        int compute_stage = kt % NS;

        issue_tile(load_stage, (kt + NS - 1) * BK);
        __pipeline_wait_prior(NS - 1);
        __syncthreads();

        compute_tile(compute_stage, /*first_iter=*/(kt == 0));
        uint32_t cur_phase = (phases >> compute_stage) & 1;
        wait_mbar(compute_stage, cur_phase);
        phases ^= (1u << compute_stage);

        __syncthreads();
    }

    // ── Drain: last NS-1 tiles, no new issues ──
    #pragma unroll
    for (int d = NS - 2; d >= 0; d--) {
        __pipeline_wait_prior(d);
        __syncthreads();
        int kt = num_tiles - 1 - d;
        int compute_stage = kt % NS;

        compute_tile(compute_stage, /*first_iter=*/(kt == 0));
        uint32_t cur_phase = (phases >> compute_stage) & 1;
        wait_mbar(compute_stage, cur_phase);
        phases ^= (1u << compute_stage);

        __syncthreads();
    }

    // ── Epilogue: tcgen05.fence → 32x32b.x8 loop → write to GMEM ──
    tcgen05_fence_after_thread_sync();

    int gr = block_row + warp_id * 32 + lane;
    #pragma unroll
    for (int n = 0; n < BN; n += 8) {
        float tmp[8];
        const uint32_t addr = tmem_d + ((warp_id * 32) << 16) + n;
        tcgen05_ld_32x32b_x8(addr, tmp);
        tcgen05_wait_ld();

        if (gr < M) {
            __nv_bfloat162 packed[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                packed[i] = __floats2bfloat162_rn(tmp[2*i], tmp[2*i + 1]);
            }
            int gc = block_col + n;
            if (gc + 7 < N) {
                *reinterpret_cast<int4*>(&C[gr * N + gc]) =
                    *reinterpret_cast<int4*>(packed);
            } else {
                #pragma unroll
                for (int i = 0; i < 8 && gc + i < N; i++) {
                    C[gr * N + gc + i] = __float2bfloat16(tmp[i]);
                }
            }
        }
    }

    __syncthreads();
    if (warp_id == 0) {
        tcgen05_dealloc(tmem_d, BN);
    }
}

// ── Launchers ───────────────────────────────────────────────────────────────

#define MAKE_LAUNCHER(BK_, NS_)                                                        \
extern "C" __global__ __launch_bounds__(128, LB_MIN_BLOCKS)                            \
void matmul_b4_pipe_bm128_bn128_bk##BK_##_nw4_ns##NS_(                                 \
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,          \
    __nv_bfloat16* __restrict__ C, int M, int K, int N)                                \
{                                                                                       \
    b4_pipe_impl<128, 128, BK_, 4, NS_>(A, B, C, M, K, N);                             \
}

MAKE_LAUNCHER(16, 2) MAKE_LAUNCHER(16, 3) MAKE_LAUNCHER(16, 4) MAKE_LAUNCHER(16, 5)
MAKE_LAUNCHER(32, 2) MAKE_LAUNCHER(32, 3) MAKE_LAUNCHER(32, 4) MAKE_LAUNCHER(32, 5)
MAKE_LAUNCHER(64, 2) MAKE_LAUNCHER(64, 3) MAKE_LAUNCHER(64, 4) MAKE_LAUNCHER(64, 5)
MAKE_LAUNCHER(128, 2) MAKE_LAUNCHER(128, 3)

#undef MAKE_LAUNCHER
