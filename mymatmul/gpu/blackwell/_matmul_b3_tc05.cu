// b3_tc05: first Blackwell tcgen05 kernel.
//
// Minimal proof-of-life:
//   - One CTA per output tile, BM=128 BN=128 BK=16 (single MMA per K-iter).
//   - cp.async loads A (M x K row-major) and B^T (N x K, transposed at load).
//   - tcgen05.mma.cta_group::1.kind::f16, SWIZZLE_NONE on both operands.
//   - TMEM holds the f32 D accumulator (128 rows x 128 cols).
//   - Single thread (tid 0) issues alloc, mma, dealloc.
//   - Sync via tcgen05.commit + mbarrier.
//   - Epilogue: each warp owns 32 rows of D in TMEM, reads via tcgen05.ld,
//     writes back to GMEM as bf16.
//
// Field encodings (from CUTLASS cute/arch/mma_sm100_desc.hpp):
//   InstrDescriptor[32]:
//     [0:2)   sparse_id2     = 0
//     [2:3)   sparse_flag    = 0
//     [3:4)   saturate       = 0
//     [4:6)   c_format       = 1 (F32)
//     [7:10)  a_format       = 1 (BF16)
//     [10:13) b_format       = 1 (BF16)
//     [13:14) a_negate       = 0
//     [14:15) b_negate       = 0
//     [15:16) a_major        = 0 (K-major)   // operand A's contiguous dim is K
//     [16:17) b_major        = 0 (K-major)
//     [17:23) n_dim          = N >> 3
//     [24:29) m_dim          = M >> 4
//     [30:32) max_shift      = 0
//
//   SmemDescriptor[64]:
//     [ 0:14) start_address  = (smem_byte_addr) >> 4
//     [16:30) leading_byte_offset (LBO)
//     [32:46) stride_byte_offset  (SBO)
//     [46:48) version        = 0
//     [49:52) base_offset    = 0
//     [52:53) lbo_mode       = 0
//     [61:64) layout_type    = 0 (SWIZZLE_NONE) | 2 (128B) | 4 (64B) | 6 (32B)
//
// For K-major operand stored in SMEM as (outer=M/N, inner=K) row-major BF16:
//   SBO = bytes between consecutive 8-row groups along the outer dim, /16.
//         = (8 rows * inner_K_bytes) / 16
//   LBO = bytes per "core matrix" stride along K, /16. With BK=16 (32 bytes)
//         and 8-col core matrices, there's only one 8-col stride per row so
//         LBO doesn't actually matter for a single MMA; we set it to (8*2)/16=1
//         to follow the canonical pattern.

#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>
#include <cuda/barrier>

#ifndef LB_MIN_BLOCKS
#define LB_MIN_BLOCKS 1
#endif

// ── Descriptor builders ──────────────────────────────────────────────────────

// Mirrors gau-nernst/learn-cuda matmul_v1.cu make_desc():
//   LBO = height * 16 bytes  (height = BM for A, BN for B)
//   SBO = 8 * 16 = 128 bytes
//   bit 46 = 1 (version)
//   layout_type = 0 (no swizzle)
__device__ __forceinline__ uint64_t make_smem_desc_kmajor_noswz(
    const void* smem_ptr, int height
) {
    uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    auto enc = [](uint64_t bytes) -> uint64_t { return (bytes >> 4) & ((1ULL << 14) - 1); };
    // For 3D-tiled layout [K_TILES][height][K_INNER=8]:
    //   LBO_bytes = height * 16     (= bytes per k_tile slab)
    //   SBO_bytes = 8 * 16 = 128    (= bytes per 8-row M-block within a k_tile)
    uint64_t desc = enc((uint64_t)saddr)
                  | (enc((uint64_t)height * 16) << 16)
                  | (enc((uint64_t)8 * 16) << 32);
    return desc;
}

__device__ __forceinline__ uint32_t make_idesc_bf16(int M, int N) {
    uint32_t d = 0;
    // c_format = F32 (1) at [4:6)
    d |= (1u << 4);
    // a_format = BF16 (1) at [7:10)
    d |= (1u << 7);
    // b_format = BF16 (1) at [10:13)
    d |= (1u << 10);
    // a_major = K (0), b_major = K (0)   — leave zero
    // n_dim at [17:23) = N >> 3
    d |= (((uint32_t)(N >> 3) & 0x3F) << 17);
    // m_dim at [24:29) = M >> 4
    d |= (((uint32_t)(M >> 4) & 0x1F) << 24);
    return d;
}

// ── tcgen05 PTX wrappers ─────────────────────────────────────────────────────

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
__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// tcgen05.ld.sync.aligned.32x32b.x8.b32 — Layout D. 32 lanes = 32 rows.
// Per call: lane L gets row warp_id*32 + L, regs 0..7 = cols base..base+7.
// Loop over col base in {0, 8, 16, ..., BN-8} to cover N=128.
__device__ __forceinline__ void tcgen05_ld_32x32b_x8(
    uint32_t taddr, float* out  // out[8]
) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
        : "r"(taddr));
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

// Unused now — kept for reference.
__device__ __forceinline__ void tcgen05_ld_32x32b_x128(
    uint32_t taddr, uint32_t* out
) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x128.b32 "
        "{ %0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "  %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "  %32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "  %48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,"
        "  %64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,"
        "  %80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,"
        "  %96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,"
        "  %112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127 },"
        "[%128];"
        : "=r"(out[0]),"=r"(out[1]),"=r"(out[2]),"=r"(out[3]),"=r"(out[4]),"=r"(out[5]),"=r"(out[6]),"=r"(out[7]),
          "=r"(out[8]),"=r"(out[9]),"=r"(out[10]),"=r"(out[11]),"=r"(out[12]),"=r"(out[13]),"=r"(out[14]),"=r"(out[15]),
          "=r"(out[16]),"=r"(out[17]),"=r"(out[18]),"=r"(out[19]),"=r"(out[20]),"=r"(out[21]),"=r"(out[22]),"=r"(out[23]),
          "=r"(out[24]),"=r"(out[25]),"=r"(out[26]),"=r"(out[27]),"=r"(out[28]),"=r"(out[29]),"=r"(out[30]),"=r"(out[31]),
          "=r"(out[32]),"=r"(out[33]),"=r"(out[34]),"=r"(out[35]),"=r"(out[36]),"=r"(out[37]),"=r"(out[38]),"=r"(out[39]),
          "=r"(out[40]),"=r"(out[41]),"=r"(out[42]),"=r"(out[43]),"=r"(out[44]),"=r"(out[45]),"=r"(out[46]),"=r"(out[47]),
          "=r"(out[48]),"=r"(out[49]),"=r"(out[50]),"=r"(out[51]),"=r"(out[52]),"=r"(out[53]),"=r"(out[54]),"=r"(out[55]),
          "=r"(out[56]),"=r"(out[57]),"=r"(out[58]),"=r"(out[59]),"=r"(out[60]),"=r"(out[61]),"=r"(out[62]),"=r"(out[63]),
          "=r"(out[64]),"=r"(out[65]),"=r"(out[66]),"=r"(out[67]),"=r"(out[68]),"=r"(out[69]),"=r"(out[70]),"=r"(out[71]),
          "=r"(out[72]),"=r"(out[73]),"=r"(out[74]),"=r"(out[75]),"=r"(out[76]),"=r"(out[77]),"=r"(out[78]),"=r"(out[79]),
          "=r"(out[80]),"=r"(out[81]),"=r"(out[82]),"=r"(out[83]),"=r"(out[84]),"=r"(out[85]),"=r"(out[86]),"=r"(out[87]),
          "=r"(out[88]),"=r"(out[89]),"=r"(out[90]),"=r"(out[91]),"=r"(out[92]),"=r"(out[93]),"=r"(out[94]),"=r"(out[95]),
          "=r"(out[96]),"=r"(out[97]),"=r"(out[98]),"=r"(out[99]),"=r"(out[100]),"=r"(out[101]),"=r"(out[102]),"=r"(out[103]),
          "=r"(out[104]),"=r"(out[105]),"=r"(out[106]),"=r"(out[107]),"=r"(out[108]),"=r"(out[109]),"=r"(out[110]),"=r"(out[111]),
          "=r"(out[112]),"=r"(out[113]),"=r"(out[114]),"=r"(out[115]),"=r"(out[116]),"=r"(out[117]),"=r"(out[118]),"=r"(out[119]),
          "=r"(out[120]),"=r"(out[121]),"=r"(out[122]),"=r"(out[123]),"=r"(out[124]),"=r"(out[125]),"=r"(out[126]),"=r"(out[127])
        : "r"(taddr) : "memory");
}

// ── Kernel ───────────────────────────────────────────────────────────────────

template <int BM, int BN, int BK, int NUM_WARPS>
__device__ __forceinline__ void b3_tc05_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    static_assert(BK == 16, "v1 only supports BK=16");
    static_assert(BM == 128 && BN == 128, "v1 fixed at 128x128 tile");
    constexpr int THREADS = NUM_WARPS * 32;

    // SMEM layout (3D-tiled to match tcgen05 descriptor conventions, mirroring
    // gau-nernst/learn-cuda matmul_v1.cu):
    //   A[K_TILES][BM][K_INNER=8]  bf16 — K split into 8-elem inner blocks
    //   B[K_TILES][BN][K_INNER=8]  bf16 — B transposed into N-major outer × K_INNER inner
    constexpr int K_INNER = 8;
    constexpr int K_TILES = BK / K_INNER;   // = 2 for BK=16
    extern __shared__ uint8_t smem_raw[];
    auto smem_A = reinterpret_cast<__nv_bfloat16 (*)[BM][K_INNER]>(smem_raw);
    auto smem_B = reinterpret_cast<__nv_bfloat16 (*)[BN][K_INNER]>(smem_raw + K_TILES * BM * K_INNER * 2);
    auto tmem_holder = reinterpret_cast<uint32_t*>(smem_raw + K_TILES * (BM + BN) * K_INNER * 2);
    auto mbar = reinterpret_cast<uint64_t*>(smem_raw + K_TILES * (BM + BN) * K_INNER * 2 + 16);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // ── alloc TMEM (warp-uniform). Layout-D: 1 D col = 1 TMEM col → BN cols.
    if (warp_id == 0) {
        tcgen05_alloc(tmem_holder, BN);
        tcgen05_relinquish();
    }
    if (tid == 0) {
        uint32_t mb_smem = (uint32_t)__cvta_generic_to_shared(mbar);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     :: "r"(mb_smem) : "memory");
    }
    __syncthreads();
    const uint32_t tmem_d = tmem_holder[0];

    // ── K-loop ────────────────────────────────────────────────────────────────
    const int num_tiles = K / BK;

    // Each iter:
    //   1. cp.async A tile + B tile (transposed)
    //   2. wait + sync
    //   3. tid 0 fires tcgen05.mma (accumulate after first iter)
    //   4. tid 0 commits → mbar arrives
    //   5. mbar.wait so SMEM is free to be overwritten next iter

    // ── Compute mbarrier phase tracking ───
    uint32_t phase = 0;

    for (int kt = 0; kt < num_tiles; kt++) {
        const int k0 = kt * BK;

        // ── Load A into smem_A[k_tile][m][k_inner] via cp.async (16B chunks) ──
        // For BM=128, K_TILES=2, THREADS=128: each thread does 2 chunks (one
        // per k_tile, fixed m=tid). Each chunk = K_INNER=8 bf16 = 16 bytes.
        #pragma unroll
        for (int kt_inner = 0; kt_inner < K_TILES; kt_inner++) {
            int m = tid % BM;
            int g_idx = tid + kt_inner * THREADS;
            int kt_idx = g_idx / BM;
            if (kt_idx < K_TILES) {
                const __nv_bfloat16* gptr =
                    &A[(block_row + m) * K + k0 + kt_idx * K_INNER];
                __pipeline_memcpy_async(&smem_A[kt_idx][m][0], gptr, 16);
            }
        }

        // ── Load B (transposed) into smem_B[k_tile][n][k_inner] ──
        // For each n in [0, BN), each thread reads K_INNER bf16 from B[k0 + kt*8 + 0..7, n]
        // (strided in K, not coalesced — v1). Uses cp.async since each chunk is 16B.
        for (int n = tid; n < BN; n += THREADS) {
            #pragma unroll
            for (int kt_inner = 0; kt_inner < K_TILES; kt_inner++) {
                // GMEM source: B[(k0 + kt_inner*8 + 0..7), block_col + n]
                // — that's 8 reads at stride N. Can't do as a single cp.async.
                // Fall back to scalar loads here.
                #pragma unroll
                for (int ki = 0; ki < K_INNER; ki++) {
                    smem_B[kt_inner][n][ki] =
                        B[(k0 + kt_inner * K_INNER + ki) * N + block_col + n];
                }
            }
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // ── Issue MMA (single thread) ──
        if (tid == 0) {
            uint64_t descA = make_smem_desc_kmajor_noswz(&smem_A[0][0], BM);
            uint64_t descB = make_smem_desc_kmajor_noswz(&smem_B[0][0], BN);
            uint32_t idesc = make_idesc_bf16(BM, BN);
            tcgen05_mma(tmem_d, descA, descB, idesc, /*enable_d=*/(kt > 0));
            tcgen05_commit(mbar);
        }

        // ── Wait for MMA to complete (consume mbar phase) ──
        {
            uint32_t mb_smem = (uint32_t)__cvta_generic_to_shared(mbar);
            asm volatile(
                "{\n\t .reg .pred P;\n\t"
                "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@P bra DONE_%=;\n\t"
                "bra WAIT_%=;\n\t"
                "DONE_%=:\n\t"
                "}"
                :: "r"(mb_smem), "r"(phase) : "memory");
            phase ^= 1;
        }

        __syncthreads();
    }

    // ── Epilogue: Layout-D pattern from gau-nernst/learn-cuda matmul_v1.cu ──
    //   - Mandatory fence between tcgen05.mma and tcgen05.ld
    //   - Loop 8 cols at a time using .32x32b.x8
    //   - Lane L within warp w → D row (w*32 + L); regs 0..7 → cols base..base+7
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
                // Boundary: write element by element.
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

extern "C" __global__ __launch_bounds__(128, LB_MIN_BLOCKS)
void matmul_b3_tc05_bm128_bn128_bk16_nw4(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    b3_tc05_impl<128, 128, 16, 4>(A, B, C, M, K, N);
}
