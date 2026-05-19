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

__device__ __forceinline__ uint64_t make_smem_desc_kmajor_noswz(
    const void* smem_ptr, int inner_K_bytes
) {
    uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t sbo = (uint64_t)(8 * inner_K_bytes) >> 4;   // 8 rows along outer dim
    uint64_t lbo = (uint64_t)(8 * 2) >> 4;                // 8 bf16 cols
    uint64_t desc = 0;
    desc |= ((uint64_t)(saddr >> 4) & ((1ULL << 14) - 1)) << 0;
    desc |= (lbo & ((1ULL << 14) - 1)) << 16;
    desc |= (sbo & ((1ULL << 14) - 1)) << 32;
    // version=0, base_offset=0, lbo_mode=0, layout_type=0 (SWIZZLE_NONE)
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
    // m_dim at [24:29) = M >> 4  (CUTLASS source)
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

// tcgen05.ld.sync.aligned.16x256b.x16.b32 — 64 regs per lane.
__device__ __forceinline__ void tcgen05_ld_16x256b_x16(
    uint32_t taddr, uint32_t out[64]
) {
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x16.b32 "
        "{ %0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "  %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "  %32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "  %48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63 },"
        "[%64];"
        : "=r"(out[0]),"=r"(out[1]),"=r"(out[2]),"=r"(out[3]),"=r"(out[4]),"=r"(out[5]),"=r"(out[6]),"=r"(out[7]),
          "=r"(out[8]),"=r"(out[9]),"=r"(out[10]),"=r"(out[11]),"=r"(out[12]),"=r"(out[13]),"=r"(out[14]),"=r"(out[15]),
          "=r"(out[16]),"=r"(out[17]),"=r"(out[18]),"=r"(out[19]),"=r"(out[20]),"=r"(out[21]),"=r"(out[22]),"=r"(out[23]),
          "=r"(out[24]),"=r"(out[25]),"=r"(out[26]),"=r"(out[27]),"=r"(out[28]),"=r"(out[29]),"=r"(out[30]),"=r"(out[31]),
          "=r"(out[32]),"=r"(out[33]),"=r"(out[34]),"=r"(out[35]),"=r"(out[36]),"=r"(out[37]),"=r"(out[38]),"=r"(out[39]),
          "=r"(out[40]),"=r"(out[41]),"=r"(out[42]),"=r"(out[43]),"=r"(out[44]),"=r"(out[45]),"=r"(out[46]),"=r"(out[47]),
          "=r"(out[48]),"=r"(out[49]),"=r"(out[50]),"=r"(out[51]),"=r"(out[52]),"=r"(out[53]),"=r"(out[54]),"=r"(out[55]),
          "=r"(out[56]),"=r"(out[57]),"=r"(out[58]),"=r"(out[59]),"=r"(out[60]),"=r"(out[61]),"=r"(out[62]),"=r"(out[63])
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
    static_assert(BN == 128, "");
    constexpr int THREADS = NUM_WARPS * 32;

    // SMEM layout:
    //   A[BM][BK] k-major bf16        =  128 *  16 * 2 = 4096 B
    //   B[BN][BK] k-major bf16 (B^T)  =  128 *  16 * 2 = 4096 B
    //   tmem_addr_holder (1 x u32)    =       4 B
    //   mbar (1 x u64)                =       8 B
    extern __shared__ uint8_t smem_raw[];
    auto smem_A = reinterpret_cast<__nv_bfloat16 (*)[BK]>(smem_raw);
    auto smem_B = reinterpret_cast<__nv_bfloat16 (*)[BK]>(smem_raw + BM * BK * 2);
    auto tmem_holder = reinterpret_cast<uint32_t*>(smem_raw + (BM + BN) * BK * 2);
    auto mbar = reinterpret_cast<uint64_t*>(smem_raw + (BM + BN) * BK * 2 + 16);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // ── alloc TMEM (warp-uniform: whole warp 0 must execute sync.aligned) ────
    if (warp_id == 0) {
        tcgen05_alloc(tmem_holder, 128);
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

    // For BM*BK = 2048 bf16 over 128 threads, 16 bf16/thread = 32B/thread.
    // Use cp.async 16B (8 bf16) chunks → 2 cp.asyncs per thread for A.
    constexpr int A_CHUNK_B = 16;
    constexpr int A_PER_THR = (BM * BK * 2) / (THREADS * A_CHUNK_B);  // 2

    // For B: GMEM is K x N row-major. We load into smem_B[n][k] (transposed).
    // Simpler approach for v1: each thread loads 16 elements of one B-row of
    // the GMEM tile, i.e., load BK x BN in natural layout into a scratch,
    // then transpose into smem_B. Too much work — instead, each thread loads
    // a single bf16 at a time (uncoalesced, slow but correct for v1).
    // We will fix this in v2.

    // ── Compute mbarrier phase tracking ───
    uint32_t phase = 0;

    for (int kt = 0; kt < num_tiles; kt++) {
        const int k0 = kt * BK;

        // ── Load A: 128x16 bf16, naturally k-inner row-major ──
        #pragma unroll
        for (int i = 0; i < A_PER_THR; i++) {
            int g = tid + i * THREADS;
            int r = (g * 8) / BK;      // row in tile
            int c = (g * 8) % BK;      // col in tile (bf16 idx)
            if (r < BM) {
                const __nv_bfloat16* gptr = &A[(block_row + r) * K + k0 + c];
                __nv_bfloat16* sptr = &smem_A[r][c];
                __pipeline_memcpy_async(sptr, gptr, A_CHUNK_B);
            }
        }

        // ── Load B and transpose: GMEM B[k0+kk, block_col+nn] → smem_B[nn, kk] ──
        // BK=16, BN=128 → 2048 bf16, 16 bf16/thread = 32B.
        // Each thread loads 8 bf16 from one row of GMEM (k fixed, 8 consecutive
        // n), then writes to 8 different rows of smem_B at the same col-k.
        // We can't transpose with cp.async — fall back to a regular load.
        // For v1 we use a non-async load: read from gmem, write to smem.
        // Simpler: load NATURALLY (B[k][n]) into a second SMEM buffer, then
        // manually transpose into smem_B[n][k]. But this needs more SMEM.
        // For absolute simplicity: each thread does a small explicit transpose.
        //
        // We use 16 threads per K-row (BK=16): thread t in [0,16) loads from
        // GMEM B[k0+t, block_col + col_group*16 .. + col_group*16 + 15] for
        // various col_groups. Actually that's K-strided too.
        //
        // The cleanest pattern: 128 threads, each owns one (n, k) pair where
        // n ∈ [0, 128), k ∈ [0, 16). But that's 2048 pairs, > 128 threads.
        // So each thread does 16 pairs (a row of B^T = 16 contiguous k vals
        // for one n). That's contiguous on the SMEM side but strided on GMEM
        // (16 reads at stride N). Still works.
        // One thread per N (BN=128 == THREADS): each loads BK consecutive K
        // values from GMEM at a fixed N, writes them as one contiguous row of
        // smem_B[n]. GMEM reads are stride-N (not coalesced) — v1 only.
        {
            int n = tid;
            if (n < BN) {
                #pragma unroll
                for (int kk = 0; kk < BK; kk++) {
                    smem_B[n][kk] = B[(k0 + kk) * N + block_col + n];
                }
            }
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // ── Issue MMA (single thread) ──
        if (tid == 0) {
            uint64_t descA = make_smem_desc_kmajor_noswz(&smem_A[0][0], BK * 2);
            uint64_t descB = make_smem_desc_kmajor_noswz(&smem_B[0][0], BK * 2);
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

    // ── Epilogue: each warp reads 32 rows × 128 cols from TMEM, writes GMEM ──
    // tcgen05.ld 32x32b.x32 loads 32 lanes × 32 b32 regs.
    // Warp w handles rows [w*32 : w*32+32].
    // Lane semantics for 32x32b: lane L gets row L, all 32 columns... actually
    // 32x32b means "32 rows × 32 bits per col", and the .x32 multiplier means
    // 32 columns. So output is: 32 regs per lane, each lane covers one row,
    // each reg one column. We need 4 such ld calls to cover N=128.

    // Just 2 calls per warp (cols 0): row 0 and row 16.
    constexpr int CALLS_PER_WARP = 4;
    uint32_t out[CALLS_PER_WARP][64];
    const uint32_t warp_base = tmem_d + (warp_id * 32) * (1 << 16);
    tcgen05_ld_16x256b_x16(warp_base + (0 << 16),   out[0]);
    tcgen05_ld_16x256b_x16(warp_base + (16 << 16),  out[1]);
    // calls 2, 3 zeroed
    for (int j = 0; j < 64; j++) { out[2][j] = 0; out[3][j] = 0; }
    tcgen05_wait_ld();

    // Dump: [4 warps][32 lanes][4 calls][64 regs] = 32768 fp32 = 128KB. NB: caller
    // must allocate at least this much.
    float* dbg = reinterpret_cast<float*>(C);
    int base = ((warp_id * 32 + lane) * CALLS_PER_WARP) * 64;
    #pragma unroll
    for (int rc = 0; rc < CALLS_PER_WARP; rc++) {
        #pragma unroll
        for (int j = 0; j < 64; j++) {
            dbg[base + rc*64 + j] = __uint_as_float(out[rc][j]);
        }
    }

    __syncthreads();
    if (warp_id == 0) {
        tcgen05_dealloc(tmem_d, 128);
    }
}

extern "C" __global__ __launch_bounds__(128, LB_MIN_BLOCKS)
void tmem_probe_bm128_bn128_bk16_nw4(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    // M=64 BN=128 BK=16 NW=4: each warp reads 16 of the 64 rows.
    b3_tc05_impl<64, 128, 16, 4>(A, B, C, M, K, N);
}
