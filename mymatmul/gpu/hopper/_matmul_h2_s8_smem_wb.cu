#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>

/*
 * h2_s8_smem_wb: h2_s8_smem_wb with single (A+B) commit group per tile.
 *
 * Motivation: s8 inherited s7_split's two-commit pattern (A and B issued as
 * separate cp.async.commit_group). The split was originally chosen for finer
 * waiting granularity, but with WAIT_SMEM(NS-2) we only ever wait on whole
 * tiles. Merging A and B into one commit group halves the number of
 * commit/wait instructions issued per K-iter.
 */

#ifndef LB_MIN_BLOCKS
#define LB_MIN_BLOCKS 1
#endif

// ── wgmma helpers ─────────────────────────────────────────────────────────────
__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
// wgmma drain helpers kept for reference; pipeline uses WAIT_MMA(n) macro instead
__device__ __forceinline__ void wgmma_wait_0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// ── GmmaDescriptors (unchanged from h2_s6) ───────────────────────────────────

template<int BK>
__device__ __forceinline__ uint64_t make_wgmma_a_desc(uint32_t smem_addr, int kk) {
    constexpr uint64_t layout = (BK == 64) ? 1ULL : (BK == 32) ? 2ULL : 3ULL;
    constexpr uint64_t sbo = (uint64_t)BK;
    uint64_t start = ((uint64_t)(smem_addr >> 4) & 0x3FFFULL) + (uint64_t)(kk * 2);
    return start | (sbo << 32) | (layout << 62);
}

template<int BN, int BK>
__device__ __forceinline__ uint64_t make_wgmma_b_desc(uint32_t smem_addr) {
    constexpr uint64_t LAYOUT_B128 = 1ULL << 62;
    constexpr int n_atoms = BN / 64;
    constexpr uint64_t lbo = (n_atoms <= 1) ? 0ULL : (uint64_t)(8 * BK);
    constexpr uint64_t sbo = 64ULL;
    uint64_t start = (uint64_t)(smem_addr >> 4) & 0x3FFFULL;
    return start | (lbo << 16) | (sbo << 32) | LAYOUT_B128;
}

// ── wgmma SS wrappers (identical to h2_s6) ────────────────────────────────────

__device__ __forceinline__
void wgmma_ss_m64n64k16(float d[32], uint64_t a, uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32,%33,1,1,1,0,1;\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
         "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
         "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
        :"l"(a),"l"(b));
}

__device__ __forceinline__
void wgmma_ss_m64n128k16(float d[64], uint64_t a, uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "%64,%65,1,1,1,0,1;\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
         "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
         "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
         "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
         "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
         "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
         "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
        :"l"(a),"l"(b));
}

__device__ __forceinline__
void wgmma_ss_m64n256k16(float d[128], uint64_t a, uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,"
        "%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,"
        "%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,"
        "%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,"
        "%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127},"
        "%128,%129,1,1,1,0,1;\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
         "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
         "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
         "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
         "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
         "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
         "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),
         "+f"(d[64]),"+f"(d[65]),"+f"(d[66]),"+f"(d[67]),"+f"(d[68]),"+f"(d[69]),"+f"(d[70]),"+f"(d[71]),
         "+f"(d[72]),"+f"(d[73]),"+f"(d[74]),"+f"(d[75]),"+f"(d[76]),"+f"(d[77]),"+f"(d[78]),"+f"(d[79]),
         "+f"(d[80]),"+f"(d[81]),"+f"(d[82]),"+f"(d[83]),"+f"(d[84]),"+f"(d[85]),"+f"(d[86]),"+f"(d[87]),
         "+f"(d[88]),"+f"(d[89]),"+f"(d[90]),"+f"(d[91]),"+f"(d[92]),"+f"(d[93]),"+f"(d[94]),"+f"(d[95]),
         "+f"(d[96]),"+f"(d[97]),"+f"(d[98]),"+f"(d[99]),"+f"(d[100]),"+f"(d[101]),"+f"(d[102]),"+f"(d[103]),
         "+f"(d[104]),"+f"(d[105]),"+f"(d[106]),"+f"(d[107]),"+f"(d[108]),"+f"(d[109]),"+f"(d[110]),"+f"(d[111]),
         "+f"(d[112]),"+f"(d[113]),"+f"(d[114]),"+f"(d[115]),"+f"(d[116]),"+f"(d[117]),"+f"(d[118]),"+f"(d[119]),
         "+f"(d[120]),"+f"(d[121]),"+f"(d[122]),"+f"(d[123]),"+f"(d[124]),"+f"(d[125]),"+f"(d[126]),"+f"(d[127])
        :"l"(a),"l"(b));
}

template<int BN>
__device__ __forceinline__
void wgmma_ss_call(float* acc, uint64_t desc_a, uint64_t desc_b) {
    if constexpr      (BN ==  64) wgmma_ss_m64n64k16 (acc, desc_a, desc_b);
    else if constexpr (BN == 128) wgmma_ss_m64n128k16(acc, desc_a, desc_b);
    else if constexpr (BN == 256) wgmma_ss_m64n256k16(acc, desc_a, desc_b);
}

// ── Kernel ────────────────────────────────────────────────────────────────────

template<int BM, int BN, int BK, int NUM_WG, int NUM_STAGES>
__device__ __forceinline__ void h2_s8_smem_wb_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    static_assert(BM % (NUM_WG * 64) == 0, "BM must be multiple of NUM_WG*64");
    static_assert(BN % 64 == 0, "BN must be multiple of 64");

    constexpr int NS         = NUM_STAGES;
    constexpr int M_ITERS    = BM / (NUM_WG * 64);
    constexpr int M_PER_WG   = BM / NUM_WG;
    constexpr int D          = BN / 2;
    constexpr int N_SUBTILES = BN / 64;

    constexpr int A_SWZ   = BK / 8;
    constexpr int A_SHIFT = 64 / BK;

    constexpr int THREADS  = NUM_WG * 128;
    constexpr int A_ELEM   = (BM * BK / THREADS >= 8) ? 8 : 4;
    constexpr int A_GROUPS = BM * BK / A_ELEM / THREADS;
    constexpr int B_ELEM   = (BK * BN / THREADS >= 8) ? 8 : 4;
    constexpr int B_GROUPS = BK * BN / B_ELEM / THREADS;

    constexpr int K_STEP_BYTES = 16 * 64 * 2;  // B per-kk advance = 2048 bytes

    const int tid        = threadIdx.x;
    const int wg_id      = tid / 128;
    const int local_warp = (tid % 128) / 32;
    const int lane       = tid % 32;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // SMEM: A[NS][BM][BK], B[NS][N_SUBTILES][BK][64]
    extern __shared__ char smem_raw[];
    constexpr int A_BYTES = NS * BM * BK * 2;

    auto A_sh = reinterpret_cast<__nv_bfloat16 (*)[BM][BK]>(smem_raw);
    auto B_sh = reinterpret_cast<__nv_bfloat16 (*)[N_SUBTILES][BK][64]>(smem_raw + A_BYTES);

    float acc[M_ITERS][D] = {};

    const int num_tiles = K / BK;

    // ── Per-thread running pointers (Triton-style) ────────────────────────────
    //   A_curr[i] / B_curr[i] hold the current global address for the i-th
    //   16-byte chunk this thread loads. Initialized to the K=0 position
    //   below; each LOAD_TILE advances them by BK (A) / BK*N (B) elements.
    const __nv_bfloat16* A_curr[A_GROUPS];
    const __nv_bfloat16* B_curr[B_GROUPS];

    // SMEM destination byte-offsets within a single stage, folded per (tid, _i).
    // These collapse the per-cp.async 2-add (A) / 3-add (B) chain to 1 add.
    int A_sh_off[A_GROUPS];
    int B_sh_off[B_GROUPS];

    #pragma unroll
    for (int _i = 0; _i < A_GROUPS; _i++) {
        const int _g  = tid + _i * THREADS;
        const int _r  = (_g * A_ELEM) / BK;
        const int _c  = (_g * A_ELEM) % BK;
        const int _sc = ((_c / 8) ^ ((_r / A_SHIFT) % A_SWZ)) * 8 + (_c % 8);
        A_curr[_i] = &A[(block_row + _r) * K + _c];   // k0 = 0
        A_sh_off[_i] = (_r * BK + _sc) * (int)sizeof(__nv_bfloat16);
    }
    #pragma unroll
    for (int _i = 0; _i < B_GROUPS; _i++) {
        const int _g    = tid + _i * THREADS;
        const int _flat = _g * B_ELEM;
        const int _st   = _flat / (BK * 64);
        const int _kr   = (_flat % (BK * 64)) / 64;
        const int _sc0  = (_flat % (BK * 64)) % 64;
        const int _sc   = ((_sc0 / 8) ^ (_kr % 8)) * 8 + (_sc0 % 8);
        B_curr[_i] = &B[_kr * N + block_col + _st * 64 + _sc0];   // k0 = 0
        B_sh_off[_i] = (_st * (BK * 64) + _kr * 64 + _sc) * (int)sizeof(__nv_bfloat16);
    }

    // ── Pipeline macros ───────────────────────────────────────────────────────
    //
    // Four abstract operations that define the pipeline regardless of backend:
    //
    //   LOAD_TILE(slot, k0) — issue async load of A+B tile from DRAM into SMEM slot
    //   WAIT_SMEM(n)        — stall until SMEM data is ready; keep n loads in flight
    //   COMPUTE_TILE(slot)  — fence + kick off wgmma for slot + commit (no drain)
    //   WAIT_MMA(n)         — stall until all but n wgmma groups are done
    //
    // Pipeline loop (same structure will be reused for TMA backend in h2_s8_smem_wb):
    //
    //   Prologue: for s=0..NS-2: LOAD_TILE(s, s*BK)
    //
    //   for k=0..num_tiles-(NS-1):
    //     WAIT_SMEM(NS-2)       ← tile k is now in SMEM
    //     __syncthreads()       ← all threads see it (WAIT_SMEM is per-thread)
    //     COMPUTE_TILE(cur)     ← kick off wgmma group k
    //     WAIT_MMA(1)           ← groups 0..k-1 done; group k still in TC pipeline
    //     __syncthreads()       ← all warpgroups agree before cooperative LOAD_TILE
    //     LOAD_TILE(nxt, ...)   ← safe: slot nxt=(k-1)%NS freed by WAIT_MMA(1)
    //
    //   for d=NS-2..0:          ← drain last NS-1 tiles (no new loads)
    //     WAIT_SMEM(d)
    //     __syncthreads()
    //     COMPUTE_TILE(slot)
    //     WAIT_MMA(1)
    //
    //   WAIT_MMA(0)             ← drain final group before epilogue

    // 1c: single commit per LOAD_TILE (A and B in one group). Halves the
    //   commit instruction count vs the split variant; WAIT_SMEM counts in
    //   whole stages (1 commit/iter).
#define LOAD_TILE(slot_)                                                            \
    do {                                                                            \
        char* const _A_base = (char*)&A_sh[(slot_)][0][0];                          \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < A_GROUPS; _i++) {                                    \
            __pipeline_memcpy_async(                                                \
                _A_base + A_sh_off[_i],                                             \
                A_curr[_i],                                                         \
                A_ELEM * (int)sizeof(__nv_bfloat16));                               \
            A_curr[_i] += BK;                                                       \
        }                                                                           \
        char* const _B_base = (char*)&B_sh[(slot_)][0][0][0];                       \
        _Pragma("unroll")                                                           \
        for (int _i = 0; _i < B_GROUPS; _i++) {                                    \
            __pipeline_memcpy_async(                                                \
                _B_base + B_sh_off[_i],                                             \
                B_curr[_i],                                                         \
                B_ELEM * (int)sizeof(__nv_bfloat16));                               \
            B_curr[_i] += BK * N;                                                   \
        }                                                                           \
        __pipeline_commit();                                                         \
    } while (0)

    // 1c: 1 commit/iter → WAIT_SMEM counts in whole stages.
#define WAIT_SMEM(n_) __pipeline_wait_prior(n_)

    // COMPUTE_TILE: wgmma.fence + wgmma calls + commit (no drain)
#define COMPUTE_TILE(slot_)                                                         \
    do {                                                                            \
        wgmma_fence();                                                              \
        _Pragma("unroll")                                                           \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                 \
            const uint32_t _bb = (uint32_t)__cvta_generic_to_shared(               \
                                     &B_sh[(slot_)][0][0][0]);                     \
            const uint64_t _db = make_wgmma_b_desc<BN, BK>(                       \
                                     (uint32_t)(_bb + _kk * K_STEP_BYTES));        \
            _Pragma("unroll")                                                       \
            for (int _m = 0; _m < M_ITERS; _m++) {                                 \
                const int _mrow = wg_id * M_PER_WG + _m * 64;                     \
                const uint32_t _aa = (uint32_t)__cvta_generic_to_shared(           \
                                         &A_sh[(slot_)][_mrow][0]);                \
                const uint64_t _da = make_wgmma_a_desc<BK>(_aa, _kk);             \
                wgmma_ss_call<BN>((float*)acc[_m], _da, _db);                      \
            }                                                                       \
        }                                                                           \
        wgmma_commit();                                                             \
    } while (0)

    // WAIT_MMA(n): wgmma.wait_group n (compile-time constant via stringification)
#define WAIT_MMA(n_) \
    asm volatile("wgmma.wait_group.sync.aligned " #n_ ";\n" ::: "memory")

    // ── Prologue ──────────────────────────────────────────────────────────────
    #pragma unroll
    for (int s = 0; s < NS - 1; s++) {
        LOAD_TILE(s);   // running pointers carry the K position
    }

    // ── Main loop ─────────────────────────────────────────────────────────────
    for (int k = 0; k < num_tiles - (NS - 1); k++) {
        const int cur = k % NS;
        const int nxt = (k + NS - 1) % NS;

        WAIT_SMEM(NS - 2);
        __syncthreads();
        COMPUTE_TILE(cur);
        WAIT_MMA(1);
        __syncthreads();
        LOAD_TILE(nxt);
    }

    // ── Drain ─────────────────────────────────────────────────────────────────
    #pragma unroll
    for (int d = NS - 2; d >= 0; d--) {
        const int slot = (num_tiles - d - 1) % NS;
        WAIT_SMEM(d);
        __syncthreads();
        COMPUTE_TILE(slot);
        WAIT_MMA(1);
    }

    WAIT_MMA(0);

#undef LOAD_TILE
#undef WAIT_SMEM
#undef COMPUTE_TILE
#undef WAIT_MMA

    // ── Epilogue: SMEM-staged 16-byte coalesced stores ─────────────────────
    // (Ported from h2_s7_smem_epi.) Per-thread writes acc[] to SMEM at the
    // wgmma-native position, sync, then THREADS cooperatively stream the
    // BM*BN tile to global C in 16-byte vector stores. 4× fewer global
    // transactions than the direct-register epilogue. Reuses the A/B SMEM
    // region which is no longer needed after WAIT_MMA(0).
    __syncthreads();
    constexpr int BN_PAD = BN + 8;  // +8 BF16 to break power-of-2 bank stride
    auto C_sh = reinterpret_cast<__nv_bfloat16 (*)[BN_PAD]>(smem_raw);

    {
        const int base_col = (lane % 4) * 2;
        const int base_row = lane / 4;
        #pragma unroll
        for (int m = 0; m < M_ITERS; m++) {
            const int row0 = wg_id * M_PER_WG + m * 64 + local_warp * 16 + base_row;
            const int row8 = row0 + 8;
            #pragma unroll
            for (int j = 0; j < BN / 8; j++) {
                const int col = j * 8 + base_col;
                *reinterpret_cast<__nv_bfloat162*>(&C_sh[row0][col]) =
                    __floats2bfloat162_rn(acc[m][j*4+0], acc[m][j*4+1]);
                *reinterpret_cast<__nv_bfloat162*>(&C_sh[row8][col]) =
                    __floats2bfloat162_rn(acc[m][j*4+2], acc[m][j*4+3]);
            }
        }
    }
    __syncthreads();

    constexpr int BF_PER_STORE      = 8;
    constexpr int STORES_PER_THREAD = (BM * BN) / (BF_PER_STORE * THREADS);
    static_assert(STORES_PER_THREAD > 0 && (BM * BN) % (BF_PER_STORE * THREADS) == 0,
                  "BM*BN must be a multiple of BF_PER_STORE*THREADS");

    #pragma unroll
    for (int s = 0; s < STORES_PER_THREAD; s++) {
        const int flat = tid + s * THREADS;
        const int local_bf16 = flat * BF_PER_STORE;
        const int row = local_bf16 / BN;
        const int col = local_bf16 % BN;
        const int gr  = block_row + row;
        const int gc  = block_col + col;
        if (gr < M && gc < N) {
            uint4 data = *reinterpret_cast<const uint4*>(&C_sh[row][col]);
            *reinterpret_cast<uint4*>(&C[gr * N + gc]) = data;
        }
    }
}

// ── Kernel entry points ───────────────────────────────────────────────────────

#define MAKE_LAUNCHER(BM_, BN_, BK_, NG_, NS_)                                   \
extern "C" __global__ __launch_bounds__(NG_ * 128, LB_MIN_BLOCKS)               \
void matmul_h2_s8_smem_wb_bm##BM_##_bn##BN_##_bk##BK_##_wg##NG_##_ns##NS_(              \
    const __nv_bfloat16* __restrict__ A,                                         \
    const __nv_bfloat16* __restrict__ B,                                         \
    __nv_bfloat16* __restrict__ C,                                               \
    int M, int K, int N)                                                         \
{                                                                                \
    h2_s8_smem_wb_impl<BM_, BN_, BK_, NG_, NS_>(A, B, C, M, K, N);                      \
}

#define MAKE3(BM_, BN_, BK_, NG_) \
    MAKE_LAUNCHER(BM_, BN_, BK_, NG_, 2) \
    MAKE_LAUNCHER(BM_, BN_, BK_, NG_, 3) \
    MAKE_LAUNCHER(BM_, BN_, BK_, NG_, 4) \
    MAKE_LAUNCHER(BM_, BN_, BK_, NG_, 5)

// NW=1 warpgroup
MAKE3( 64,  64, 16, 1) MAKE3( 64,  64, 32, 1) MAKE3( 64,  64, 64, 1)
MAKE3( 64, 128, 16, 1) MAKE3( 64, 128, 32, 1) MAKE3( 64, 128, 64, 1)
MAKE3( 64, 256, 16, 1) MAKE3( 64, 256, 32, 1) MAKE3( 64, 256, 64, 1)
MAKE3(128,  64, 16, 1) MAKE3(128,  64, 32, 1) MAKE3(128,  64, 64, 1)
MAKE3(128, 128, 16, 1) MAKE3(128, 128, 32, 1) MAKE3(128, 128, 64, 1)
MAKE3(128, 256, 16, 1) MAKE3(128, 256, 32, 1) MAKE3(128, 256, 64, 1)
MAKE3(256,  64, 16, 1) MAKE3(256,  64, 32, 1) MAKE3(256,  64, 64, 1)
MAKE3(256, 128, 16, 1) MAKE3(256, 128, 32, 1) MAKE3(256, 128, 64, 1)

// NW=2 warpgroups
MAKE3(128,  64, 16, 2) MAKE3(128,  64, 32, 2) MAKE3(128,  64, 64, 2)
MAKE3(128, 128, 16, 2) MAKE3(128, 128, 32, 2) MAKE3(128, 128, 64, 2)
MAKE3(128, 256, 16, 2) MAKE3(128, 256, 32, 2) MAKE3(128, 256, 64, 2)
MAKE3(256,  64, 16, 2) MAKE3(256,  64, 32, 2) MAKE3(256,  64, 64, 2)
MAKE3(256, 128, 16, 2) MAKE3(256, 128, 32, 2) MAKE3(256, 128, 64, 2)
MAKE3(256, 256, 16, 2) MAKE3(256, 256, 32, 2) MAKE3(256, 256, 64, 2)

#undef MAKE3
#undef MAKE_LAUNCHER
