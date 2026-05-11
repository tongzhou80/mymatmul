#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

/*
 * H2 Stage 2: TMA + mbarrier (same as S1) + wgmma replaces ldmatrix-B + mma.sync.
 *
 * Key changes vs H2-S1:
 *   B TMA swizzle : NONE → 128B  (hardware writes B in bank-conflict-free layout)
 *   B compute     : ldmatrix_x2_trans + mma.sync → wgmma.mma_async (reads B direct from SMEM)
 *   A compute     : ldmatrix_x4 unchanged (A is still in registers for wgmma)
 *   Thread block  : always 128 threads = 1 warpgroup (BM = 64 fixed)
 *   Accumulators  : float acc[BN/2] per thread  (vs float acc[WM][WN][4] per warp in tc5)
 *
 * wgmma m64nBNk16 with transB=1 (N-major B, i.e. B stored [BK][BN] row-major):
 *   A : 4 uint32 registers per thread (ldmatrix_x4 across warpgroup, same loading as tc5)
 *   B : uint64 SMEM descriptor — hardware reads directly from 128B-swizzled SMEM
 *   D : BN/2 float32 registers per thread
 *
 * Compiled with -arch=sm_90a.
 */

#ifndef LB_MIN_BLOCKS
#define LB_MIN_BLOCKS 1
#endif

// ── Reuse TMA / mbarrier helpers from S1 ─────────────────────────────────────

struct alignas(64) TmaDesc { uint64_t opaque[16]; };

__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(mbar);
    uint64_t state;
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;\n"
                 : "=l"(state) : "r"(addr), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t phase) {
    uint32_t done = 0, addr = (uint32_t)__cvta_generic_to_shared(mbar);
    while (!done) {
        asm volatile("{\n.reg .pred P;\n"
                     "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2;\n"
                     "selp.u32 %0, 1, 0, P;\n}\n"
                     : "=r"(done) : "r"(addr), "r"(phase) : "memory");
    }
}

__device__ __forceinline__ void tma_load_2d(
    const TmaDesc* desc, void* smem_ptr, uint64_t* mbar, int32_t coord0, int32_t coord1
) {
    uint32_t dst  = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
        : : "r"(dst), "l"((unsigned long long)desc), "r"(mbar_addr),
            "r"(coord0), "r"(coord1) : "memory");
}

// ── ldmatrix for A (unchanged from tc5_lb) ────────────────────────────────────

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_ptr
) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(smem_ptr));
}

// ── wgmma B SMEM descriptor ───────────────────────────────────────────────────
//
// B is stored [BK][BN] row-major in SMEM, written by TMA with 128B swizzle.
// wgmma reads it as MN-major (N-contiguous), transB=1.
//
// GmmaDescriptor bit layout (from CUTLASS mma_sm90_desc.hpp):
//   [13:0]  start_address_ : smem_addr[17:4]  (>> 4, 14 bits)
//   [29:16] leading_byte_offset_ : LBO >> 4
//   [45:32] stride_byte_offset_  : SBO >> 4
//   [63:62] layout_type_ : 1 = B128
//
// LBO, SBO from tilelang MN-major formula (b_swizzle = 128B):
//   n_atoms = BN / 64  (64 bf16 per 128-byte swizzle atom)
//   if n_atoms <= 1: LBO = 0,           SBO = BN
//   else:            LBO = 8 * BK,      SBO = 64

template<int BN, int BK>
__device__ __forceinline__ uint64_t make_wgmma_b_desc(uint32_t smem_addr) {
    constexpr uint64_t LAYOUT_B128 = 1ULL << 62;
    constexpr int n_atoms = BN / 64;
    constexpr uint64_t lbo = (n_atoms <= 1) ? 0ULL        : (uint64_t)(8 * BK);
    constexpr uint64_t sbo = (n_atoms <= 1) ? (uint64_t)BN : 64ULL;
    uint64_t start = (uint64_t)(smem_addr >> 4) & 0x3FFF;
    return start | (lbo << 16) | (sbo << 32) | LAYOUT_B128;
}

// ── wgmma.mma_async wrappers ──────────────────────────────────────────────────
//
// Always accumulate (scaleD=1); caller zero-inits acc[].
// transB=1 = N-major B (our [BK][BN] row-major layout).

__device__ __forceinline__
void wgmma_m64n64k16(float d[32], uint32_t a[4], uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "{%32,%33,%34,%35},%36,1,1,1,1;\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
         "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
         "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(b));
}

__device__ __forceinline__
void wgmma_m64n128k16(float d[64], uint32_t a[4], uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "{%64,%65,%66,%67},%68,1,1,1,1;\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
         "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
         "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
         "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
         "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
         "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
         "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(b));
}

__device__ __forceinline__
void wgmma_m64n256k16(float d[128], uint32_t a[4], uint64_t b) {
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
        "{%128,%129,%130,%131},%132,1,1,1,1;\n"
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
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(b));
}

template<int BN>
__device__ __forceinline__
void wgmma_call(float* acc, uint32_t a[4], uint64_t desc_b) {
    if constexpr      (BN == 64)  wgmma_m64n64k16 (acc, a, desc_b);
    else if constexpr (BN == 128) wgmma_m64n128k16(acc, a, desc_b);
    else if constexpr (BN == 256) wgmma_m64n256k16(acc, a, desc_b);
}

// ── Kernel implementation ─────────────────────────────────────────────────────

template<int BN, int BK>
__device__ __forceinline__ void h2s2_impl(
    const TmaDesc& tma_A,
    const TmaDesc& tma_B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    constexpr int BM = 64;        // one warpgroup covers 64 output rows
    constexpr int D  = BN / 2;    // f32 accumulators per thread

    // 1D thread block: 128 threads = 1 warpgroup (4 warps)
    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // ── SMEM layout ───────────────────────────────────────────────────────────
    // A[2][BM][BK] (no swizzle, same as H2-S1)
    // B[2][BK][BN] (128B swizzle applied by TMA; wgmma reads via descriptor)
    // mbar[2]
    extern __shared__ char smem_raw[];
    constexpr int A_BYTES  = 2 * BM * BK * 2;
    constexpr int B_BYTES  = 2 * BK * BN * 2;
    constexpr int MBAR_OFF = (A_BYTES + B_BYTES + 7) & ~7;

    auto A_sh = reinterpret_cast<__nv_bfloat16 (*)[BM][BK]>(smem_raw);
    auto B_sh = reinterpret_cast<__nv_bfloat16 (*)[BK][BN]>(smem_raw + A_BYTES);
    auto mbar = reinterpret_cast<uint64_t*>(smem_raw + MBAR_OFF);

    // ── Accumulators (zero-initialised) ──────────────────────────────────────
    float acc[D] = {};

    // A swizzle: only BK=64 uses TMA 128B swizzle (period=8 rows, 16B-chunk-aligned XOR).
    // BK=16/32 use SWIZZLE_NONE (64B/32B swizzle XORs at 8B granularity which
    // breaks ldmatrix 16B chunk boundaries and corrupts data).
    constexpr int A_SWZ_PERIOD = (BK == 64) ? 8 : 0;

    const int num_tiles = K / BK;

    // ── B SMEM sub-tile helpers ───────────────────────────────────────────────
    // For 128B swizzle, boxDim[0] must be exactly 64 BF16 (= 128 bytes).
    // A BN-wide B tile is loaded as BN/64 sub-tiles of 64 columns each, packed
    // back-to-back: sub-tile i starts at B_sh[buf][0][0] + i * BK * 64 * 2 bytes.
    // The wgmma descriptor's LBO encodes the stride between consecutive sub-tiles.
    //
    // Within a pipeline stage, advancing past kk*16 K-rows inside sub-tile 0:
    //   byte offset = kk * 16 * 64 * 2  (16 K-rows × 64 BF16-wide × 2 bytes)
    // This is the formula used for the wgmma descriptor start_address per kk step.
    constexpr int SUBTILE_COL = 64;                           // BF16 per sub-tile column
    constexpr int K_STEP_BYTES = 16 * SUBTILE_COL * 2;       // = 2048: bytes per kk advance
    // (For BN==64, sub-tile stride = BK*128 = BK*K_STEP_BYTES/16 = ... but only one sub-tile)

    // ── Prologue ──────────────────────────────────────────────────────────────
    if (tid == 0) {
        mbar_init(&mbar[0], 1); mbar_init(&mbar[1], 1);
        mbar_arrive_expect_tx(&mbar[0], (BM * BK + BK * BN) * 2);
        tma_load_2d(&tma_A, &A_sh[0][0][0], &mbar[0], /*col=*/0, /*row=*/block_row);
        // Load BN/64 sub-tiles of B (each 64 columns wide, BK rows tall)
        #pragma unroll
        for (int i = 0; i < BN / SUBTILE_COL; i++) {
            void* b_dst = (char*)(&B_sh[0][0][0]) + i * BK * SUBTILE_COL * 2;
            tma_load_2d(&tma_B, b_dst, &mbar[0], block_col + i * SUBTILE_COL, /*row=*/0);
        }
    }
    __syncthreads();

    // ── Main K loop ───────────────────────────────────────────────────────────
    for (int k = 0; k < num_tiles - 1; k++) {
        const int cur = k & 1, nxt = 1 - cur;

        if (tid == 0) {
            mbar_init(&mbar[nxt], 1);
            mbar_arrive_expect_tx(&mbar[nxt], (BM * BK + BK * BN) * 2);
            tma_load_2d(&tma_A, &A_sh[nxt][0][0], &mbar[nxt], (k+1)*BK, block_row);
            #pragma unroll
            for (int i = 0; i < BN / SUBTILE_COL; i++) {
                void* b_dst = (char*)(&B_sh[nxt][0][0]) + i * BK * SUBTILE_COL * 2;
                tma_load_2d(&tma_B, b_dst, &mbar[nxt], block_col + i * SUBTILE_COL, (k+1)*BK);
            }
        }

        mbar_wait(&mbar[cur], 0);

        // Make SMEM visible to async proxy, then fence wgmma register state.
        asm volatile("fence.proxy.async;\n" ::: "memory");
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

        // Inner BK/16 wgmma steps
        #pragma unroll
        for (int kk = 0; kk < BK / 16; kk++) {
            uint32_t a[4];
            const int a_row     = warp * 16 + (lane % 16);
            const int a_col_log  = kk * 2 + (lane / 16);
            const int a_col_phys = (A_SWZ_PERIOD > 0)
                ? (a_col_log ^ (a_row % A_SWZ_PERIOD)) : a_col_log;
            ldmatrix_x4(a[0], a[1], a[2], a[3],
                (uint32_t)__cvta_generic_to_shared(&A_sh[cur][a_row][a_col_phys * 8]));

            // B descriptor: advance by kk × K_STEP_BYTES from sub-tile 0 base.
            // Each kk step = 16 K-rows × 64 BF16-wide × 2 bytes = 2048 bytes.
            // (Must NOT use B_sh[cur][kk*16][0] — that uses the row-major stride
            //  BN*2 which is wrong when BN > 64 and TMA wrote packed sub-tiles.)
            uint32_t b_base = (uint32_t)__cvta_generic_to_shared(&B_sh[cur][0][0]);
            uint64_t desc_b = make_wgmma_b_desc<BN, BK>(b_base + kk * K_STEP_BYTES);

            wgmma_call<BN>(acc, a, desc_b);
        }

        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

        __syncthreads();
    }

    // ── Epilogue: last tile ───────────────────────────────────────────────────
    const int last = (num_tiles - 1) & 1;
    mbar_wait(&mbar[last], 0);

    asm volatile("fence.proxy.async;\n" ::: "memory");
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

    #pragma unroll
    for (int kk = 0; kk < BK / 16; kk++) {
        uint32_t a[4];
        const int a_row      = warp * 16 + (lane % 16);
        const int a_col_log  = kk * 2 + (lane / 16);
        const int a_col_phys = (A_SWZ_PERIOD > 0)
            ? (a_col_log ^ (a_row % A_SWZ_PERIOD)) : a_col_log;
        ldmatrix_x4(a[0], a[1], a[2], a[3],
            (uint32_t)__cvta_generic_to_shared(&A_sh[last][a_row][a_col_phys * 8]));
        uint32_t b_base_last = (uint32_t)__cvta_generic_to_shared(&B_sh[last][0][0]);
        uint64_t desc_b = make_wgmma_b_desc<BN, BK>(b_base_last + kk * K_STEP_BYTES);
        wgmma_call<BN>(acc, a, desc_b);
    }

    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

    // ── Write accumulators to C ───────────────────────────────────────────────
    // wgmma m64nNk16 output layout (same structure as mma.sync m16n8k16 per-warp,
    // extended to 4 warps covering all 64 rows):
    //   Thread (warp=w, lane=l) owns acc[j*4 : j*4+4] for each j in 0..BN/8-1:
    //     acc[j*4+0], acc[j*4+1]  →  row (w*16 + l/4),     col (j*8 + (l%4)*2)
    //     acc[j*4+2], acc[j*4+3]  →  row (w*16 + l/4 + 8), col (j*8 + (l%4)*2)
    //   Each pair is written as a bfloat162 (vectorised store).
    const int base_col = (lane % 4) * 2;
    const int base_row = lane / 4;
    #pragma unroll
    for (int j = 0; j < BN / 8; j++) {
        const int gc  = block_col + j * 8 + base_col;
        const int gr0 = block_row + warp * 16 + base_row;
        const int gr8 = gr0 + 8;
        if (gr0 < M && gc < N)
            *reinterpret_cast<__nv_bfloat162*>(&C[gr0 * N + gc]) =
                __floats2bfloat162_rn(acc[j*4+0], acc[j*4+1]);
        if (gr8 < M && gc < N)
            *reinterpret_cast<__nv_bfloat162*>(&C[gr8 * N + gc]) =
                __floats2bfloat162_rn(acc[j*4+2], acc[j*4+3]);
    }
}

// ── Kernel entry points ───────────────────────────────────────────────────────

#define MAKE_LAUNCHER(BN_, BK_)                                                  \
extern "C" __global__ __launch_bounds__(128, LB_MIN_BLOCKS)                     \
void matmul_h2s2_bn##BN_##_bk##BK_(                                              \
    const __grid_constant__ TmaDesc tma_A,                                       \
    const __grid_constant__ TmaDesc tma_B,                                       \
    __nv_bfloat16* __restrict__ C, int M, int K, int N)                          \
{                                                                                 \
    h2s2_impl<BN_, BK_>(tma_A, tma_B, C, M, K, N);                              \
}

MAKE_LAUNCHER( 64, 16) MAKE_LAUNCHER( 64, 32) MAKE_LAUNCHER( 64, 64)
MAKE_LAUNCHER(128, 16) MAKE_LAUNCHER(128, 32) MAKE_LAUNCHER(128, 64)
MAKE_LAUNCHER(256, 16) MAKE_LAUNCHER(256, 32) MAKE_LAUNCHER(256, 64)

#undef MAKE_LAUNCHER
