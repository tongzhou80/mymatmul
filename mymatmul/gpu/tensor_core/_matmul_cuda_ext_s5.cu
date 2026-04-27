#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda::wmma;

/*
 * Stage 5: Tensor Core WMMA
 * ==========================
 * Same tile structure as Stage 3 best config (BM=BN=128, BK=32, 256 threads),
 * but replaces scalar FMA with warp-level WMMA instructions (tensor cores).
 *
 * WMMA tile shape: M=16, N=16, K=16  (bf16 inputs, float32 accumulators)
 *
 * Thread block: 256 threads = 8 warps, physical layout 32x8
 * Warp layout:  WARPS_M=2 x WARPS_N=4  (2 rows, 4 cols of warps within the block tile)
 * Per-warp output region: WM=64 x WN=32
 *   → WTILES_M = WM/16 = 4  wmma tiles in M
 *   → WTILES_N = WN/16 = 2  wmma tiles in N
 *   → 4x2 = 8 accumulator fragments per warp  (same data as Stage 3's 64 float accumulators)
 * K steps per BK: BK/WMMA_K = 32/16 = 2
 *
 * Shared memory layout:
 *   A_shared[128][32] bf16 = 8192 B
 *   B_shared[ 32][128] bf16 = 8192 B
 *   Total = 16384 B = 16 KB
 *
 * Write-back (no scratch buffer):
 *   Accumulators live in the general register file as fragment<accumulator,...>.x[0..7].
 *   On SM 8.x the 16×16 float32 wmma accumulator fragment layout is:
 *     thread t (lane 0..31), element e (0..7):
 *       frag_row = (t / 4) + row_offset[e]   where row_offset = {0,0,8,8,0,0,8,8}
 *       frag_col = (t % 4)*2 + col_offset[e] where col_offset = {0,1,0,1,8,9,8,9}
 *   Each thread writes 8 bf16 values directly to global C — no shared memory needed.
 *
 * Key difference vs Stage 3:
 *   The 64 float32 accumulators per thread are still needed (same arithmetic),
 *   but tensor core mma_sync handles BK*16*16 = 8192 FMAs per warp in one instruction
 *   rather than explicit scalar FMA loops. The compiler no longer needs to keep the
 *   a[TM] and b[TN] register arrays live across a fully unrolled BK loop, which
 *   should significantly reduce non-accumulator register pressure.
 *
 * Compile with -Xptxas -v to compare register counts against Stage 3.
 */

constexpr int S5_BM      = 128;
constexpr int S5_BN      = 128;
constexpr int S5_BK      = 32;
constexpr int S5_WMMA_M  = 16;
constexpr int S5_WMMA_N  = 16;
constexpr int S5_WMMA_K  = 16;
constexpr int S5_WARPS_M = 2;
constexpr int S5_WARPS_N = 4;
constexpr int S5_WM      = S5_BM / S5_WARPS_M;         // 64
constexpr int S5_WN      = S5_BN / S5_WARPS_N;         // 32
constexpr int S5_WTILES_M = S5_WM / S5_WMMA_M;         // 4
constexpr int S5_WTILES_N = S5_WN / S5_WMMA_N;         // 2
constexpr int S5_K_STEPS  = S5_BK / S5_WMMA_K;         // 2
constexpr int S5_THREADS  = S5_WARPS_M * S5_WARPS_N * 32; // 256

__global__ void matmul_s5_wmma_bm128_bn128_bk32(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int K, int N
) {
    __shared__ __nv_bfloat16 A_shared[S5_BM][S5_BK];                     // 8192 B
    __shared__ __nv_bfloat16 B_shared[S5_BK][S5_BN];                     // 8192 B

    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane_id  = tid % 32;
    const int warp_row = warp_id / S5_WARPS_N;   // [0, WARPS_M) = [0, 2)
    const int warp_col = warp_id % S5_WARPS_N;   // [0, WARPS_N) = [0, 4)

    const int block_row = blockIdx.y * S5_BM;
    const int block_col = blockIdx.x * S5_BN;

    // Accumulator fragments: WTILES_M x WTILES_N = 4x2 = 8 per warp
    // Each holds a 16x16 float32 tile; on-chip they occupy ~8 registers/fragment
    fragment<accumulator, S5_WMMA_M, S5_WMMA_N, S5_WMMA_K, float>
        acc[S5_WTILES_M][S5_WTILES_N];
    for (int i = 0; i < S5_WTILES_M; i++)
        for (int j = 0; j < S5_WTILES_N; j++)
            fill_fragment(acc[i][j], 0.0f);

    // -----------------------------------------------------------------------
    // Main K-loop
    // -----------------------------------------------------------------------
    for (int k0 = 0; k0 < K; k0 += S5_BK) {

        // Cooperative load A tile (BM x BK = 128x32 = 4096 elems, 16 per thread)
        for (int idx = tid; idx < S5_BM * S5_BK; idx += S5_THREADS) {
            int r = idx / S5_BK, c = idx % S5_BK;
            int gr = block_row + r, gc = k0 + c;
            A_shared[r][c] = (gr < M && gc < K) ? A[gr * K + gc]
                                                 : __float2bfloat16(0.f);
        }
        // Cooperative load B tile (BK x BN = 32x128 = 4096 elems, 16 per thread)
        for (int idx = tid; idx < S5_BK * S5_BN; idx += S5_THREADS) {
            int r = idx / S5_BN, c = idx % S5_BN;
            int gr = k0 + r, gc = block_col + c;
            B_shared[r][c] = (gr < K && gc < N) ? B[gr * N + gc]
                                                 : __float2bfloat16(0.f);
        }
        __syncthreads();

        // Warp-level MMA over BK in steps of WMMA_K=16
        for (int ki = 0; ki < S5_K_STEPS; ki++) {
            // Load A fragments for this warp's M sub-tile
            fragment<matrix_a, S5_WMMA_M, S5_WMMA_N, S5_WMMA_K,
                     __nv_bfloat16, row_major> a_frag[S5_WTILES_M];
            for (int i = 0; i < S5_WTILES_M; i++) {
                const __nv_bfloat16* aptr =
                    &A_shared[warp_row * S5_WM + i * S5_WMMA_M][ki * S5_WMMA_K];
                load_matrix_sync(a_frag[i], aptr, S5_BK);
            }

            // Load B fragments for this warp's N sub-tile
            fragment<matrix_b, S5_WMMA_M, S5_WMMA_N, S5_WMMA_K,
                     __nv_bfloat16, row_major> b_frag[S5_WTILES_N];
            for (int j = 0; j < S5_WTILES_N; j++) {
                const __nv_bfloat16* bptr =
                    &B_shared[ki * S5_WMMA_K][warp_col * S5_WN + j * S5_WMMA_N];
                load_matrix_sync(b_frag[j], bptr, S5_BN);
            }

            // MMA: acc[i][j] += a_frag[i] * b_frag[j]
            for (int i = 0; i < S5_WTILES_M; i++)
                for (int j = 0; j < S5_WTILES_N; j++)
                    mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        __syncthreads();
    }

    // -----------------------------------------------------------------------
    // Write back: directly access fragment .x[] members — no scratch buffer.
    //
    // SM 8.x wmma f32 accumulator layout (16×16, row_major):
    //   thread t (lane 0..31), element e (0..7):
    //     frag_row = (t / 4) + row_off[e]    row_off = {0,0,8,8,0,0,8,8}
    //     frag_col = (t % 4)*2 + col_off[e]  col_off = {0,1,0,1,8,9,8,9}
    // Each thread writes 8 bf16 values directly to global C.
    // -----------------------------------------------------------------------
    constexpr int row_off[8] = {0, 0, 8, 8, 0, 0, 8, 8};
    constexpr int col_off[8] = {0, 1, 0, 1, 8, 9, 8, 9};

    const int base_row = lane_id / 4;
    const int base_col = (lane_id % 4) * 2;

    for (int i = 0; i < S5_WTILES_M; i++) {
        for (int j = 0; j < S5_WTILES_N; j++) {
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                int frag_row = base_row + row_off[e];
                int frag_col = base_col + col_off[e];
                int gr = block_row + warp_row * S5_WM + i * S5_WMMA_M + frag_row;
                int gc = block_col + warp_col * S5_WN + j * S5_WMMA_N + frag_col;
                if (gr < M && gc < N)
                    C[gr * N + gc] = __float2bfloat16(acc[i][j].x[e]);
            }
        }
    }
}


// ---------------------------------------------------------------------------
// PyTorch dispatch wrapper
// ---------------------------------------------------------------------------
torch::Tensor matmul_s5_wmma_bm128_bn128(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "Only bfloat16 supported");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    const int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(32, 8);   // 256 threads, 8 warps
    dim3 blocks((N + S5_BN - 1) / S5_BN, (M + S5_BM - 1) / S5_BM);

    matmul_s5_wmma_bm128_bn128_bk32<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, K, N
    );
    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Stage 5: Tensor Core WMMA — BM=BN=128, BK=32, 2x4 warp layout";
    m.def("matmul_s5_wmma_bm128_bn128",
          &matmul_s5_wmma_bm128_bn128,
          "WMMA bf16 kernel: BM=128,BN=128,BK=32, WARPS_M=2,WARPS_N=4, "
          "4x2 wmma tiles per warp (16x16x16 each)");
}
