// CUTLASS 3 BF16 row-major GEMM, multi-config.
// Compiled to a shared library; each MAKE_LAUNCHER instantiation produces
// one extern "C" entry point with the tile/cluster shape baked in.
// Python wrapper autotunes over them.

#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <cuda_runtime.h>

using namespace cute;

#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
#error "Need SM90 support"
#endif

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAcc = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;
static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;

// ── Cached workspace ────────────────────────────────────────────────────────
// CUTLASS may need a small workspace allocation per call. Cache it across
// calls so cudaMalloc/cudaFree overhead doesn't dominate small problem sizes.
namespace {
void*  g_workspace      = nullptr;
size_t g_workspace_size = 0;

void* ensure_workspace(size_t bytes) {
    if (bytes == 0) return nullptr;
    if (bytes > g_workspace_size) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, bytes);
        g_workspace_size = bytes;
    }
    return g_workspace;
}
}  // namespace

// ── Templated GEMM impl ─────────────────────────────────────────────────────

template <typename TileShape, typename ClusterShape>
struct CutlassGemm {
    // Cooperative warp-specialised schedule: avoids the wgmma-across-function-
    // boundary serialisation that hits the non-cooperative path.
    using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, ElementAcc,
        ElementC, LayoutC, AlignC,
        ElementC, LayoutC, AlignC,
        EpilogueSchedule
      >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignA,
        ElementB, LayoutB, AlignB,
        ElementAcc,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule
      >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue>;

    using Gemm   = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;

    // Per-instantiation cache: the initialized `Gemm` object is reused as long
    // as (M, N, K, A, B, C) match the previous call. First call per shape pays
    // the descriptor-build cost (~30µs); subsequent calls just `gemm.run()`.
    struct Cache {
        Gemm gemm;
        const void* A = nullptr;
        const void* B = nullptr;
        void* C = nullptr;
        int M = -1, N = -1, K = -1;
        bool initialized = false;
    };

    static int run(const void* A_ptr, const void* B_ptr, void* C_ptr,
                   int M, int N, int K) {
        static Cache cache;
        bool needs_init = !cache.initialized
                          || cache.M != M || cache.N != N || cache.K != K
                          || cache.A != A_ptr || cache.B != B_ptr || cache.C != C_ptr;
        if (needs_init) {
            StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
            StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
            StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});

            typename Gemm::Arguments args{
                cutlass::gemm::GemmUniversalMode::kGemm,
                {M, N, K},
                {
                    reinterpret_cast<const ElementA*>(A_ptr), stride_A,
                    reinterpret_cast<const ElementB*>(B_ptr), stride_B,
                },
                {
                    {1.0f, 0.0f},
                    nullptr, stride_C,
                    reinterpret_cast<ElementC*>(C_ptr), stride_C,
                }
            };

            if (cache.gemm.can_implement(args) != cutlass::Status::kSuccess) return -1;

            void* ws = ensure_workspace(Gemm::get_workspace_size(args));
            if (cache.gemm.initialize(args, ws) != cutlass::Status::kSuccess) return -2;
            cache.A = A_ptr; cache.B = B_ptr; cache.C = C_ptr;
            cache.M = M; cache.N = N; cache.K = K;
            cache.initialized = true;
        }
        return (cache.gemm.run() == cutlass::Status::kSuccess) ? 0 : -3;
    }
};

// ── Launcher table ──────────────────────────────────────────────────────────

#define MAKE_LAUNCHER(BM, BN, BK, CX, CY)                                       \
    extern "C" int cutlass_gemm_bf16_bm##BM##_bn##BN##_bk##BK##_cx##CX##_cy##CY(\
        const void* A, const void* B, void* C, int M, int N, int K) {           \
        return CutlassGemm<Shape<_##BM,_##BN,_##BK>, Shape<_##CX,_##CY,_1>>     \
                ::run(A, B, C, M, N, K);                                        \
    }

// Generated configs (kept in sync with matmul_cutlass.py _CONFIGS):
// Cooperative schedule requires BM ≥ 128 (M-tile size at least 128).
MAKE_LAUNCHER(128, 128, 64, 1, 1)
MAKE_LAUNCHER(128, 128, 64, 2, 1)
MAKE_LAUNCHER(128, 128, 64, 1, 2)
MAKE_LAUNCHER(128, 256, 64, 1, 1)
MAKE_LAUNCHER(128, 256, 64, 2, 1)
MAKE_LAUNCHER(256, 128, 64, 1, 1)
MAKE_LAUNCHER(256, 128, 64, 1, 2)

// Free the cached workspace (called at exit from Python).
extern "C" void cutlass_free_workspace() {
    if (g_workspace) {
        cudaFree(g_workspace);
        g_workspace = nullptr;
        g_workspace_size = 0;
    }
}
