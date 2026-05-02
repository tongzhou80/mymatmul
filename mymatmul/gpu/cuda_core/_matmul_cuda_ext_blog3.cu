#include <cuda_runtime.h>
#include <stdint.h>

/*
 * Blog series article 3 kernel (lpha-z.hatenablog.com/entry/2025/01/26/231500).
 *
 * Fixed N=4096. Key features vs article 1:
 *   - TJ=16, TI=8 (128×128 tile instead of 64×256)
 *   - cp.async double-buffered global→smem (k2 alternates)
 *   - reg_b[2] double-buffering: compute with one half, load into the other
 *   - BK=16 split into 4×4-column pipeline stages within each tile
 *   - Compile: -arch=compute_89 -code=sm_89, no --maxrregcount
 *
 * NOTE: 1/2 == 0 in C++ integer arithmetic, so REG_MM_A(n, 0, 1/2) expands
 * to an empty loop and the interleaving in LDS_AND_REG_MM degenerates to:
 *   1. compute 512 FMAs (REG_MM_A(n, 0, 1))
 *   2. load new reg_a  (LDS_A)
 *   3. load new reg_b[1-n] (LDS_B)
 * B register double-buffering is still meaningful; A latency is hidden by
 * the pipeline stage structure across LDS_AND_REG_MM calls.
 */

static constexpr int N = 4096;

static constexpr int ThreadsPerBlockJ = 16;
static constexpr int ThreadsPerBlockI = 8;

static constexpr int SharedMemBlockK  = 16;
static constexpr int RegisterBlockJ   = 8;
static constexpr int RegisterBlockI   = 16;

static constexpr int SizeJ        = RegisterBlockJ * ThreadsPerBlockJ;                         // 128
static constexpr int SizeI        = RegisterBlockI * ThreadsPerBlockI;                         // 128
static constexpr int LoopX        = SizeI * SharedMemBlockK / ThreadsPerBlockJ / ThreadsPerBlockI; // 16
static constexpr int LoopY        = SharedMemBlockK / ThreadsPerBlockI;                        // 2
static constexpr int LoopZ        = SizeJ / ThreadsPerBlockJ;                                  // 8
static constexpr int ThreadsPerBlock = ThreadsPerBlockI * ThreadsPerBlockJ;                    // 128

__device__ void cp_async_cg_shared_global_L2_128B(float* dst, float* src) {
    uint32_t dst_u = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
        :: "r"(dst_u), "l"(src), "n"(sizeof(float4))
    );
}

__device__ void cp_async_barrier_thread_block() {
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();
}

#define LDG_STS \
    for (int x = 0; x < LoopX/4; ++x) { \
        int ix = x * ThreadsPerBlock / (SharedMemBlockK/4); \
        int kx = x * ThreadsPerBlock % (SharedMemBlockK/4) * 4; \
        cp_async_cg_shared_global_L2_128B( \
            &p_sa[(1-k2)*SharedMemBlockK*SizeI + ix*SharedMemBlockK + kx], \
            &p_ga[ix*N+kx]); \
    } \
    for (int z = 0; z < LoopZ/4; ++z) \
    for (int y = 0; y < LoopY; ++y) { \
        int ky = y * ThreadsPerBlockI; \
        int jz = z * ThreadsPerBlockJ * 4; \
        cp_async_cg_shared_global_L2_128B( \
            &p_sb[(1-k2)*SharedMemBlockK*SizeJ + ky*SizeJ + jz], \
            &p_gb[ky*N+jz]); \
    }

#define LDS_A(offset, t, u) \
    for (int k1 = 0; k1 < 4; ++k1) \
    for (int i1 = RegisterBlockI*(t); i1 < RegisterBlockI*(u); ++i1) { \
        int il = i1 * ThreadsPerBlockI; \
        reg_a[i1][k1] = p_la[k2*SharedMemBlockK*SizeI + il*SharedMemBlockK + (k1+offset)]; \
    }

#define LDS_B(n, offset) \
    for (int k1 = 0; k1 < 4; ++k1) \
    for (int j1 = 0; j1 < RegisterBlockJ/4; ++j1) \
    for (int j3 = 0; j3 < 4; ++j3) { \
        int jl = j1 * ThreadsPerBlockJ * 4 + j3; \
        reg_b[n][k1][j1*4+j3] = p_lb[k2*SharedMemBlockK*SizeJ + (k1+offset)*SizeJ + jl]; \
    }

#define REG_MM_A(n, t, u) \
    for (int k1 = 0; k1 < 4; ++k1) \
    for (int i1 = RegisterBlockI*(t); i1 < RegisterBlockI*(u); ++i1) \
    for (int j1 = 0; j1 < RegisterBlockJ/4; ++j1) \
    for (int j3 = 0; j3 < 4; ++j3) { \
        sum[j1*4+j3][i1] = fma(reg_a[i1][k1], reg_b[n][k1][j1*4+j3], sum[j1*4+j3][i1]); \
    }

/* 1/2 == 0 in integer arithmetic: REG_MM_A(n,0,0) and LDS_A(offset,0,0) are empty,
 * so this expands to: REG_MM_A(n,0,1), LDS_A(offset,0,1), LDS_B(1-n,offset). */
#define LDS_AND_REG_MM(n, offset) \
    REG_MM_A(n, 0, 1/2) \
    LDS_A(offset, 0, 1/2) \
    REG_MM_A(n, 1/2, 1) \
    LDS_A(offset, 1/2, 1) \
    LDS_B(1-n, offset)

extern "C" __global__
void matmul_cuda_blog3(float* a, float* b, float* c) {
    int i0  = blockIdx.y * SizeI;
    int j0  = blockIdx.x * SizeJ;
    int i2  = threadIdx.y;
    int j2  = threadIdx.x;
    int tid = threadIdx.y * ThreadsPerBlockJ + threadIdx.x;

    int it = tid / (SharedMemBlockK/4);
    int kt = tid % (SharedMemBlockK/4) * 4;
    int jt = j2 * 4;

    float sum[RegisterBlockJ][RegisterBlockI] = {};

    __align__(16) __shared__ float local_b[2*SharedMemBlockK*SizeJ];  // 16 KiB
    __align__(16) __shared__ float local_a[2*SizeI*SharedMemBlockK];  // 16 KiB

    float* p_ga = &a[(i0+it)*N + kt];
    float* p_gb = &b[i2*N + (j0+jt)];
    float* p_sa = &local_a[it*SharedMemBlockK + kt];
    float* p_sb = &local_b[i2*SizeJ + jt];
    float* p_la = &local_a[i2*SharedMemBlockK];
    float* p_lb = &local_b[jt];
    float* p_gc = &c[(i0+i2)*N + (j0+jt)];

    int k2 = 0;

    float reg_a[RegisterBlockI][4];
    float reg_b[2][4][RegisterBlockJ];

    // Prologue: load first tile (k0=0) into buffer k2=1.
    {
        k2 = 1 - k2;
        LDG_STS;
        p_ga += SharedMemBlockK;
        p_gb += SharedMemBlockK * N;
        cp_async_barrier_thread_block();
    }

    // Prologue: load second tile (k0=BK) into buffer 1-k2, compute first 3/4 of k0=0.
    {
        k2 = 1 - k2;
        LDG_STS;
        LDS_A(0, 0, 1);
        LDS_B(1, 0);

        LDS_AND_REG_MM(1, 4);
        LDS_AND_REG_MM(0, 8);
        LDS_AND_REG_MM(1, 12);

        p_ga += SharedMemBlockK;
        p_gb += SharedMemBlockK * N;
        cp_async_barrier_thread_block();
    }

    for (int k0 = SharedMemBlockK; k0 < N - SharedMemBlockK; k0 += SharedMemBlockK) {
        k2 = 1 - k2;

        // Compute last 1/4 of previous tile (using reg_a/reg_b[0] from prior iter).
        LDS_AND_REG_MM(0, 0);

        // Async-load next tile while computing current tile.
        LDG_STS;

        LDS_AND_REG_MM(1, 4);
        LDS_AND_REG_MM(0, 8);
        LDS_AND_REG_MM(1, 12);

        p_ga += SharedMemBlockK;
        p_gb += SharedMemBlockK * N;
        cp_async_barrier_thread_block();
    }

    // Epilogue: compute the last two tiles (no more global loads).
    {
        k2 = 1 - k2;

        LDS_AND_REG_MM(0, 0);
        LDS_AND_REG_MM(1, 4);
        LDS_AND_REG_MM(0, 8);
        LDS_AND_REG_MM(1, 12);

        REG_MM_A(0, 0, 1);  // final 512 FMAs from reg_b[0]
    }

    // Writeback (C must be zeroed by caller).
    for (int i1 = 0; i1 < RegisterBlockI; ++i1)
    for (int j1 = 0; j1 < RegisterBlockJ/4; ++j1)
    for (int j3 = 0; j3 < 4; ++j3) {
        int i = i1 * ThreadsPerBlockI;
        int j = j1 * ThreadsPerBlockJ * 4 + j3;
        p_gc[i*N+j] += sum[j1*4+j3][i1];
    }
}
