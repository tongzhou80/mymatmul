import torch


def matmul_cuda_naive_ijk(A_gpu, B_gpu):
    """Naive CUDA matmul: 2D grid of threads (i->blockIdx.x, j->blockIdx.y).
    Each thread (i,j) computes full k reduction.

    Assumes A_gpu and B_gpu are already on GPU.
    Returns result as GPU tensor.
    """
    from . import _matmul_cuda_ext
    return _matmul_cuda_ext.matmul_cuda_naive_ijk(A_gpu, B_gpu)


def matmul_cuda_naive_ijk_jx(A_gpu, B_gpu):
    """Naive CUDA matmul: 2D grid of threads (j->blockIdx.x, i->blockIdx.y).
    Maps j (column, fastest-changing in row-major) to x (fastest CUDA dimension).
    Each thread (i,j) computes full k reduction.

    Assumes A_gpu and B_gpu are already on GPU.
    Returns result as GPU tensor.
    """
    from . import _matmul_cuda_ext
    return _matmul_cuda_ext.matmul_cuda_naive_ijk_jx(A_gpu, B_gpu)


def matmul_cuda_tiled_32x32(A_gpu, B_gpu):
    """Tiled CUDA matmul: 32x32 tiles with shared memory.
    Processes K dimension in 32-element tiles, loading into shared memory.
    Each thread (i,j) maintains its own running sum across all K-tiles.
    Thread block: 32x32 = 1024 threads.

    Assumes A_gpu and B_gpu are already on GPU.
    Returns result as GPU tensor.
    """
    from . import _matmul_cuda_ext
    return _matmul_cuda_ext.matmul_cuda_tiled_32x32(A_gpu, B_gpu)


def matmul_cuda_tiled_32x32_threads_16x16(A_gpu, B_gpu):
    """Optimized tiled CUDA matmul: 32x32 tiles with 16x16 threads.

    Thread mapping:
    - Block computes C tile [block_row : block_row+32, block_col : block_col+32]
    - Each thread computes a 2x2 micro-tile with 4 accumulator registers

    Memory access pattern:
    - Each thread loads 4 elements into A_shared and 4 into B_shared
    - Shared memory: A_shared[32][32] + B_shared[32][32]

    Benefits:
    - Reduced synchronization overhead (256 vs 1024 threads)
    - Same data reuse from shared memory (32x)
    - Better register usage per thread
    - More instruction-level parallelism (4 independent accumulators)
    - 2.4x faster than 32x32 threads version

    Assumes A_gpu and B_gpu are already on GPU and contiguous.
    Returns result as GPU tensor.
    """
    from . import _matmul_cuda_ext
    return _matmul_cuda_ext.matmul_cuda_tiled_32x32_16x16_threads(A_gpu, B_gpu)


def matmul_cuda_tiled_32x32_threads_32x8(A_gpu, B_gpu):
    """Warp-aligned tiled CUDA matmul: 32x32 tiles with 32x8 threads.

    Thread mapping:
    - Blocksize: 32x8 = 256 threads (8 warps)
    - Block computes C tile [block_row : block_row+32, block_col : block_col+32]
    - Each thread computes a 4x1 micro-tile (4 rows, 1 column)

    Warp alignment:
    - One warp (32 threads) operates on one row stripe of the output tile
    - Each thread (tx, ty) computes 4 consecutive rows with tx as the column

    Memory access pattern:
    - Linearized loading: each thread loads 4 elements from A and B
    - Shared memory: A_shared[32][32] + B_shared[32][32]

    Assumes A_gpu and B_gpu are already on GPU and contiguous.
    Returns result as GPU tensor.
    """
    from . import _matmul_cuda_ext
    return _matmul_cuda_ext.matmul_cuda_tiled_32x32_32x8_threads(A_gpu, B_gpu)


def matmul_cuda_tiled_32x32_threads_32x4(A_gpu, B_gpu):
    """Ultra-warp-aligned tiled CUDA matmul: 32x32 tiles with 32x4 threads.

    Thread mapping:
    - Blocksize: 32x4 = 128 threads (4 warps)
    - Block computes C tile [block_row : block_row+32, block_col : block_col+32]
    - Each thread computes an 8x1 micro-tile (8 rows, 1 column)

    Warp alignment:
    - Each warp operates on 8 consecutive rows of the output tile
    - Extreme thread efficiency: each of 32 threads in warp handles one output column

    Memory access pattern:
    - Linearized loading: each thread loads 8 elements from A and B (1024 / 128 = 8)
    - Shared memory: A_shared[32][32] + B_shared[32][32]
    - Loop unrolled loads with #pragma unroll

    Assumes A_gpu and B_gpu are already on GPU and contiguous.
    Returns result as GPU tensor.
    """
    from . import _matmul_cuda_ext
    return _matmul_cuda_ext.matmul_cuda_tiled_32x32_32x4_threads(A_gpu, B_gpu)  # Calls the refactored kernel
