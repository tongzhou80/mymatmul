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
    return _matmul_cuda_ext.matmul_cuda_tiled_32x32_16x16_threads(A_gpu, B_gpu)  # Calls the refactored kernel
