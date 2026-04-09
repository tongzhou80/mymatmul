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


def matmul_cuda_tiled_32x32_opt(A_gpu, B_gpu):
    """Optimized tiled CUDA matmul: 32x32 tiles with 16x16 threads.
    Still processes 32x32 tiles but with fewer threads per block.
    Each thread computes a 2x2 sub-grid (4 elements) instead of 1.
    Thread block: 16x16 = 256 threads, each computing 4 elements.

    Benefits:
    - Reduced synchronization overhead (256 vs 1024 threads)
    - Same data reuse from shared memory
    - Better register usage per thread
    - Same tile coverage but more work per thread

    Assumes A_gpu and B_gpu are already on GPU.
    Returns result as GPU tensor.
    """
    from . import _matmul_cuda_ext
    return _matmul_cuda_ext.matmul_cuda_tiled_32x32_opt(A_gpu, B_gpu)
