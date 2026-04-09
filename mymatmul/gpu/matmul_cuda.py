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
