import torch


def matmul_torch(A_gpu, B_gpu):
    """GPU matmul using PyTorch's optimized GEMM (highly tuned cuBLAS wrapper).

    Assumes A_gpu and B_gpu are already torch.cuda.FloatTensor on GPU.
    Returns result as GPU tensor (no host transfers).
    """
    return torch.mm(A_gpu, B_gpu)
