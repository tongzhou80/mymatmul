import torch


def matmul_torch_naive_ijk(A_gpu, B_gpu):
    """GPU matmul using PyTorch's optimized GEMM.

    Assumes A_gpu and B_gpu are already torch.cuda.FloatTensor on GPU.
    Returns result as GPU tensor.
    """
    return torch.mm(A_gpu, B_gpu)
