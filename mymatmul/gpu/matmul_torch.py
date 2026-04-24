import torch


def matmul_torch(A_gpu, B_gpu):
    """GPU matmul using PyTorch's optimized GEMM (highly tuned cuBLAS wrapper).

    Assumes A_gpu and B_gpu are already torch.cuda.FloatTensor on GPU.
    Returns result as GPU tensor (no host transfers).
    """
    return torch.mm(A_gpu, B_gpu)


def matmul_torch_fp32_notf32(A_gpu, B_gpu):
    """cuBLAS FP32 matmul with TF32 disabled: pure FP32 SIMT, comparable to our s4 kernels.

    Accepts BF16 inputs (to fit the BF16 benchmark framework), converts to FP32
    internally, disables TF32 for the computation, then returns BF16.
    """
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    result = torch.mm(A_gpu.float(), B_gpu.float())
    torch.backends.cuda.matmul.allow_tf32 = old
    return result.to(A_gpu.dtype)
