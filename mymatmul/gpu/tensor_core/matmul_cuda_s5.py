"""Stage 5 CUDA matmul kernels: Tensor Core WMMA."""

def _ext():
    from . import _matmul_cuda_ext_s5
    return _matmul_cuda_ext_s5

def matmul_s5_wmma_bm128_bn128(A, B):
    return _ext().matmul_s5_wmma_bm128_bn128(A, B)
