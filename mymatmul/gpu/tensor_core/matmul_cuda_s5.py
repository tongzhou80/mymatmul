"""Stage 5 CUDA matmul kernels: Tensor Core WMMA (bfloat16 input, bfloat16 output)."""

import torch
from .._pycuda_loader import launch_matmul

_EXT = "_matmul_cuda_ext_s5"
_BM, _BN = 128, 128


def matmul_s5_wmma_bm128_bn128(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, _ = A.shape
    _, N = B.shape
    block = (32, 8, 1)
    grid = ((N + _BN - 1) // _BN, (M + _BM - 1) // _BM, 1)
    return launch_matmul(_EXT, "matmul_s5_wmma_bm128_bn128_bk32", A, B, block, grid,
                         out_dtype=torch.bfloat16)
