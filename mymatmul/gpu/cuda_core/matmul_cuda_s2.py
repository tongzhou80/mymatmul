"""Stage 2: Shared Memory Tiling — Python wrappers."""

import torch
from .._pycuda_loader import launch_matmul

_EXT = "_matmul_cuda_ext_s2"


def matmul_s2_bm8_bn32(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, _ = A.shape
    _, N = B.shape
    block = (32, 4, 1)
    grid = ((N + 31) // 32, (M + 7) // 8, 1)
    return launch_matmul(_EXT, "smem_tiled_bm8_bn32_bk32_threads32x4", A, B, block, grid)


def matmul_s2_bm16_bn32(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, _ = A.shape
    _, N = B.shape
    block = (32, 8, 1)
    grid = ((N + 31) // 32, (M + 15) // 16, 1)
    return launch_matmul(_EXT, "smem_tiled_bm16_bn32_bk32_threads32x8", A, B, block, grid)
