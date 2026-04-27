"""Stage 0/1 CUDA matmul kernels."""

import torch
from .._pycuda_loader import launch_matmul

_EXT = "_matmul_cuda_ext"

# Grid/block configs keyed by kernel function name.
# block = (x, y, z), grid derived from M, N, tile sizes.
_CONFIGS = {
    "matmul_naive_ijk_2d_grid":           dict(block=(16,16,1), tile=(16,16)),
    "matmul_naive_ijk_2d_grid_jx":        dict(block=(16,16,1), tile=(16,16), jx=True),
    "matmul_tiled_32x32":                 dict(block=(32,32,1), tile=(32,32)),
    "matmul_tiled_32x32_16x16_threads":   dict(block=(16,16,1), tile=(32,32)),
    "matmul_tiled_32x32_32x8_threads":    dict(block=(32, 8,1), tile=(32,32)),
    "matmul_tiled_32x32_32x4_threads":    dict(block=(32, 4,1), tile=(32,32)),
    "matmul_tiled_32x64_32x4_threads":    dict(block=(32, 4,1), tile=(32,64)),
    "matmul_tiled_32x64_tm4_tn4":         dict(block=(32, 4,1), tile=(32,64)),
}


def _make(kernel_name, cfg):
    block = cfg["block"]
    BM, BN = cfg["tile"]
    jx = cfg.get("jx", False)
    def fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, _ = A.shape
        _, N = B.shape
        if jx:
            grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
        else:
            grid = ((M + BM - 1) // BM, (N + BN - 1) // BN, 1)
        return launch_matmul(_EXT, kernel_name, A, B, block, grid)
    fn.__name__ = kernel_name
    return fn


def matmul_cuda_naive_ijk(A, B):
    M, _ = A.shape; _, N = B.shape
    return launch_matmul(_EXT, "matmul_naive_ijk_2d_grid", A, B,
                         (16,16,1), ((M+15)//16, (N+15)//16, 1))

def matmul_cuda_naive_ijk_jx(A, B):
    M, _ = A.shape; _, N = B.shape
    return launch_matmul(_EXT, "matmul_naive_ijk_2d_grid_jx", A, B,
                         (16,16,1), ((N+15)//16, (M+15)//16, 1))

def matmul_cuda_tiled_32x32(A, B):
    M, _ = A.shape; _, N = B.shape
    return launch_matmul(_EXT, "matmul_tiled_32x32", A, B,
                         (32,32,1), ((N+31)//32, (M+31)//32, 1))

def matmul_cuda_tiled_32x32_threads_16x16(A, B):
    M, _ = A.shape; _, N = B.shape
    return launch_matmul(_EXT, "matmul_tiled_32x32_16x16_threads", A, B,
                         (16,16,1), ((N+31)//32, (M+31)//32, 1))

def matmul_cuda_tiled_32x32_threads_32x8(A, B):
    M, _ = A.shape; _, N = B.shape
    return launch_matmul(_EXT, "matmul_tiled_32x32_32x8_threads", A, B,
                         (32,8,1), ((N+31)//32, (M+31)//32, 1))

def matmul_cuda_tiled_32x32_threads_32x4(A, B):
    M, _ = A.shape; _, N = B.shape
    return launch_matmul(_EXT, "matmul_tiled_32x32_32x4_threads", A, B,
                         (32,4,1), ((N+31)//32, (M+31)//32, 1))

def matmul_cuda_tiled_32x64_tm4_tn4(A, B):
    M, _ = A.shape; _, N = B.shape
    return launch_matmul(_EXT, "matmul_tiled_32x64_tm4_tn4", A, B,
                         (32,4,1), ((N+63)//64, (M+31)//32, 1))

def matmul_cuda_tiled_32x64_threads_32x4(A, B):
    M, _ = A.shape; _, N = B.shape
    return launch_matmul(_EXT, "matmul_tiled_32x64_32x4_threads", A, B,
                         (32,4,1), ((N+63)//64, (M+31)//32, 1))
