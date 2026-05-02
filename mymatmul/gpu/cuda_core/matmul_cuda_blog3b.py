"""Blog3 variant: 256-thread, 256×128 tile, cp.async + register pipelining.

Fixed N=4096. Only benchmarkable at that size.
"""

import numpy as np
import torch
import pycuda.driver as drv
from .._pycuda_loader import get_module

_EXT   = "_matmul_cuda_ext_blog3b"
_KNAME = "matmul_cuda_blog3b"
_SMEM  = (2 * 256 * 16 + 2 * 16 * 128) * 4  # 48 KiB
_N     = 4096


def matmul_blog3b(A, B):
    M, K = A.shape
    _, N = B.shape
    assert M == K == N == _N, f"blog3b is fixed to {_N}×{_N}"
    C = torch.zeros((M, N), device="cuda", dtype=torch.float32)
    fn = get_module(_EXT).get_function(_KNAME)
    fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, _SMEM)
    fn(
        np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(C.data_ptr()),
        block=(16, 16, 1),
        grid=(N // 128, N // 256, 1),
        shared=_SMEM,
    )
    return C
