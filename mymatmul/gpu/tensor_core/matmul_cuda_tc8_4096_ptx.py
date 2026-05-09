"""TC8_4096_PTX: Load and run the saved Triton BF16 PTX directly.

Fixed config from Triton autotuner: BM=64, BN=128, BK=16, GROUP_M=8,
num_warps=4, num_stages=5. Achieves 143 TFLOPS at N=4096.

Constraints: M % 64 == 0, N % 128 == 0, K % 16 == 0.
"""

import os
import numpy as np
import torch
from .._pycuda_loader import get_module_ptx
import pycuda.driver as drv

DTYPE = torch.bfloat16

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PTX    = os.path.join(_HERE, "../../../triton_ptx/triton_bf16_4096_v2.ptx")
_KNAME  = "_matmul_bf16_kernel"
_BLOCK_M = 64
_BLOCK_N = 128
_SMEM   = 24576

_fn = None


def _get_fn():
    global _fn
    if _fn is None:
        mod = get_module_ptx(_PTX)
        _fn = mod.get_function(_KNAME)
        _fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, 100352)
    return _fn


def matmul_tc8_4096_ptx(A, B):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    fn = _get_fn()
    grid_x = (M // _BLOCK_M) * (N // _BLOCK_N)
    fn(np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(C.data_ptr()),
       np.int32(M), np.int32(N), np.int32(K),
       np.int32(K), np.int32(N), np.int32(N),
       np.intp(0), np.intp(0),
       block=(128, 1, 1), grid=(grid_x, 1, 1), shared=_SMEM)
    return C
