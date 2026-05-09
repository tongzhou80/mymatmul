"""TC8_gemini: Gemini-generated Triton-style 5-stage BF16 matmul.

Fixed config: BM=64, BN=128, BK=16, NW=4, STAGES=5 (5 smem slots).
Issue-before-compute with GROUP_M=8 CTA swizzle.
1-D grid launch: grid_size = (M/BM) * (N/BN).

Smem: 5 × (BM×BK + BK×BN) × 2 = 30720 bytes.
Constraints: M % 64 == 0, N % 128 == 0, K % 16 == 0.
"""

import numpy as np
import torch
from .._pycuda_loader import get_module, _ensure_ctx
import pycuda.driver as drv

DTYPE = torch.bfloat16

_EXT   = "_matmul_cuda_ext_tc8_gemini"
_KNAME = "matmul_cuda_tc8_gemini"
_BM    = 64
_BN    = 128
_SMEM  = 30720  # 5 slots × 3072 elements × 2 bytes

_fn = None


def _get_fn():
    global _fn
    if _fn is None:
        mod = get_module(_EXT)
        _fn = mod.get_function(_KNAME)
        _fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, _SMEM)
    return _fn


def matmul_tc8_gemini(A, B):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    fn = _get_fn()
    grid_size = (M // _BM) * (N // _BN)
    fn(np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(C.data_ptr()),
       np.int32(M), np.int32(K), np.int32(N),
       block=(128, 1, 1), grid=(grid_size, 1, 1), shared=_SMEM)
    return C
