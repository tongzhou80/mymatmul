"""TC8_4096: Triton-style 5-stage BF16 tensor-core matmul.

Fixed config: BM=64, BN=128, BK=16, NW=4, NS=5 (4 smem ring-buffer slots).
Issue-after-compute pipeline with cp.async.cg (bypass L1) and 1 sync/iter.

Constraints: M % 64 == 0, N % 128 == 0, K % 16 == 0.
"""

import torch
from .._pycuda_loader import get_module, launch_matmul

DTYPE = torch.bfloat16

_EXT    = "_matmul_cuda_ext_tc8_4096"
_KNAME  = "matmul_cuda_tc8_4096"
_BM     = 64
_BN     = 128
_SMEM   = 24576  # 4 slots × 3072 elements × 2 bytes


def _grid(M, N):
    return (N // _BN, M // _BM, 1)


def matmul_tc8_4096(A, B):
    M, K = A.shape
    _, N = B.shape
    return launch_matmul(_EXT, _KNAME, A, B,
                         block=(32, 4, 1),
                         grid=_grid(M, N),
                         smem_bytes=_SMEM)
