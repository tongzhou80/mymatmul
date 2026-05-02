"""Blog series article 1 kernel: 128-thread, 64×256 tile, BK=16, no pipelining."""

import torch
from .._pycuda_loader import launch_matmul, get_module, _EXTRA_FLAGS

_EXT    = "_matmul_cuda_ext_blog1"
_KNAME  = "matmul_cuda_blog1"
_BM, _BN, _BK = 64, 256, 16
_SMEM   = (_BM * 2 * _BK + _BK * 2 * _BN) * 4   # over-allocated: 40 KiB

_EXTRA_FLAGS[_EXT] = ["-arch=compute_86", "-code=sm_86", "--maxrregcount=255"]


def _block():
    return (32, 4, 1)   # TJ=32, TI=4


def _grid(M, N):
    return ((N + _BN - 1) // _BN, (M + _BM - 1) // _BM, 1)


def matmul_blog1(A, B):
    M, K = A.shape
    _, N = B.shape
    get_module(_EXT)
    return launch_matmul(_EXT, _KNAME, A, B, _block(), _grid(M, N), smem_bytes=_SMEM)
