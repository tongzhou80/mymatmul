"""s4st + intra-warp shuffle to reduce shared memory reads."""

from .._pycuda_loader import launch_matmul

_EXT    = "_matmul_cuda_ext_s4st_shfl"
_KERNEL = "matmul_cuda_s4st_shfl_tm8_tn8_bm128_bn128_bk16"
_BM, _BN = 128, 128
_BLOCK   = (256, 1, 1)


def matmul_s4st_shfl_tm8_tn8_bm128_bn128_bk16(A, B):
    M, _ = A.shape
    _, N = B.shape
    grid = ((N + _BN - 1) // _BN, (M + _BM - 1) // _BM, 1)
    return launch_matmul(_EXT, _KERNEL, A, B, _BLOCK, grid)
