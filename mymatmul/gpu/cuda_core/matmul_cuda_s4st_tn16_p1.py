"""s4st_tn16_p1: register-prefetch software pipeline (prefetch kk=0, loop issues next loads then FMAs)."""

from .._pycuda_loader import launch_matmul

_EXT = "_matmul_cuda_ext_s4st_tn16_p1"

_BM, _BN, _TM, _TN = 128, 256, 8, 16
_THREADS = (_BM // _TM) * (_BN // _TN)   # 256
_BLOCK = (32, _THREADS // 32, 1)

def matmul_s4st_tn16_p1_tm8_tn16_bm128_bn256_bk16(A, B):
    M, _ = A.shape
    _, N = B.shape
    grid = ((N + _BN - 1) // _BN, (M + _BM - 1) // _BM, 1)
    return launch_matmul(_EXT,
                         "matmul_cuda_s4st_tn16_p1_tm8_tn16_bm128_bn256_bk16",
                         A, B, _BLOCK, grid)
