"""s4st_tn16_m2: hand-crafted 2-way smem software pipelining (loads×2 then FMAs×2)."""

from .._pycuda_loader import launch_matmul

_EXT = "_matmul_cuda_ext_s4st_tn16_m2"

_BM, _BN, _TM, _TN = 128, 256, 8, 16
_THREADS = (_BM // _TM) * (_BN // _TN)   # 256
_BLOCK = (32, _THREADS // 32, 1)

def matmul_s4st_tn16_m2_tm8_tn16_bm128_bn256_bk16(A, B):
    M, _ = A.shape
    _, N = B.shape
    grid = ((N + _BN - 1) // _BN, (M + _BM - 1) // _BM, 1)
    return launch_matmul(_EXT,
                         "matmul_cuda_s4st_tn16_m2_tm8_tn16_bm128_bn256_bk16",
                         A, B, _BLOCK, grid)
