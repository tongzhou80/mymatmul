"""s4st TN=16: TM=8, TN=16, BM=128, BN=256, BK=16; PTX immediate-offset smem loads."""

from .._pycuda_loader import launch_matmul

_EXT = "_matmul_cuda_ext_s4st_tn16"

_BM, _BN, _TM, _TN = 128, 256, 8, 16
_THREADS = (_BM // _TM) * (_BN // _TN)   # 256
_BLOCK = (32, _THREADS // 32, 1)          # (32, 8, 1)

def _make(kernel_name):
    def fn(A, B):
        M, _ = A.shape
        _, N = B.shape
        grid = ((N + _BN - 1) // _BN, (M + _BM - 1) // _BM, 1)
        return launch_matmul(_EXT, kernel_name, A, B, _BLOCK, grid)
    fn.__name__ = kernel_name
    return fn

for _u in [1, 2, 4, 8, 16]:
    _name = f"matmul_cuda_s4st_tn16_tm8_tn16_bm128_bn256_bk16_u{_u}"
    globals()[_name.replace("matmul_cuda_", "matmul_")] = _make(_name)
