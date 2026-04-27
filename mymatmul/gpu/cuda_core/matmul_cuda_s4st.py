"""Stage 4 Strided (s4st): double-buffered cp.async with strided output assignment."""

import re
from .._pycuda_loader import launch_matmul

_EXT = "_matmul_cuda_ext_s4st"

def _make(kernel_name):
    m = re.search(r'tm(\d+)_tn(\d+)_bm(\d+)_bn(\d+)', kernel_name)
    TM, TN, BM, BN = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    THREADS = (BM // TM) * (BN // TN)
    block = (32, THREADS // 32, 1)
    def fn(A, B):
        M, _ = A.shape
        _, N = B.shape
        grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
        return launch_matmul(_EXT, kernel_name, A, B, block, grid)
    fn.__name__ = kernel_name
    return fn

for _k in ["tm8_tn8_bm64_bn64", "tm8_tn8_bm128_bn128"]:
    for _u in [1, 4, 8, 16]:
        _name = f"matmul_cuda_s4st_{_k}_bk16_u{_u}"
        globals()[_name.replace("matmul_cuda_", "matmul_")] = _make(_name)
