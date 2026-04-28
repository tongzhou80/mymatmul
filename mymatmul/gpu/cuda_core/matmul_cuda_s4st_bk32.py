"""s4st BK=32 with dynamic shared memory (64 KB, above 48 KB static limit)."""

import re
from .._pycuda_loader import launch_matmul

_EXT = "_matmul_cuda_ext_s4st_bk32"

def _make(kernel_name, smem_bytes):
    m = re.search(r'tm(\d+)_tn(\d+)_bm(\d+)_bn(\d+)', kernel_name)
    TM, TN, BM, BN = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    THREADS = (BM // TM) * (BN // TN)
    block = (32, THREADS // 32, 1)
    def fn(A, B):
        M, _ = A.shape
        _, N = B.shape
        grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
        return launch_matmul(_EXT, kernel_name, A, B, block, grid, smem_bytes=smem_bytes)
    fn.__name__ = kernel_name
    return fn

BK = 32
for _u in [1, 4, 8, 16, 32]:
    _k = "tm8_tn8_bm128_bn128"
    BM, BN = 128, 128
    _smem = 2 * (BM * BK + BK * BN) * 4   # 65536 bytes
    _name = f"matmul_cuda_s4st_bk32_{_k}_bk32_u{_u}"
    globals()[_name.replace("matmul_cuda_", "matmul_")] = _make(_name, _smem)
