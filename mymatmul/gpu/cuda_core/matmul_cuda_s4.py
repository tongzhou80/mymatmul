"""Stage 4 CUDA matmul kernels: double-buffered with cp.async."""

import re
from .._pycuda_loader import launch_matmul

_EXT = "_matmul_cuda_ext_s4"

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

matmul_s4_tm4_tn4_bm32_bn64_bk16  = _make("matmul_cuda_s4_tm4_tn4_bm32_bn64_bk16")
matmul_s4_tm4_tn4_bm64_bn64_bk16  = _make("matmul_cuda_s4_tm4_tn4_bm64_bn64_bk16")
matmul_s4_tm8_tn4_bm64_bn64_bk16  = _make("matmul_cuda_s4_tm8_tn4_bm64_bn64_bk16")

for _k in ["tm8_tn8_bm128_bn64", "tm8_tn8_bm128_bn128", "tm8_tn8_bm64_bn64"]:
    for _u in [1, 2, 4, 8, 16]:
        _name = f"matmul_cuda_s4_{_k}_bk16_u{_u}"
        globals()[_name.replace("matmul_cuda_", "matmul_")] = _make(_name)

# s4pad: A_shared row-padded by 1 float to eliminate bank conflicts (bm64_bn64 only)
for _u in [1, 2, 4, 8, 16]:
    _name = f"matmul_cuda_s4pad_tm8_tn8_bm64_bn64_bk16_u{_u}"
    globals()[_name.replace("matmul_cuda_", "matmul_")] = _make(_name)
