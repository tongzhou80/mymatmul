"""Stage 3 + explicit warp tiling CUDA matmul kernels."""

import re
from .._pycuda_loader import launch_matmul

_EXT = "_matmul_cuda_ext_s3_warp"

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

_KERNELS = [
    "matmul_cuda_s3_warp_tm8_tn8_bm128_bn128_bk32_wm64_wn32_u8",
    "matmul_cuda_s3_warp_tm8_tn8_bm128_bn128_bk32_wm32_wn64_u8",
    "matmul_cuda_s3_warp_tm8_tn8_bm128_bn64_bk32_wm64_wn32_u8",
    "matmul_cuda_s3_warp_tm8_tn8_bm128_bn64_bk32_wm32_wn64_u8",
    "matmul_cuda_s3_warp_tm8_tn4_bm64_bn64_bk32_wm32_wn32_u8",
    "matmul_cuda_s3_warp_tm4_tn4_bm64_bn64_bk32_wm32_wn16_u8",
    "matmul_cuda_s3_warp_tm4_tn4_bm32_bn64_bk32_wm16_wn32_u8",
]

for _name in _KERNELS:
    globals()[_name.replace("matmul_cuda_", "matmul_")] = _make(_name)
