"""Stage 3 + explicit warp tiling CUDA matmul kernels."""

def _ext():
    from . import _matmul_cuda_ext_s3_warp
    return _matmul_cuda_ext_s3_warp

def _make(name):
    def fn(A, B): return getattr(_ext(), name)(A, B)
    fn.__name__ = name
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
