"""Stage 3 CUDA matmul kernels: square tile sizes via tx-remap."""

def _ext():
    from . import _matmul_cuda_ext_s3
    return _matmul_cuda_ext_s3

def _make(name):
    def fn(A, B): return getattr(_ext(), name)(A, B)
    fn.__name__ = name
    return fn

# BK=32, unroll=8
matmul_s3_tm4_tn4_bm32_bn64_bk32_u8   = _make("matmul_cuda_s3_tm4_tn4_bm32_bn64_bk32_u8")
matmul_s3_tm4_tn4_bm64_bn64_bk32_u8   = _make("matmul_cuda_s3_tm4_tn4_bm64_bn64_bk32_u8")
matmul_s3_tm8_tn4_bm64_bn64_bk32_u8   = _make("matmul_cuda_s3_tm8_tn4_bm64_bn64_bk32_u8")
matmul_s3_tm8_tn8_bm128_bn64_bk32_u8  = _make("matmul_cuda_s3_tm8_tn8_bm128_bn64_bk32_u8")
matmul_s3_tm8_tn8_bm128_bn128_bk32_u8 = _make("matmul_cuda_s3_tm8_tn8_bm128_bn128_bk32_u8")

# BK=16, unroll=1,2,4,8
for _u in (1, 2, 4, 8):
    for _name in (
        f"matmul_cuda_s3_tm4_tn4_bm32_bn64_bk16_u{_u}",
        f"matmul_cuda_s3_tm4_tn4_bm64_bn64_bk16_u{_u}",
        f"matmul_cuda_s3_tm8_tn4_bm64_bn64_bk16_u{_u}",
        f"matmul_cuda_s3_tm8_tn8_bm128_bn64_bk16_u{_u}",
        f"matmul_cuda_s3_tm8_tn8_bm128_bn128_bk16_u{_u}",
    ):
        globals()[_name.replace("matmul_cuda_", "matmul_")] = _make(_name)
