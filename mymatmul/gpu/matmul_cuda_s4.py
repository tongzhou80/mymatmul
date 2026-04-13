"""Stage 4 CUDA matmul kernels: double-buffered with cp.async."""

def _ext():
    from . import _matmul_cuda_ext2
    return _matmul_cuda_ext2

def _make(name):
    def fn(A, B): return getattr(_ext(), name)(A, B)
    fn.__name__ = name
    return fn

matmul_s4_tm4_tn4_bm32_bn64_bk16  = _make("matmul_cuda_s4_tm4_tn4_bm32_bn64_bk16")
matmul_s4_tm4_tn4_bm64_bn64_bk16  = _make("matmul_cuda_s4_tm4_tn4_bm64_bn64_bk16")
matmul_s4_tm8_tn4_bm64_bn64_bk16  = _make("matmul_cuda_s4_tm8_tn4_bm64_bn64_bk16")
matmul_s4_tm8_tn8_bm128_bn64_bk16 = _make("matmul_cuda_s4_tm8_tn8_bm128_bn64_bk16")
matmul_s4_tm8_tn8_bm128_bn128_bk16= _make("matmul_cuda_s4_tm8_tn8_bm128_bn128_bk16")
