"""Stage 4 + A-swizzle CUDA matmul kernels: XOR swizzle on A_shared to eliminate bank conflicts."""

def _ext():
    from . import _matmul_cuda_ext_s4sw
    return _matmul_cuda_ext_s4sw

def _make(name):
    def fn(A, B): return getattr(_ext(), name)(A, B)
    fn.__name__ = name
    return fn

for _k in ["tm8_tn8_bm128_bn128", "tm8_tn8_bm128_bn64", "tm8_tn8_bm64_bn64"]:
    for _u in [1, 2, 4, 8, 16]:
        _name = f"matmul_cuda_s4sw_{_k}_bk16_u{_u}"
        globals()[_name.replace("matmul_cuda_", "matmul_")] = _make(_name)
