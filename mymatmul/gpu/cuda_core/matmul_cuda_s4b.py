"""Stage 4b CUDA matmul kernels: Stage 4 + A_shared bank-conflict padding."""

def _ext():
    from . import _matmul_cuda_ext_s4b
    return _matmul_cuda_ext_s4b

def _make(name):
    def fn(A, B): return getattr(_ext(), name)(A, B)
    fn.__name__ = name
    return fn

for _u in [8, 16]:
    _name = f"matmul_cuda_s4b_tm8_tn8_bm128_bn128_bk16_u{_u}"
    globals()[_name.replace("matmul_cuda_", "matmul_")] = _make(_name)
