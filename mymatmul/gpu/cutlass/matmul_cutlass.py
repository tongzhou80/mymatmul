"""CUTLASS BF16 GEMM Python wrapper with multi-config autotune."""

import atexit
import ctypes
import os
import subprocess
import threading

import torch
import triton.testing


_HERE = os.path.dirname(os.path.abspath(__file__))
_CU   = os.path.join(_HERE, "_matmul_cutlass.cu")
_SO   = os.path.join(_HERE, "_matmul_cutlass.so")
_CUTLASS_INC      = os.path.normpath(os.path.join(_HERE, "../../../third_party/cutlass/include"))
_CUTLASS_UTIL_INC = os.path.normpath(os.path.join(_HERE, "../../../third_party/cutlass/tools/util/include"))
_NVCC = "/usr/local/cuda-13/bin/nvcc"

# Kept in sync with MAKE_LAUNCHER entries in _matmul_cutlass.cu
_CONFIGS = [
    # Cooperative schedule requires BM ≥ 128.
    (128, 128, 64, 1, 1),
    (128, 128, 64, 2, 1),
    (128, 128, 64, 1, 2),
    (128, 256, 64, 1, 1),
    (128, 256, 64, 2, 1),
    (256, 128, 64, 1, 1),
    (256, 128, 64, 1, 2),
]

_lib_lock = threading.Lock()
_lib = None


def _kname(bm, bn, bk, cx, cy):
    return f"cutlass_gemm_bf16_bm{bm}_bn{bn}_bk{bk}_cx{cx}_cy{cy}"


def _get_lib():
    global _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        if not os.path.exists(_SO) or os.path.getmtime(_CU) > os.path.getmtime(_SO):
            print(f"[cutlass] compiling {len(_CONFIGS)}-config GEMM (~5-10 min) ...", flush=True)
            cmd = [_NVCC, "-arch=sm_90a", "-O3", "--std=c++17",
                   "-shared", "-Xcompiler", "-fPIC",
                   "-I" + _CUTLASS_INC,
                   "-I" + _CUTLASS_UTIL_INC,
                   "--expt-relaxed-constexpr",
                   "--diag-suppress=20012,20011",
                   _CU, "-o", _SO]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"nvcc failed:\n{r.stderr}")
            print("[cutlass] done", flush=True)
        lib = ctypes.CDLL(_SO)
        for cfg in _CONFIGS:
            fn = getattr(lib, _kname(*cfg))
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_int, ctypes.c_int, ctypes.c_int]
            fn.restype  = ctypes.c_int
        lib.cutlass_free_workspace.argtypes = []
        lib.cutlass_free_workspace.restype  = None
        atexit.register(lib.cutlass_free_workspace)
        _lib = lib
        return _lib


# Cache output buffers by (M, N) so the C++ side's per-config Gemm-state
# cache can stay hot across bench iterations (same C ptr = no re-initialize).
_C_cache: dict = {}


def _get_C(M, N):
    key = (M, N)
    C = _C_cache.get(key)
    if C is None:
        C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        _C_cache[key] = C
    return C


def _launch(cfg, A, B):
    M, K = A.shape
    _, N = B.shape
    C = _get_C(M, N)
    lib = _get_lib()
    fn = getattr(lib, _kname(*cfg))
    rc = fn(
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(B.data_ptr()),
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
    )
    if rc != 0:
        raise RuntimeError(f"{_kname(*cfg)} returned {rc}")
    return C


_best: dict = {}


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    cfgs = [c for c in _CONFIGS
            if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0]
    best_t, best = float("inf"), cfgs[0]
    for cfg in cfgs:
        try:
            ms_med, _, _ = triton.testing.do_bench(
                lambda cfg=cfg: _launch(cfg, A, B),
                warmup=10, rep=100, quantiles=(0.5, 0.0, 1.0))
        except Exception as e:
            print(f"  cutlass {cfg} FAILED: {e}"); continue
        tf = 2 * M * N * K / (ms_med / 1e3) / 1e12
        bm, bn, bk, cx, cy = cfg
        print(f"  BM={bm:3d} BN={bn:3d} BK={bk:2d} CX={cx} CY={cy}  {tf:6.1f} TFLOPS")
        if ms_med < best_t:
            best_t, best = ms_med, cfg
    return best


def matmul_cutlass_bf16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """C = A @ B, all bfloat16, row-major. Autotunes over CUTLASS configs."""
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    _, N = B.shape
    key = (M, N, K)
    if key not in _best:
        print(f"[cutlass] autotuning {M}×{N}×{K} over {len(_CONFIGS)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, cx, cy = _best[key]
        print(f"[cutlass] best: BM={bm} BN={bn} BK={bk} CX={cx} CY={cy}")
    return _launch(_best[key], A, B)
