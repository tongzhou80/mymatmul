"""Lazy PyCUDA module loader for CUDA kernels.

Compiles .cu files with nvcc on first use and caches the resulting .cubin
next to the source file.  Shares PyTorch's primary CUDA context so tensors
and kernels can freely exchange device pointers.
"""

import atexit
import os
import subprocess
import threading

import torch
import pycuda.driver as drv

NVCC = "/usr/local/cuda/bin/nvcc"
SM_ARCH = "sm_89"   # RTX 4090 (Ada Lovelace)

_lock = threading.Lock()
_ctx: drv.Context | None = None
_modules: dict[str, drv.Module] = {}


def _pop_ctx() -> None:
    global _ctx
    if _ctx is not None:
        try:
            _ctx.pop()
        except Exception:
            pass
        _ctx = None


def _ensure_ctx() -> None:
    global _ctx
    if _ctx is not None:
        return
    torch.cuda.init()
    drv.init()
    _ctx = drv.Device(torch.cuda.current_device()).retain_primary_context()
    _ctx.push()
    atexit.register(_pop_ctx)


def _find_cu(ext_name: str) -> str:
    gpu_dir = os.path.dirname(os.path.abspath(__file__))
    for sub in ("cuda_core", "tensor_core"):
        path = os.path.join(gpu_dir, sub, f"{ext_name}.cu")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{ext_name}.cu not found under cuda_core/ or tensor_core/")


def _cubin_path(cu_path: str) -> str:
    return cu_path[:-3] + f"_{SM_ARCH}.cubin"


def _compile(cu_path: str, cubin: str) -> None:
    cmd = [NVCC, f"-arch={SM_ARCH}", "-O3", "--std=c++17", "--cubin",
           cu_path, "-o", cubin]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{r.stderr}")


def get_module(ext_name: str) -> drv.Module:
    with _lock:
        if ext_name in _modules:
            return _modules[ext_name]
        _ensure_ctx()
        cu_path = _find_cu(ext_name)
        cubin = _cubin_path(cu_path)
        if not os.path.exists(cubin) or os.path.getmtime(cu_path) > os.path.getmtime(cubin):
            print(f"[pycuda] compiling {os.path.basename(cu_path)} ...", end=" ", flush=True)
            _compile(cu_path, cubin)
            print("done")
        mod = drv.module_from_file(cubin)
        _modules[ext_name] = mod
        return mod


def get_kernel(ext_name: str, kernel_name: str) -> drv.Function:
    return get_module(ext_name).get_function(kernel_name)


def launch_matmul(ext_name: str, kernel_name: str, A, B,
                  block: tuple, grid: tuple, out_dtype=None):
    """Launch a PyCUDA matmul kernel and return a new output tensor.

    The kernel signature must be:
        (const float* A, const float* B, float* C, int M, int K, int N)
    or the bf16 variant for stage 5.

    block and grid are (x, y, z) tuples as required by PyCUDA.
    out_dtype defaults to A.dtype.
    """
    import numpy as np
    import torch
    M, K = A.shape
    _K2, N = B.shape
    dtype = out_dtype if out_dtype is not None else A.dtype
    C = torch.zeros((M, N), device="cuda", dtype=dtype)
    fn = get_kernel(ext_name, kernel_name)
    fn(np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(C.data_ptr()),
       np.int32(M), np.int32(K), np.int32(N),
       block=block, grid=grid)
    return C
