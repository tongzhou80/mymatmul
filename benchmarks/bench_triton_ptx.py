"""Directly compile and benchmark the saved Triton BF16 PTX for N=4096.

Kernel: BLOCK_M=64, BLOCK_N=128, BLOCK_K=16, GROUP_M=8, num_warps=4, num_stages=5
Grid:   1-D, size = ceil(M/64) * ceil(N/128)
Block:  128 threads (4 warps)
Smem:   24576 bytes  (4 ring-buffer slots × 6144 bytes/slot)

Parameters passed to the kernel (11 total):
  param_0  u64  A pointer
  param_1  u64  B pointer
  param_2  u64  C pointer
  param_3  u32  M
  param_4  u32  N
  param_5  u32  K
  param_6  u32  stride_am  (= K for row-major)
  param_7  u32  stride_bk  (= N for row-major)
  param_8  u32  stride_cm  (= N for row-major)
  param_9  u64  unused (Triton internal, pass 0)
  param_10 u64  unused (Triton internal, pass 0)
"""

import os, struct, ctypes
import torch
import triton.testing
import pycuda.driver as drv
import pycuda.autoinit  # noqa: initialises CUDA context

PTX_FILE  = os.path.join(os.path.dirname(__file__),
                         "../triton_ptx/triton_bf16_4096_v2.ptx")
SMEM_BYTES = 24576
BLOCK_M, BLOCK_N = 64, 128


_CUBIN = os.path.join(os.path.dirname(__file__),
                      "../triton_ptx/triton_bf16_4096_v2.cubin")


def _ensure_cubin():
    """Compile PTX → cubin if needed (ptxas strips non-ASCII comments)."""
    ptx_path   = os.path.abspath(PTX_FILE)
    cubin_path = os.path.abspath(_CUBIN)
    if (not os.path.exists(cubin_path) or
            os.path.getmtime(ptx_path) > os.path.getmtime(cubin_path)):
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix=".ptx", mode="w", delete=False) as f:
            # replace the em-dash in the first comment line
            src = open(ptx_path).read().replace("—", "-")
            f.write(src)
            tmp = f.name
        ptxas = "/usr/local/cuda-12.8/bin/ptxas"
        subprocess.check_call([ptxas, "-arch=sm_89", "-o", cubin_path, tmp])
        os.unlink(tmp)


def _load_module():
    _ensure_cubin()
    mod = drv.module_from_file(os.path.abspath(_CUBIN))
    fn  = mod.get_function("_matmul_bf16_kernel")
    # allow up to 98 KB dynamic smem on sm_89
    fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, 100352)
    return fn


def _pack_params(A, B, C, M, N, K):
    """Pack the 11 kernel parameters into a raw byte buffer."""
    # pycuda accepts a list of items; each is a ctypes scalar or numpy scalar.
    # We use ctypes.c_uint64 / c_uint32 so pycuda passes them by value.
    a_ptr = ctypes.c_uint64(A.data_ptr())
    b_ptr = ctypes.c_uint64(B.data_ptr())
    c_ptr = ctypes.c_uint64(C.data_ptr())
    m_    = ctypes.c_uint32(M)
    n_    = ctypes.c_uint32(N)
    k_    = ctypes.c_uint32(K)
    s_am  = ctypes.c_uint32(K)   # stride_am = K (row-major A)
    s_bk  = ctypes.c_uint32(N)   # stride_bk = N (row-major B)
    s_cm  = ctypes.c_uint32(N)   # stride_cm = N (row-major C)
    null  = ctypes.c_uint64(0)   # param_9  (unused)
    null2 = ctypes.c_uint64(0)   # param_10 (unused)
    return [a_ptr, b_ptr, c_ptr, m_, n_, k_, s_am, s_bk, s_cm, null, null2]


_fn = None


def matmul_triton_ptx(A, B):
    global _fn
    if _fn is None:
        _fn = _load_module()

    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    grid_x = (M // BLOCK_M) * (N // BLOCK_N)
    params  = _pack_params(A, B, C, M, N, K)

    _fn(*params,
        block=(128, 1, 1),
        grid=(grid_x, 1, 1),
        shared=SMEM_BYTES)

    return C


def validate(fn, A, B):
    C_ref = torch.mm(A.float(), B.float()).bfloat16()
    C_out = fn(A, B)
    max_err = (C_ref - C_out).abs().max().item()
    assert max_err < 2.0, f"max error {max_err}"
    print(f"  validation OK  (max err {max_err:.4f})")


if __name__ == "__main__":
    M = N = K = 4096
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    print(f"[triton_ptx] validating {M}x{K}x{N} ...")
    validate(matmul_triton_ptx, A, B)

    ms_med, ms_min, _ = triton.testing.do_bench(
        lambda: matmul_triton_ptx(A, B),
        warmup=100, rep=500, quantiles=(0.5, 0.0, 1.0),
    )
    tflops = 2 * M * N * K / ms_min * 1e-9
    print(f"  {M}x{K}x{N}: {tflops:.1f} TFLOPS  "
          f"(median {ms_med:.3f} ms, best {ms_min:.3f} ms)")
