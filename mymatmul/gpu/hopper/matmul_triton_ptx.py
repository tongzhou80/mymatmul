"""Triton PTX wrapper: loads pre-compiled Triton BF16 matmul kernel.

Kernel: BM=128, BN=256, BK=32, NS=4, NW=8 (256 threads, 2 warpgroups).
  Uses cp.async + wgmma SS mode — the config Triton tuned for N≈8192.
  Works for any M, N, K that are multiples of 128/256/32 respectively.

Signature (from PTX):
  matmul_kernel(A*, B*, C*, M, N, K, stride_am, stride_bk, stride_cm, _unused*, _unused*)
  Grid: 1D, num_tiles_m * num_tiles_n CTAs (Triton's grouped swizzle maps ctaid.x → tile)
"""

import ctypes, math, os, subprocess, threading, atexit
import numpy as np, torch, triton.testing
from cuda.bindings import driver as cudrvr

# ── CUDA context ──────────────────────────────────────────────────────────────

_ctx_ready = False
_ctx_lock  = threading.Lock()

def _ensure_ctx():
    global _ctx_ready
    with _ctx_lock:
        if _ctx_ready: return
        torch.cuda.init()
        (err,) = cudrvr.cuInit(0); _chk(err, "cuInit")
        err, ctx = cudrvr.cuDevicePrimaryCtxRetain(torch.cuda.current_device())
        _chk(err, "cuDevicePrimaryCtxRetain")
        (err,) = cudrvr.cuCtxSetCurrent(ctx); _chk(err, "cuCtxSetCurrent")
        atexit.register(lambda: cudrvr.cuDevicePrimaryCtxRelease(torch.cuda.current_device()))
        _ctx_ready = True

def _chk(err, op=""):
    if err != cudrvr.CUresult.CUDA_SUCCESS:
        _, name = cudrvr.cuGetErrorName(err)
        _, desc = cudrvr.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error in {op}: {name.decode()} — {desc.decode()}")

# ── PTX → cubin compilation ───────────────────────────────────────────────────

_PTX_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "..", "triton_ptx")
_PTX_PATH = os.path.join(_PTX_DIR, "triton_bf16_bm128_bn256_bk32_ns4_nw8_n8192.ptx")
_CUBIN    = os.path.join(_PTX_DIR, "triton_bf16_bm128_bn256_bk32_ns4_nw8_n8192_sm90a.cubin")
_PTXAS    = "/usr/local/cuda/bin/ptxas"

_mod_lock = threading.Lock()
_module   = None
_fn       = None

# SMEM: BM*BK*NS + BN*BK*NS all in BF16 = (128*32 + 256*32) * 4 * 2 bytes
_SMEM_BYTES = (128 * 32 + 256 * 32) * 4 * 2   # = 98304 = 96 KB

def _get_fn():
    global _module, _fn
    with _mod_lock:
        if _fn is not None: return _fn
        _ensure_ctx()
        if not os.path.exists(_CUBIN) or os.path.getmtime(_PTX_PATH) > os.path.getmtime(_CUBIN):
            print("[triton_ptx] compiling PTX → cubin ...", end=" ", flush=True)
            cmd = [_PTXAS, "--gpu-name=sm_90a", "--output-file", _CUBIN, _PTX_PATH]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0: raise RuntimeError(f"ptxas failed:\n{r.stderr}")
            print("done")
        err, mod = cudrvr.cuModuleLoad(_CUBIN.encode()); _chk(err, "cuModuleLoad")
        _module = mod
        err, fn = cudrvr.cuModuleGetFunction(mod, b"matmul_kernel")
        _chk(err, "cuModuleGetFunction")
        (err,) = cudrvr.cuFuncSetAttribute(
            fn, cudrvr.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            _SMEM_BYTES)
        _chk(err, "cuFuncSetAttribute")
        _fn = fn
    return _fn

# ── Launch ────────────────────────────────────────────────────────────────────

def matmul_triton_ptx(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Run the pre-compiled Triton BF16 kernel. M/N/K must be multiples of 128/256/32."""
    M, K = A.shape;  _, N = B.shape
    assert M % 128 == 0 and N % 256 == 0 and K % 32 == 0, \
        f"M={M} must be %128, N={N} must be %256, K={K} must be %32"
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    fn = _get_fn()

    # 1D grid: Triton's grouped swizzle maps ctaid.x → (tile_m, tile_n)
    grid_x = math.ceil(M / 128) * math.ceil(N / 256)

    # Params: A*, B*, C*, M, N, K, stride_am(=K), stride_bk(=N), stride_cm(=N), null, null
    c_A   = ctypes.c_void_p(A.data_ptr())
    c_B   = ctypes.c_void_p(B.data_ptr())
    c_C   = ctypes.c_void_p(C.data_ptr())
    c_M   = ctypes.c_int(M)
    c_N   = ctypes.c_int(N)
    c_K   = ctypes.c_int(K)
    c_sam = ctypes.c_int(K)   # stride_am = K  (A is row-major [M][K])
    c_sbk = ctypes.c_int(N)   # stride_bk = N  (B is row-major [K][N])
    c_scm = ctypes.c_int(N)   # stride_cm = N  (C is row-major [M][N])
    c_p9  = ctypes.c_void_p(0)
    c_p10 = ctypes.c_void_p(0)

    params = np.array([ctypes.addressof(x) for x in
                       [c_A, c_B, c_C, c_M, c_N, c_K,
                        c_sam, c_sbk, c_scm, c_p9, c_p10]], dtype=np.intp)
    (err,) = cudrvr.cuLaunchKernel(fn, grid_x, 1, 1, 256, 1, 1,
                                   _SMEM_BYTES, 0, params, 0)
    _chk(err, "cuLaunchKernel(triton_ptx)")
    return C
