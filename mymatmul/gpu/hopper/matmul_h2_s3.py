"""H2 Stage 3: Stage 2 + multi-warpgroup (BM = NUM_WG * 64).

Change vs H2-S2:
  BM / threads : fixed 64 / 128 → NUM_WG*64 / NUM_WG*128  (NUM_WG ∈ {1, 2})

With 2 warpgroups both share B tile; each owns 64 rows of A/C independently.
"""

import ctypes, os, subprocess, threading, atexit

import numpy as np
import torch
import triton.testing

from cuda.bindings import driver as cudrvr

# ── Shared CUDA context ───────────────────────────────────────────────────────

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

# ── Compilation ───────────────────────────────────────────────────────────────

_HOPPER_DIR = os.path.dirname(os.path.abspath(__file__))
_CU_PATH    = os.path.join(_HOPPER_DIR, "_matmul_h2_s3.cu")
_NVCC       = "/usr/local/cuda/bin/nvcc"
_ARCH       = "sm_90a"
_CUBIN      = os.path.join(_HOPPER_DIR, f"_matmul_h2s3_{_ARCH}.cubin")

_mod_lock = threading.Lock()
_module   = None

def _get_mod():
    global _module
    with _mod_lock:
        if _module is not None: return _module
        _ensure_ctx()
        if not os.path.exists(_CUBIN) or os.path.getmtime(_CU_PATH) > os.path.getmtime(_CUBIN):
            print(f"[h2s3] compiling for {_ARCH} ...", end=" ", flush=True)
            cmd = [_NVCC, f"-arch={_ARCH}", "-O3", "--std=c++17", "--cubin",
                   "-DLB_MIN_BLOCKS=1", _CU_PATH, "-o", _CUBIN]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0: raise RuntimeError(f"nvcc failed:\n{r.stderr}")
            print("done")
        err, mod = cudrvr.cuModuleLoad(_CUBIN.encode()); _chk(err, "cuModuleLoad")
        _module = mod
    return _module

# ── TMA descriptor ────────────────────────────────────────────────────────────

_DTYPE_BF16      = cudrvr.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
_SWIZZLE_NONE    = cudrvr.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
_SWIZZLE_32B     = cudrvr.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B
_SWIZZLE_64B     = cudrvr.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B
_SWIZZLE_128B    = cudrvr.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B

def _a_swizzle(bk):
    """TMA swizzle for A[BM][BK].
    Only BK=64 (128B rows) is compatible with ldmatrix.x4 which reads 16-byte chunks:
    128B swizzle XORs in 16-byte multiples → chunk boundaries preserved → correct.
    64B/32B swizzle XORs at 8-byte granularity → splits 16-byte ldmatrix chunks → garbage.
    For BK=16/32, use NONE (bank conflicts remain; swizzle is addressed in a later stage).
    """
    if bk == 64: return _SWIZZLE_128B
    return _SWIZZLE_NONE
_INTERLEAVE_NONE = cudrvr.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE
_L2_PROMO_NONE   = cudrvr.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE
_OOB_FILL_NONE   = cudrvr.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE

def _make_tma_desc(ptr_int, nrows, ncols, box_rows, box_cols, swizzle):
    u32, u64 = cudrvr.cuuint32_t, cudrvr.cuuint64_t
    err, tmap = cudrvr.cuTensorMapEncodeTiled(
        _DTYPE_BF16, 2, ptr_int,
        [u64(ncols), u64(nrows)],
        [u64(ncols * 2)],
        [u32(box_cols), u32(box_rows)],
        [u32(1), u32(1)],
        _INTERLEAVE_NONE, swizzle, _L2_PROMO_NONE, _OOB_FILL_NONE,
    )
    _chk(err, "cuTensorMapEncodeTiled")
    return np.array([int(v) for v in tmap.opaque], dtype=np.uint64).tobytes()

# ── Config space ──────────────────────────────────────────────────────────────

_NWG_S = [1, 2]
_BNS   = [64, 128, 256]
_BKS   = [16, 32, 64]
_MAX_SMEM = 228 * 1024

def _bm(nwg):   return nwg * 64

def _smem(nwg, bn, bk):
    a = 2 * _bm(nwg) * bk * 2
    b = 2 * bk * bn * 2
    return a + b + 16

def _reg_estimate(bn, bk):
    return bn // 2 + 4 * (bk // 16)

_CONFIGS = [
    (nwg, bn, bk)
    for nwg in _NWG_S for bn in _BNS for bk in _BKS
    if _smem(nwg, bn, bk) <= _MAX_SMEM
    and _reg_estimate(bn, bk) <= 255
]

def _kname(nwg, bn, bk):
    return f"matmul_h2s3_wg{nwg}_bn{bn}_bk{bk}"

# ── Kernel launch ─────────────────────────────────────────────────────────────

def _get_fn(mod, kname):
    err, fn = cudrvr.cuModuleGetFunction(mod, kname.encode())
    _chk(err, f"cuModuleGetFunction({kname})")
    return fn

def _set_max_smem(fn, smem_bytes):
    (err,) = cudrvr.cuFuncSetAttribute(
        fn,
        cudrvr.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        smem_bytes)
    _chk(err, "cuFuncSetAttribute")


def _launch(mod, kname, A, B, nwg, bn, bk):
    M, K = A.shape;  _, N = B.shape
    bm = _bm(nwg)
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    fn = _get_fn(mod, kname)
    sb = _smem(nwg, bn, bk)
    _set_max_smem(fn, sb)

    buf_a = ctypes.create_string_buffer(
        _make_tma_desc(A.data_ptr(), M, K, bm, bk, _a_swizzle(bk)), 128)
    buf_b = ctypes.create_string_buffer(
        _make_tma_desc(B.data_ptr(), K, N, bk, 64, _SWIZZLE_128B), 128)
    c_C = ctypes.c_void_p(C.data_ptr())
    c_M = ctypes.c_int(M); c_K = ctypes.c_int(K); c_N = ctypes.c_int(N)
    params = np.array([
        ctypes.addressof(buf_a), ctypes.addressof(buf_b),
        ctypes.addressof(c_C),  ctypes.addressof(c_M),
        ctypes.addressof(c_K),  ctypes.addressof(c_N),
    ], dtype=np.intp)

    grid = ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)
    (err,) = cudrvr.cuLaunchKernel(fn, grid[0], grid[1], 1,
                                    nwg * 128, 1, 1, sb, 0, params, 0)
    _chk(err, f"cuLaunchKernel({kname})")
    return C

# ── Autotuner ─────────────────────────────────────────────────────────────────

_best: dict = {}
DTYPE = torch.bfloat16


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    mod = _get_mod()
    cfgs = [(nwg, bn, bk) for nwg, bn, bk in _CONFIGS
            if M % _bm(nwg) == 0 and N % bn == 0 and K % bk == 0]
    best_t, best = float("inf"), cfgs[0]
    for i, (nwg, bn, bk) in enumerate(cfgs):
        kn = _kname(nwg, bn, bk)
        try:
            _, ms, _ = triton.testing.do_bench(
                lambda nwg=nwg, bn=bn, bk=bk, kn=kn: _launch(mod, kn, A, B, nwg, bn, bk),
                warmup=10, rep=50, quantiles=(0.5, 0.0, 1.0))
        except Exception as e:
            print(f"  [{i+1}/{len(cfgs)}] {kn} FAILED: {e}"); continue
        tf = 2 * M * N * K / (ms / 1e3) / 1e12
        print(f"  [{i+1:2d}/{len(cfgs)}] WG={nwg} BN={bn:3d} BK={bk:2d}  {tf:6.1f} TFLOPS")
        if ms < best_t: best_t, best = ms, (nwg, bn, bk)
    return best


def matmul_h2_s3(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape;  _, N = B.shape
    key = (M, N, K)
    if key not in _best:
        cfgs = [(nwg, bn, bk) for nwg, bn, bk in _CONFIGS
                if M % _bm(nwg) == 0 and N % bn == 0 and K % bk == 0]
        print(f"[h2_s3] autotuning {M}×{N}×{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        nwg, bn, bk = _best[key]
        print(f"[h2_s3] best: WG={nwg} BN={bn} BK={bk}")
    nwg, bn, bk = _best[key]
    return _launch(_get_mod(), _kname(nwg, bn, bk), A, B, nwg, bn, bk)
