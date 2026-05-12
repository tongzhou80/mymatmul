"""H2 Stage 4: Stage 3 + M-dimension loop per warpgroup (larger BM tiles).

Key change vs H2-S3:
  BM is now a free parameter (not locked to NUM_WG*64).
  M_ITERS = BM / (NUM_WG * 64): each warpgroup issues M_ITERS wgmma calls
  per kk step, each covering the next 64 M-rows.

  BM=256, NUM_WG=2 → M_ITERS=2: 2× arithmetic intensity vs h2_s3 (BM=128)
  for the same BN/BK, using ~128 KB SMEM vs ~96 KB.
"""

import ctypes, os, subprocess, threading, atexit
import numpy as np, torch, triton.testing
from cuda.bindings import driver as cudrvr

# ── Context (identical to h2_s3) ─────────────────────────────────────────────

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
_CU_PATH    = os.path.join(_HOPPER_DIR, "_matmul_h2_s4.cu")
_NVCC       = "/usr/local/cuda/bin/nvcc"
_ARCH       = "sm_90a"
_CUBIN      = os.path.join(_HOPPER_DIR, f"_matmul_h2s4_{_ARCH}.cubin")

_mod_lock = threading.Lock()
_module   = None

def _get_mod():
    global _module
    with _mod_lock:
        if _module is not None: return _module
        _ensure_ctx()
        if not os.path.exists(_CUBIN) or os.path.getmtime(_CU_PATH) > os.path.getmtime(_CUBIN):
            print(f"[h2s4] compiling for {_ARCH} ...", end=" ", flush=True)
            cmd = [_NVCC, f"-arch={_ARCH}", "-O3", "--std=c++17", "--cubin",
                   "-DLB_MIN_BLOCKS=1", _CU_PATH, "-o", _CUBIN]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0: raise RuntimeError(f"nvcc failed:\n{r.stderr}")
            print("done")
        err, mod = cudrvr.cuModuleLoad(_CUBIN.encode()); _chk(err, "cuModuleLoad")
        _module = mod
    return _module

# ── TMA descriptors (same helpers as h2_s3) ───────────────────────────────────

_DTYPE_BF16      = cudrvr.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
_SWIZZLE_NONE    = cudrvr.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
_SWIZZLE_128B    = cudrvr.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
_INTERLEAVE_NONE = cudrvr.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE
_L2_PROMO_NONE   = cudrvr.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE
_OOB_FILL_NONE   = cudrvr.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE

def _a_swizzle(bk):
    return _SWIZZLE_128B if bk == 64 else _SWIZZLE_NONE

def _make_tma_desc(ptr_int, nrows, ncols, box_rows, box_cols, swizzle):
    u32, u64 = cudrvr.cuuint32_t, cudrvr.cuuint64_t
    err, tmap = cudrvr.cuTensorMapEncodeTiled(
        _DTYPE_BF16, 2, ptr_int,
        [u64(ncols), u64(nrows)], [u64(ncols * 2)],
        [u32(box_cols), u32(box_rows)], [u32(1), u32(1)],
        _INTERLEAVE_NONE, swizzle, _L2_PROMO_NONE, _OOB_FILL_NONE)
    _chk(err, "cuTensorMapEncodeTiled")
    return np.array([int(v) for v in tmap.opaque], dtype=np.uint64).tobytes()

# ── Config space ──────────────────────────────────────────────────────────────

_BMS  = [64, 128, 256]   # NEW: BM can now be 128 or 256 (M_ITERS > 1)
_BNS  = [64, 128, 256]
_BKS  = [16, 32, 64]
_NWS  = [1, 2]
_MAX_SMEM = 200 * 1024

DTYPE = torch.bfloat16


def _m_iters(bm, nwg): return bm // (nwg * 64)

def _smem(bm, bn, bk):
    return (2 * bm * bk + 2 * bk * bn) * 2 + 16   # 2-stage + 2 mbarriers

def _reg_estimate(bm, bn, bk, nwg):
    mi = _m_iters(bm, nwg)
    return mi * (bn // 2) + 20               # acc[M_ITERS][BN/2] + misc


_CONFIGS = [
    (bm, bn, bk, nwg)
    for bm in _BMS for bn in _BNS for bk in _BKS for nwg in _NWS
    if bm % (nwg * 64) == 0                         # M_ITERS is integer
    and _smem(bm, bn, bk) <= _MAX_SMEM
    and _reg_estimate(bm, bn, bk, nwg) <= 512        # fits in 512 regs (LB=1, 128 threads)
]


def _kname(bm, bn, bk, nwg):
    return f"matmul_h2s4_bm{bm}_bn{bn}_bk{bk}_wg{nwg}"

# ── Launch ────────────────────────────────────────────────────────────────────

def _get_fn(mod, kname):
    err, fn = cudrvr.cuModuleGetFunction(mod, kname.encode())
    _chk(err, f"cuModuleGetFunction({kname})")
    return fn

def _set_max_smem(fn, sb):
    (err,) = cudrvr.cuFuncSetAttribute(
        fn, cudrvr.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sb)
    _chk(err, "cuFuncSetAttribute")


def _launch(mod, kname, A, B, bm, bn, bk, nwg):
    M, K = A.shape;  _, N = B.shape
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    fn = _get_fn(mod, kname)
    sb = _smem(bm, bn, bk)
    _set_max_smem(fn, sb)
    # A: boxDim = [bk, bm] — bm is now the full M tile (may be 256)
    # B: boxDim = [64, bk] — 128B swizzle, BN/64 sub-tiles per stage
    buf_a = ctypes.create_string_buffer(
        _make_tma_desc(A.data_ptr(), M, K, bm, bk, _a_swizzle(bk)), 128)
    buf_b = ctypes.create_string_buffer(
        _make_tma_desc(B.data_ptr(), K, N, bk, 64, _SWIZZLE_128B), 128)
    c_C = ctypes.c_void_p(C.data_ptr())
    c_M, c_K, c_N = ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N)
    params = np.array([ctypes.addressof(x) for x in
                       [buf_a, buf_b, c_C, c_M, c_K, c_N]], dtype=np.intp)
    grid = ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)
    (err,) = cudrvr.cuLaunchKernel(fn, *grid, nwg * 128, 1, 1, sb, 0, params, 0)
    _chk(err, f"cuLaunchKernel({kname})")
    return C


def _launch_by_name(kname: str, M: int, N: int, K: int) -> torch.Tensor:
    cfg = next((c for c in _CONFIGS if _kname(*c) == kname), None)
    if cfg is None:
        raise ValueError(f"Kernel {kname!r} not found in _CONFIGS")
    bm, bn, bk, nwg = cfg
    A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    B = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    return _launch(_get_mod(), kname, A, B, bm, bn, bk, nwg)

# ── Autotuner ─────────────────────────────────────────────────────────────────

_best: dict = {}


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    mod = _get_mod()
    cfgs = [(bm, bn, bk, nwg) for bm, bn, bk, nwg in _CONFIGS
            if M % bm == 0 and N % bn == 0 and K % bk == 0]
    best_t, best = float("inf"), cfgs[0]
    for i, (bm, bn, bk, nwg) in enumerate(cfgs):
        kn = _kname(bm, bn, bk, nwg)
        try:
            _, ms, _ = triton.testing.do_bench(
                lambda bm=bm,bn=bn,bk=bk,nwg=nwg,kn=kn:
                    _launch(mod, kn, A, B, bm, bn, bk, nwg),
                warmup=10, rep=50, quantiles=(0.5, 0.0, 1.0))
        except Exception as e:
            print(f"  [{i+1}/{len(cfgs)}] {kn} FAILED: {e}"); continue
        tf = 2 * M * N * K / (ms / 1e3) / 1e12
        mi = _m_iters(bm, nwg)
        print(f"  [{i+1:3d}/{len(cfgs)}] BM={bm:3d} BN={bn:3d} BK={bk:2d} "
              f"WG={nwg} M_ITERS={mi}  {tf:6.1f} TFLOPS")
        if ms < best_t:
            best_t, best = ms, (bm, bn, bk, nwg)
    return best


def matmul_h2_s4(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape;  _, N = B.shape
    key = (M, N, K)
    if key not in _best:
        cfgs = [(bm, bn, bk, nwg) for bm, bn, bk, nwg in _CONFIGS
                if M % bm == 0 and N % bn == 0 and K % bk == 0]
        print(f"[h2_s4] autotuning {M}×{N}×{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, nwg = _best[key]
        mi = _m_iters(bm, nwg)
        print(f"[h2_s4] best: BM={bm} BN={bn} BK={bk} WG={nwg} M_ITERS={mi}")
    bm, bn, bk, nwg = _best[key]
    return _launch(_get_mod(), _kname(bm, bn, bk, nwg), A, B, bm, bn, bk, nwg)
