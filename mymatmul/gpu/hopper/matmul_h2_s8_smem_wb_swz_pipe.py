"""H2 Stage 7: h2_s6 + wgmma.wait_group 1 (overlap wgmma with next tile load).

Key change vs h2_s6:
  Instead of draining wgmma fully (wait_group 0) after every tile, use
  wait_group 1: keep 1 wgmma group in flight while loading the next tile.
  Hardware serialises acc[] writes across groups automatically.
  ISSUE placed after wait_group 1 (not before compute) for SMEM slot safety.
  Derived from Triton PTX analysis (notes-hopper/triton_ptx_analysis.md).
"""

import ctypes, os, subprocess, threading, atexit
import numpy as np, torch, triton.testing
from cuda.bindings import driver as cudrvr

# ── Context ───────────────────────────────────────────────────────────────────

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
_CU_PATH    = os.path.join(_HOPPER_DIR, "_matmul_h2_s8_smem_wb_swz_pipe.cu")
_NVCC       = "/usr/local/cuda/bin/nvcc"
_ARCH       = "sm_90a"
_CUBIN      = os.path.join(_HOPPER_DIR, f"_matmul_h2_s8_smem_wb_swz_pipe_{_ARCH}.cubin")

_mod_lock = threading.Lock()
_module   = None

def _get_mod():
    global _module
    with _mod_lock:
        if _module is not None: return _module
        _ensure_ctx()
        if not os.path.exists(_CUBIN) or os.path.getmtime(_CU_PATH) > os.path.getmtime(_CUBIN):
            print(f"[h2_s8_smem_wb_swz_pipe] compiling for {_ARCH} ...", end=" ", flush=True)
            cmd = [_NVCC, f"-arch={_ARCH}", "-O3", "--std=c++17", "--cubin",
                   "-DLB_MIN_BLOCKS=1", _CU_PATH, "-o", _CUBIN]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0: raise RuntimeError(f"nvcc failed:\n{r.stderr}")
            print("done")
        err, mod = cudrvr.cuModuleLoad(_CUBIN.encode()); _chk(err, "cuModuleLoad")
        _module = mod
    return _module

# ── Config space ──────────────────────────────────────────────────────────────

_BMS = [64, 128, 256]
_BNS = [64, 128, 256]
_BKS = [32, 64]  # BK=16 never autotunes as best — dropped to shrink search
_NWS = [1, 2]
_NSS = [2, 3, 4, 5]
_GMS = [1, 2, 4, 8]
_MAX_SMEM = 200 * 1024

DTYPE = torch.bfloat16


def _m_iters(bm, nwg): return bm // (nwg * 64)

def _smem(bm, bn, bk, ns):
    ab = (ns * bm * bk + ns * bk * bn) * 2
    c  = bm * (bn + 8) * 2
    return max(ab, c)

def _reg_estimate(bm, bn, bk, nwg):
    return _m_iters(bm, nwg) * (bn // 2) + 20


_CONFIGS = [
    (bm, bn, bk, nwg, ns, gm)
    for bm in _BMS for bn in _BNS for bk in _BKS
    for nwg in _NWS for ns in _NSS for gm in _GMS
    if bm % (nwg * 64) == 0
    and _smem(bm, bn, bk, ns) <= _MAX_SMEM
    and _reg_estimate(bm, bn, bk, nwg) <= 512
]


def _kname(bm, bn, bk, nwg, ns, gm):
    return f"matmul_h2_s8_smem_wb_swz_pipe_bm{bm}_bn{bn}_bk{bk}_wg{nwg}_ns{ns}_gm{gm}"

# ── Launch ────────────────────────────────────────────────────────────────────

def _get_fn(mod, kname):
    err, fn = cudrvr.cuModuleGetFunction(mod, kname.encode())
    _chk(err, f"cuModuleGetFunction({kname})")
    return fn

def _set_max_smem(fn, sb):
    (err,) = cudrvr.cuFuncSetAttribute(
        fn, cudrvr.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sb)
    _chk(err, "cuFuncSetAttribute")


def _launch(mod, kname, A, B, bm, bn, bk, nwg, ns):
    M, K = A.shape;  _, N = B.shape
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    fn = _get_fn(mod, kname)
    sb = _smem(bm, bn, bk, ns)
    _set_max_smem(fn, sb)
    c_A = ctypes.c_void_p(A.data_ptr())
    c_B = ctypes.c_void_p(B.data_ptr())
    c_C = ctypes.c_void_p(C.data_ptr())
    c_M, c_K, c_N = ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N)
    params = np.array([ctypes.addressof(x) for x in
                       [c_A, c_B, c_C, c_M, c_K, c_N]], dtype=np.intp)
    grid = ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)
    (err,) = cudrvr.cuLaunchKernel(fn, *grid, nwg * 128, 1, 1, sb, 0, params, 0)
    _chk(err, f"cuLaunchKernel({kname})")
    return C


def _launch_by_name(kname: str, M: int, N: int, K: int) -> torch.Tensor:
    cfg = next((c for c in _CONFIGS if _kname(*c) == kname), None)
    if cfg is None:
        raise ValueError(f"Kernel {kname!r} not found in _CONFIGS")
    bm, bn, bk, nwg, ns, _gm = cfg
    A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    B = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    return _launch(_get_mod(), kname, A, B, bm, bn, bk, nwg, ns)

# ── Autotuner ─────────────────────────────────────────────────────────────────

_best: dict = {}


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    mod = _get_mod()
    cfgs = [c for c in _CONFIGS
            if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0
            and K // c[2] >= c[4]
            # GROUP_M swizzle requires gridDim.y divisible by GROUP_M
            and (M // c[0]) % c[5] == 0]
    best_t, best = float("inf"), cfgs[0]
    for i, (bm, bn, bk, nwg, ns, gm) in enumerate(cfgs):
        kn = _kname(bm, bn, bk, nwg, ns, gm)
        try:
            # Score by median (less noise-sensitive than min); rep 50→100 ms.
            ms_med, _, _ = triton.testing.do_bench(
                lambda bm=bm, bn=bn, bk=bk, nwg=nwg, ns=ns, kn=kn:
                    _launch(mod, kn, A, B, bm, bn, bk, nwg, ns),
                warmup=10, rep=100, quantiles=(0.5, 0.0, 1.0))
        except Exception as e:
            print(f"  [{i+1}/{len(cfgs)}] {kn} FAILED: {e}"); continue
        tf = 2 * M * N * K / (ms_med / 1e3) / 1e12
        mi = _m_iters(bm, nwg)
        print(f"  [{i+1:3d}/{len(cfgs)}] BM={bm:3d} BN={bn:3d} BK={bk:2d} "
              f"WG={nwg} NS={ns} GM={gm} M_ITERS={mi}  {tf:6.1f} TFLOPS")
        if ms_med < best_t:
            best_t, best = ms_med, (bm, bn, bk, nwg, ns, gm)
    return best


def matmul_h2_s8_smem_wb_swz_pipe(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape;  _, N = B.shape
    key = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS
                if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0
                and K // c[2] >= c[4] and (M // c[0]) % c[5] == 0]
        print(f"[h2_s8_smem_wb_swz_pipe] autotuning {M}×{N}×{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, nwg, ns, gm = _best[key]
        mi = _m_iters(bm, nwg)
        print(f"[h2_s8_smem_wb_swz_pipe] best: BM={bm} BN={bn} BK={bk} WG={nwg} NS={ns} GM={gm} M_ITERS={mi}")
    bm, bn, bk, nwg, ns, gm = _best[key]
    return _launch(_get_mod(), _kname(bm, bn, bk, nwg, ns, gm), A, B, bm, bn, bk, nwg, ns)
