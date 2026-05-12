"""H3: h1_ms with wgmma replacing mma.sync.

Changes vs h1_ms:
  Compute  : ldmatrix_x2_trans + mma.sync → wgmma.mma_async m64nBNk16
  B swizzle: XOR removed from cp.async writes; B stored linearly;
             wgmma reads via INTERLEAVE (no-swizzle) descriptor
  A swizzle: unchanged — same XOR, same ldmatrix_x4
  Thread block: (32,NW,1) → (NW*32,1,1); BM = (NW//4)*64

Multi-stage pipeline is identical to h1_ms (NS ∈ {2,3,4,5}).
"""

import os, subprocess, threading
import numpy as np, torch, triton.testing
import pycuda.driver as drv

from ..tensor_core.matmul_cuda_tc5_regpruned import _LB_FOR_NW
from .._pycuda_loader import SM_ARCH, _ensure_ctx

# ── Compilation ───────────────────────────────────────────────────────────────

_HOPPER_DIR = os.path.dirname(os.path.abspath(__file__))
_CU_PATH    = os.path.join(_HOPPER_DIR, "_matmul_h3.cu")
_NVCC       = "/usr/local/cuda/bin/nvcc"
_ARCH     = "sm_90a"   # wgmma requires sm_90a
_MAX_SMEM = 200 * 1024
_CUBIN    = os.path.join(_HOPPER_DIR, f"_matmul_h3_{_ARCH}.cubin")

_mod_lock = threading.Lock()
_module   = None


def _get_mod() -> drv.Module:
    global _module
    with _mod_lock:
        if _module is not None:
            return _module
        _ensure_ctx()
        if not os.path.exists(_CUBIN) or os.path.getmtime(_CU_PATH) > os.path.getmtime(_CUBIN):
            print(f"[h3] compiling for {_ARCH} ...", end=" ", flush=True)
            cmd = [_NVCC, f"-arch={_ARCH}", "-O3", "--std=c++17", "--cubin",
                   "-DLB_MIN_BLOCKS=1", _CU_PATH, "-o", _CUBIN]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"nvcc failed:\n{r.stderr}")
            print("done")
        _module = drv.module_from_file(_CUBIN)
    return _module

# ── Config space ──────────────────────────────────────────────────────────────

_BNS = [64, 128, 256]
_BKS = [16, 32, 64]
_NWS = [4]      # NW=4 → 1 warpgroup BM=64
_NSS = [2, 3, 4, 5]

DTYPE = torch.bfloat16


def _bm(nw):   return (nw // 4) * 64

def _smem(bn, bk, nw, ns):
    bm = _bm(nw)
    return (ns * bm * bk + ns * bk * bn) * 2

def _reg_estimate(bn, bk):
    # wgmma accumulators (BN/2) + A frags (4 per kk) + misc
    return bn // 2 + 4 * (bk // 16) + 20


# Single cubin with LB=1 — no LB axis in config (same as h2_s2/h2_s3).
_CONFIGS = [
    (bn, bk, nw, ns)
    for bn in _BNS for bk in _BKS for nw in _NWS for ns in _NSS
    if _smem(bn, bk, nw, ns) <= _MAX_SMEM
]


def _kname(bn, bk, nw, ns):
    return f"matmul_h3_bn{bn}_bk{bk}_nw{nw}_ns{ns}"

# ── Launch ────────────────────────────────────────────────────────────────────

def _block(nw): return (nw * 32, 1, 1)

def _grid(M, N, bn, nw):
    bm = _bm(nw)
    return ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)


def _launch(mod, kname, A, B, nw, bn, bk, ns):
    M, K = A.shape;  _, N = B.shape
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    sb = _smem(bn, bk, nw, ns)
    fn = mod.get_function(kname)
    if sb > 0:
        fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, sb)
    fn(np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(C.data_ptr()),
       np.int32(M), np.int32(K), np.int32(N),
       block=_block(nw), grid=_grid(M, N, bn, nw), shared=sb)
    return C


def _launch_by_name(kname: str, M: int, N: int, K: int) -> torch.Tensor:
    """Launch a specific named kernel directly — used by profilers."""
    cfg = next((c for c in _CONFIGS if _kname(*c) == kname), None)
    if cfg is None:
        raise ValueError(f"Kernel {kname!r} not found in _CONFIGS")
    bn, bk, nw, ns = cfg
    A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    B = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    return _launch(_get_mod(), kname, A, B, nw, bn, bk, ns)

# ── Autotuner ─────────────────────────────────────────────────────────────────

_best: dict = {}


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    mod = _get_mod()

    cfgs = [c for c in _CONFIGS
            if M % _bm(c[2]) == 0 and N % c[0] == 0 and K % c[1] == 0
            and K // c[1] >= c[3]]   # num_tiles >= NS

    best_t, best_cfg = float("inf"), cfgs[0]
    for idx, (bn, bk, nw, ns) in enumerate(cfgs):
        kn = _kname(bn, bk, nw, ns)
        try:
            _, ms, _ = triton.testing.do_bench(
                lambda bn=bn,bk=bk,nw=nw,ns=ns,kn=kn:
                    _launch(mod, kn, A, B, nw, bn, bk, ns),
                warmup=10, rep=50, quantiles=(0.5, 0.0, 1.0))
        except Exception as e:
            print(f"  [{idx+1}/{len(cfgs)}] {kn} FAILED: {e}"); continue
        tf = 2 * M * N * K / (ms / 1e3) / 1e12
        print(f"  [{idx+1:3d}/{len(cfgs)}] BN={bn:3d} BK={bk:2d} NW={nw} NS={ns}  {tf:6.1f} TFLOPS")
        if ms < best_t:
            best_t, best_cfg = ms, (bn, bk, nw, ns)
    return best_cfg


def matmul_h3(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape;  _, N = B.shape
    key = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS
                if M % _bm(c[2]) == 0 and N % c[0] == 0 and K % c[1] == 0
                and K // c[1] >= c[3]]
        print(f"[h3] autotuning {M}×{N}×{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bn, bk, nw, ns = _best[key]
        print(f"[h3] best: BN={bn} BK={bk} NW={nw} NS={ns} (BM={_bm(nw)})")

    bn, bk, nw, ns = _best[key]
    return _launch(_get_mod(), _kname(bn, bk, nw, ns), A, B, nw, bn, bk, ns)
