"""H1 multi-stage: tc5_lb with NUM_STAGES ∈ {2,3,4,5} as a tunable axis.

NS=2 is identical to tc5_regpruned. Larger NS hides more cp.async latency at
the cost of more SMEM (NS × tile bytes instead of 2 × tile bytes).

Config space adds NS on top of tc5_regpruned's (BM, BN, BK, NW, LB):
  SMEM constraint:  NS × (BM×BK + BK×BN) × 2 ≤ MAX_SMEM
  Register budget:  unchanged (same mma.sync path)
  K constraint:     K/BK ≥ NS  (enforced at launch time)
"""

import os, subprocess, threading, atexit
import numpy as np, torch, triton.testing
import pycuda.driver as drv

from ..tensor_core.matmul_cuda_tc5_regpruned import (
    _reg_estimate, _LB_FOR_NW, _block, _grid
)
from .._pycuda_loader import SM_ARCH, _ensure_ctx

# ── Compilation ───────────────────────────────────────────────────────────────

_HOPPER_DIR = os.path.dirname(os.path.abspath(__file__))
_CU_PATH    = os.path.join(_HOPPER_DIR, "_matmul_h1_ms.cu")
_NVCC       = "/usr/local/cuda/bin/nvcc"

# H800 has 228 KB SMEM/SM; we leave headroom for the runtime and set 200 KB.
_MAX_SMEM = 200 * 1024

_mod_lock = threading.Lock()
_modules: dict = {}   # lb → drv.Module


def _cubin_path(lb: int) -> str:
    return os.path.join(_HOPPER_DIR, f"_matmul_h1ms_lb{lb}_{SM_ARCH}.cubin")


def _get_mod(lb: int) -> drv.Module:
    with _mod_lock:
        if lb not in _modules:
            _ensure_ctx()
            cubin = _cubin_path(lb)
            if not os.path.exists(cubin) or os.path.getmtime(_CU_PATH) > os.path.getmtime(cubin):
                print(f"[h1ms] compiling LB={lb} ...", end=" ", flush=True)
                cmd = [_NVCC, f"-arch={SM_ARCH}", "-O3", "--std=c++17", "--cubin",
                       f"-DLB_MIN_BLOCKS={lb}", _CU_PATH, "-o", cubin]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"nvcc failed:\n{r.stderr}")
                print("done")
            _modules[lb] = drv.module_from_file(cubin)
    return _modules[lb]

# ── Config space ──────────────────────────────────────────────────────────────

_BMS = [64, 128, 256]
_BNS = [64, 128, 256]
_BKS = [16, 32, 64]
_NWS = [4, 8]
_NSS = [2, 3, 4, 5]


def _smem(bm, bn, bk, ns):
    return (ns * bm * bk + ns * bk * bn) * 2


_CONFIGS = [
    (bm, bn, bk, nw, lb, ns)
    for bm in _BMS for bn in _BNS for bk in _BKS
    for nw in _NWS for ns in _NSS
    for lb in _LB_FOR_NW[nw]
    if _smem(bm, bn, bk, ns) <= _MAX_SMEM
    and bm * bn <= 4096 * nw
    and _reg_estimate(bm, bn, bk, nw) <= 65536 // (nw * 32 * lb)
]


def _kname(bm, bn, bk, nw, ns):
    return f"matmul_h1ms_bm{bm}_bn{bn}_bk{bk}_nw{nw}_ns{ns}"


def _smem_val(bm, bn, bk, ns):
    return _smem(bm, bn, bk, ns)

# ── Launch ────────────────────────────────────────────────────────────────────

DTYPE = torch.bfloat16


def _launch(mod, kname, A, B, block, grid, smem_bytes):
    M, K = A.shape;  _, N = B.shape
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    fn = mod.get_function(kname)
    if smem_bytes > 0:
        fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes)
    fn(np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(C.data_ptr()),
       np.int32(M), np.int32(K), np.int32(N),
       block=block, grid=grid, shared=smem_bytes)
    return C

# ── Autotuner ─────────────────────────────────────────────────────────────────

_best: dict = {}


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    all_lbs = sorted({lb for *_, lb, _ in _CONFIGS})
    mods = {lb: _get_mod(lb) for lb in all_lbs}

    cfgs = [c for c in _CONFIGS
            if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0
            and K // c[2] >= c[5]]   # num_tiles >= NS

    best_t, best_cfg = float("inf"), cfgs[0]
    for idx, (bm, bn, bk, nw, lb, ns) in enumerate(cfgs):
        kn = _kname(bm, bn, bk, nw, ns)
        sb = _smem_val(bm, bn, bk, ns)
        try:
            _, ms, _ = triton.testing.do_bench(
                lambda bm=bm,bn=bn,bk=bk,nw=nw,lb=lb,ns=ns,kn=kn:
                    _launch(mods[lb], kn, A, B, _block(nw), _grid(M, N, bm, bn), sb),
                warmup=10, rep=50, quantiles=(0.5, 0.0, 1.0))
        except Exception as e:
            print(f"  [{idx+1}/{len(cfgs)}] {kn} LB={lb} FAILED: {e}"); continue
        tf = 2 * M * N * K / (ms / 1e3) / 1e12
        print(f"  [{idx+1:3d}/{len(cfgs)}] BM={bm:3d} BN={bn:3d} BK={bk:2d} "
              f"NW={nw} LB={lb} NS={ns}  {tf:6.1f} TFLOPS")
        if ms < best_t:
            best_t, best_cfg = ms, (bm, bn, bk, nw, lb, ns)
    return best_cfg


def matmul_h1_ms(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape;  _, N = B.shape
    key = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS
                if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0
                and K // c[2] >= c[5]]
        print(f"[h1_ms] autotuning {M}×{N}×{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, nw, lb, ns = _best[key]
        print(f"[h1_ms] best: BM={bm} BN={bn} BK={bk} NW={nw} LB={lb} NS={ns}")

    bm, bn, bk, nw, lb, ns = _best[key]
    return _launch(_get_mod(lb), _kname(bm, bn, bk, nw, ns), A, B,
                   _block(nw), _grid(M, N, bm, bn), _smem_val(bm, bn, bk, ns))
