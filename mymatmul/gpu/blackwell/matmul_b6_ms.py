"""b6_ms: b5_tma + multi-stage TMA prefetch (NUM_STAGES tunable).

Autotunes over (BN, BK, NS) for fixed BM=128.
"""

import os
import numpy as np
import torch
import triton.testing
import pycuda.driver as drv

from .._pycuda_loader import get_module_jit, SM_ARCH
from . import _tma_utils as tma

DTYPE = torch.bfloat16
_GPU_DIR = os.path.dirname(os.path.abspath(__file__))
_CU_PATH = os.path.join(_GPU_DIR, "_matmul_b6_ms.cu")
_CUBIN   = os.path.join(_GPU_DIR, f"_matmul_b6_ms_{SM_ARCH}.cubin")

BM = 128
NW = 4
_BNS = [64, 128, 256]
_BKS = [64, 128, 256]
_NSS = [2, 3, 4, 5]

_MAX_SMEM = 200 * 1024


def _smem(bn, bk, ns):
    return ns * (BM + bn) * bk * 2


def _legal(bn, bk, ns):
    return _smem(bn, bk, ns) <= _MAX_SMEM


_CONFIGS = [(bn, bk, ns) for bn in _BNS for bk in _BKS for ns in _NSS if _legal(bn, bk, ns)]


def _get_mod():
    return get_module_jit(_CU_PATH, _CUBIN, ["-arch=sm_100a", "-DLB_MIN_BLOCKS=1"])


def _kname(bn, bk, ns):
    return f"matmul_b6_ms_bm{BM}_bn{bn}_bk{bk}_ns{ns}"


def _launch(mod, kname, A, B, bn, bk, smem_bytes):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device="cuda", dtype=DTYPE)

    # v2's convention: B is laid out as (N, K) row-major in memory.
    B_t = B.t().contiguous()
    A_tmap = tma.build_tma_2d(A.data_ptr(),   M, K, BM, 64, tma.SWIZZLE_128B)
    B_tmap = tma.build_tma_2d(B_t.data_ptr(), N, K, bn, 64, tma.SWIZZLE_128B)

    fn = mod.get_function(kname)
    if smem_bytes > 0:
        fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes)
    block = (NW * 32, 1, 1)
    grid = ((M // BM) * (N // bn), 1, 1)
    fn(A_tmap, B_tmap,
       np.intp(C.data_ptr()),
       np.int32(M), np.int32(N), np.int32(K),
       block=block, grid=grid, shared=smem_bytes)
    return C


_best: dict = {}


def _legal_for_problem(cfg, M, N, K):
    bn, bk, ns = cfg
    if M % BM != 0 or N % bn != 0 or K % bk != 0:
        return False
    if K // bk < ns:    # need at least NS K-tiles to fill the pipeline
        return False
    return True


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=DTYPE)
    B = torch.randn(K, N, device="cuda", dtype=DTYPE)
    mod = _get_mod()

    cfgs = [c for c in _CONFIGS if _legal_for_problem(c, M, N, K)]
    best_t = float("inf")
    best_cfg = cfgs[0]
    n = len(cfgs)
    for idx, cfg in enumerate(cfgs):
        bn, bk, ns = cfg
        kn = _kname(bn, bk, ns)
        sb = _smem(bn, bk, ns)
        try:
            ms_med, _, _ = triton.testing.do_bench(
                lambda bn=bn, bk=bk, ns=ns, kn=kn, sb=sb:
                    _launch(mod, kn, A, B, bn, bk, sb),
                warmup=10, rep=50, quantiles=(0.5, 0.0, 1.0))
        except Exception as e:
            print(f"  [{idx+1}/{n}] BN={bn} BK={bk} NS={ns}  FAILED: {e}")
            continue
        tflops = 2 * M * N * K / (ms_med / 1e3) / 1e12
        print(f"  [{idx+1:2d}/{n}] BN={bn:3d} BK={bk:3d} NS={ns}  {tflops:6.1f} TFLOPS")
        if ms_med < best_t:
            best_t = ms_med
            best_cfg = cfg
    return best_cfg


def matmul_b6_ms(A, B):
    M, K = A.shape
    _, N = B.shape
    key = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS if _legal_for_problem(c, M, N, K)]
        print(f"[b6_ms] autotuning {M}x{N}x{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bn, bk, ns = _best[key]
        print(f"[b6_ms] best: BN={bn} BK={bk} NS={ns}")
    bn, bk, ns = _best[key]
    return _launch(_get_mod(), _kname(bn, bk, ns), A, B, bn, bk, _smem(bn, bk, ns))
