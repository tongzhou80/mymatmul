"""b4_pipe: tcgen05.mma + NS-deep cp.async prefetch + autotuning (h2_s8 style).

Single cubin with __launch_bounds__(*, 1). Tunes over BM/BN/BK/NS only.

Legal HW shapes for tcgen05.mma.cta_group::1.kind::f16:
  BM ∈ {64, 128}, BN ∈ {8, 16, ..., 256} step 8, BK = multiple of 16.

BM=64 currently disabled (epilogue assumes M=128 TMEM layout) — locked to 128.
NW = BM/32 (epilogue tcgen05.ld.32x32b drains 32 rows per warp).
"""

import os
import numpy as np
import torch
import triton.testing
import pycuda.driver as drv

from .._pycuda_loader import get_module_jit, SM_ARCH

DTYPE = torch.bfloat16

_GPU_DIR = os.path.dirname(os.path.abspath(__file__))
_CU_PATH = os.path.join(_GPU_DIR, "_matmul_b4_pipe.cu")
_CUBIN   = os.path.join(_GPU_DIR, f"_matmul_b4_pipe_{SM_ARCH}.cubin")

# BM=64 disabled — see module docstring.
_BMS = [128]
_BNS = [64, 128, 256]
_BKS = [16, 32, 64, 128]
_NSS = [2, 3, 4, 5]

_MAX_SMEM = 200 * 1024


def _nw(bm):
    return bm // 32   # 4 for BM=128


def _smem(bm, bn, bk, ns):
    return ns * (bm + bn) * bk * 2 + 32


def _legal(bm, bn, bk, ns):
    return _smem(bm, bn, bk, ns) <= _MAX_SMEM


_CONFIGS = [
    (bm, bn, bk, ns)
    for bm in _BMS for bn in _BNS for bk in _BKS for ns in _NSS
    if _legal(bm, bn, bk, ns)
]


def _get_mod():
    return get_module_jit(_CU_PATH, _CUBIN, ["-arch=sm_100a", "-DLB_MIN_BLOCKS=1"])


def _kname(bm, bn, bk, ns):
    return f"matmul_b4_pipe_bm{bm}_bn{bn}_bk{bk}_nw{_nw(bm)}_ns{ns}"


def _launch(mod, kname, A, B, bm, bn, smem_bytes):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device="cuda", dtype=DTYPE)
    fn = mod.get_function(kname)
    if smem_bytes > 0:
        fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes)
    block = (32, _nw(bm), 1)
    grid = (N // bn, M // bm, 1)
    fn(np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(C.data_ptr()),
       np.int32(M), np.int32(K), np.int32(N),
       block=block, grid=grid, shared=smem_bytes)
    return C


_best: dict = {}


def _legal_for_problem(cfg, M, N, K):
    bm, bn, bk, ns = cfg
    if M % bm != 0 or N % bn != 0 or K % bk != 0:
        return False
    if K // bk < ns:
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
        bm, bn, bk, ns = cfg
        kn = _kname(bm, bn, bk, ns)
        sb = _smem(bm, bn, bk, ns)
        try:
            ms_med, _, _ = triton.testing.do_bench(
                lambda bm=bm, bn=bn, bk=bk, ns=ns, kn=kn, sb=sb:
                    _launch(mod, kn, A, B, bm, bn, sb),
                warmup=10, rep=50, quantiles=(0.5, 0.0, 1.0))
        except Exception as e:
            print(f"  [{idx+1}/{n}] BM={bm} BN={bn} BK={bk} NS={ns}  FAILED: {e}")
            continue

        tflops = 2 * M * N * K / (ms_med / 1e3) / 1e12
        print(f"  [{idx+1:3d}/{n}] BM={bm:3d} BN={bn:3d} BK={bk:3d} NS={ns}  "
              f"{tflops:6.1f} TFLOPS")
        if ms_med < best_t:
            best_t = ms_med
            best_cfg = cfg
    return best_cfg


def matmul_b4_pipe(A, B):
    M, K = A.shape
    _, N = B.shape
    key = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS if _legal_for_problem(c, M, N, K)]
        print(f"[b4_pipe] autotuning {M}x{N}x{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, ns = _best[key]
        print(f"[b4_pipe] best: BM={bm} BN={bn} BK={bk} NS={ns}")

    bm, bn, bk, ns = _best[key]
    return _launch(_get_mod(), _kname(bm, bn, bk, ns), A, B, bm, bn, _smem(bm, bn, bk, ns))
