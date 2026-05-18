"""b1_tc5: Blackwell port of the 4090-best BF16 mma.sync kernel (tc5_lb).

This is a direct port — same .cu logic, only the compile arch changes from
sm_89 → sm_100.  All the primitives it uses (ldmatrix, mma.sync m16n8k16,
cp.async via __pipeline_memcpy_async) are sm_80-level and forward-compatible
to Blackwell.  Hopper-era extras (TMA, mbarrier pipeline, CTA swizzle,
SMEM-staged writeback) are intentionally not present — those come in later
b-series kernels.

Autotuned over (BM, BN, BK, NW, LB_MIN_BLOCKS).
"""

import os
import numpy as np
import torch
import triton.testing
import pycuda.driver as drv

from .._pycuda_loader import get_module_jit, SM_ARCH

DTYPE = torch.bfloat16

_GPU_DIR = os.path.dirname(os.path.abspath(__file__))
_CU_PATH = os.path.join(_GPU_DIR, "_matmul_b1_tc5.cu")

_BMS = [64, 128, 256]
_BNS = [64, 128, 256]
_BKS = [16, 32, 64]
_NWS = [4, 8]

# B200 dynamic SMEM cap per CTA is ~228 KB (sm_100). We keep the 4090's
# 100 KB cap initially — it constrains tile size to what sm_89 could hold,
# but we'll widen it once we know what configs win.
_MAX_SMEM = 100352

# LB_MIN_BLOCKS: same register-budget knob as on Ada.
_LB_FOR_NW = {4: [1, 2, 3, 4], 8: [1, 2]}

# Field labels for the (BM, BN, BK, NW, LB) entries stored in _best.
# Read by benchmarks/bench.py to format the summary table's config column.
_BEST_FIELDS = ("BM", "BN", "BK", "NW", "LB")


def _smem(bm, bn, bk):
    return (2 * bm * bk + 2 * bk * bn) * 2


_CONFIGS = [
    (bm, bn, bk, nw, lb)
    for bm in _BMS for bn in _BNS for bk in _BKS for nw in _NWS
    for lb in _LB_FOR_NW[nw]
    if _smem(bm, bn, bk) <= _MAX_SMEM
    and bm * bn <= 4096 * nw
]


def _cubin_path(lb):
    return os.path.join(_GPU_DIR, f"_matmul_b1_tc5_lb{lb}_{SM_ARCH}.cubin")


def _get_mod(lb):
    return get_module_jit(_CU_PATH, _cubin_path(lb), [f"-DLB_MIN_BLOCKS={lb}"])


def _kname(bm, bn, bk, nw):
    return f"matmul_cuda_tc5_bm{bm}_bn{bn}_bk{bk}_nw{nw}"


def _block(nw):
    return (32, nw, 1)


def _grid(M, N, bm, bn):
    return ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)


def _launch(mod, kname, A, B, block, grid, smem_bytes):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    fn = mod.get_function(kname)
    if smem_bytes > 0:
        fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes)
    fn(np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(C.data_ptr()),
       np.int32(M), np.int32(K), np.int32(N),
       block=block, grid=grid, shared=smem_bytes)
    return C


_best: dict = {}


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    mods = {lb: _get_mod(lb) for lb in (1, 2, 3, 4)}

    cfgs = [c for c in _CONFIGS if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0]
    best_t = float("inf")
    best_cfg = cfgs[0]
    n = len(cfgs)

    for idx, cfg in enumerate(cfgs):
        bm, bn, bk, nw, lb = cfg
        kn = _kname(bm, bn, bk, nw)
        block = _block(nw)
        grid = _grid(M, N, bm, bn)
        sb = _smem(bm, bn, bk)
        mod = mods[lb]
        try:
            ms_med, _, _ = triton.testing.do_bench(
                lambda: _launch(mod, kn, A, B, block, grid, sb),
                warmup=10, rep=50, quantiles=(0.5, 0.0, 1.0),
            )
        except Exception as e:
            print(f"  [{idx+1}/{n}] BM={bm} BN={bn} BK={bk} NW={nw} LB={lb}  FAILED: {e}")
            continue

        tflops = 2 * M * N * K / (ms_med / 1e3) / 1e12
        print(f"  [{idx+1:3d}/{n}] BM={bm:3d} BN={bn:3d} BK={bk:2d} NW={nw} LB={lb}  {tflops:6.1f} TFLOPS")

        if ms_med < best_t:
            best_t = ms_med
            best_cfg = cfg

    return best_cfg


def matmul_b1_tc5(A, B):
    M, K = A.shape
    _, N = B.shape
    key = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0]
        print(f"[b1_tc5] autotuning {M}x{N}x{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, nw, lb = _best[key]
        print(f"[b1_tc5] best: BM={bm} BN={bn} BK={bk} NW={nw} LB={lb}")

    bm, bn, bk, nw, lb = _best[key]
    return _launch(_get_mod(lb), _kname(bm, bn, bk, nw), A, B,
                   _block(nw), _grid(M, N, bm, bn), _smem(bm, bn, bk))
