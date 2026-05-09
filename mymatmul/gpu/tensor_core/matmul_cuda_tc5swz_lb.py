"""TC5swz_lb: TC5swz (GROUP_M L2 swizzle) with tunable __launch_bounds__.

TC5swz adds a GROUP_M CTA traversal order that improves B-tile L2 reuse.
This variant adds LB_MIN_BLOCKS tuning compiled to four cubins (LB=1..4)
with register-estimate pruning. SW dimension covers [1,2,4,8].
"""

import os
import numpy as np
import torch
import triton.testing
import pycuda.driver as drv

from .._pycuda_loader import get_module_jit

DTYPE = torch.bfloat16

_GPU_DIR   = os.path.dirname(os.path.abspath(__file__))
_CU_SRC    = os.path.join(_GPU_DIR, "_matmul_cuda_ext_tc5swz_lb.cu")
_SM_ARCH   = "sm_89"

_LB_BLOCKS  = [1, 2, 3, 4]
_BMS        = [64, 128, 256]
_BNS        = [64, 128, 256]
_BKS        = [16, 32, 64]
_NWS        = [4, 8]
_SWIZZLES   = [1, 2, 4, 8]
_MAX_SMEM   = 100352
_LB_FOR_NW  = {4: [1, 2, 3, 4], 8: [1, 2]}


def _smem(bm, bn, bk):
    return (2 * bm * bk + 2 * bk * bn) * 2


def _reg_estimate(bm, bn, bk, nw):
    wm = (bm * 2) // (nw * 16)
    wn = bn // 16
    kk = bk // 16
    return wm * wn * 4 + kk * (wm * 4 + wn * 2)


_CONFIGS = [
    (bm, bn, bk, nw, sw, lb)
    for bm in _BMS for bn in _BNS for bk in _BKS for nw in _NWS
    for sw in _SWIZZLES for lb in _LB_FOR_NW[nw]
    if _smem(bm, bn, bk) <= _MAX_SMEM
    and bm * bn <= 4096 * nw
    and _reg_estimate(bm, bn, bk, nw) <= 65536 // (nw * 32 * lb)
]


def _cubin_path(lb):
    return os.path.join(_GPU_DIR, f"_matmul_cuda_ext_tc5swz_lb{lb}_{_SM_ARCH}.cubin")


def _get_mod(lb):
    return get_module_jit(_CU_SRC, _cubin_path(lb), [f"-DLB_MIN_BLOCKS={lb}"])


def _kname(bm, bn, bk, nw, sw):
    return f"matmul_cuda_tc5swz_bm{bm}_bn{bn}_bk{bk}_nw{nw}_sw{sw}"


def _block(nw):
    return (32, nw, 1)


def _grid(M, N, bm, bn):
    return ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)


def _launch(mod, kname, A, B, block, grid, smem_bytes):
    M, K = A.shape
    _, N  = B.shape
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
    mods = {lb: _get_mod(lb) for lb in _LB_BLOCKS}
    cfgs = [c for c in _CONFIGS if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0]
    best_t, best_cfg, n = float("inf"), cfgs[0], len(cfgs)
    for idx, cfg in enumerate(cfgs):
        bm, bn, bk, nw, sw, lb = cfg
        kn    = _kname(bm, bn, bk, nw, sw)
        block = _block(nw)
        grid  = _grid(M, N, bm, bn)
        sb    = _smem(bm, bn, bk)
        try:
            _, ms_min, _ = triton.testing.do_bench(
                lambda: _launch(mods[lb], kn, A, B, block, grid, sb),
                warmup=10, rep=50, quantiles=(0.5, 0.0, 1.0),
            )
        except Exception as e:
            print(f"  [{idx+1}/{n}] BM={bm} BN={bn} BK={bk} NW={nw} SW={sw} LB={lb}  FAILED: {e}")
            continue
        tflops = 2 * M * N * K / (ms_min / 1e3) / 1e12
        print(f"  [{idx+1:3d}/{n}] BM={bm:3d} BN={bn:3d} BK={bk:2d} NW={nw} SW={sw} LB={lb}  {tflops:6.1f} TFLOPS")
        if ms_min < best_t:
            best_t, best_cfg = ms_min, cfg
    return best_cfg


def matmul_tc5swz_lb(A, B):
    M, K = A.shape
    _, N = B.shape
    key  = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0]
        print(f"[tc5swz_lb] autotuning {M}x{N}x{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, nw, sw, lb = _best[key]
        print(f"[tc5swz_lb] best: BM={bm} BN={bn} BK={bk} NW={nw} SW={sw} LB={lb}")
    bm, bn, bk, nw, sw, lb = _best[key]
    return _launch(_get_mod(lb), _kname(bm, bn, bk, nw, sw), A, B,
                   _block(nw), _grid(M, N, bm, bn), _smem(bm, bn, bk))
