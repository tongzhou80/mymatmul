"""S6_lb: s6 with two-arg __launch_bounds__ LB tuning and register-estimate pruning.

Adds LB_MIN_BLOCKS as a second __launch_bounds__ argument, compiled as four
cubins (LB=1..4 for NW=4, LB=1..2 for NW=8). Register-estimate pruning removes
configs where the accumulator + register-buffer footprint exceeds the budget.
"""

import os
import numpy as np
import torch
import triton.testing
import pycuda.driver as drv

from .._pycuda_loader import get_module_jit, SM_ARCH

DTYPE = torch.float32

_HERE   = os.path.dirname(os.path.abspath(__file__))
_CU_SRC = os.path.join(_HERE, "_matmul_cuda_ext_s6_lb.cu")

_BMS     = [64, 128, 256]
_BNS     = [64, 128, 256]
_BKS     = [16, 32]
_UNROLLS = [16, 8, 4, 2]
_NWS     = [4, 8]
_MAX_SMEM = 100352

_LB_FOR_NW = {4: [1, 2, 3, 4], 8: [1, 2]}


def _smem(bm, bn, bk):
    return (2 * bm * bk + 2 * bk * bn) * 4  # float32, 4 bytes


def _reg_estimate(bm, bn, nw):
    # acc[TM][TN] + _a[2][TM] + _bv[2][TN/4] (float4 = 4 regs)
    tm = bm // (nw * 2)   # BM / WARP_M / LWARP_M = BM / (NW/2) / 4
    tn = bn // 16          # BN / WARP_N / LWARP_N = BN / 2 / 8
    return tm * tn + 2 * tm + 2 * tn


_CONFIGS = [
    (bm, bn, bk, u, nw, lb)
    for bm in _BMS for bn in _BNS for bk in _BKS for u in _UNROLLS for nw in _NWS
    for lb in _LB_FOR_NW[nw]
    if _smem(bm, bn, bk) <= _MAX_SMEM
    and bm * bn <= 4096 * nw
    and _reg_estimate(bm, bn, nw) <= 65536 // (nw * 32 * lb)
]


def _cubin_path(lb):
    return os.path.join(_HERE, f"_matmul_cuda_ext_s6_lb{lb}_{SM_ARCH}.cubin")


_modules: dict = {}


def _get_mod(lb):
    if lb not in _modules:
        _modules[lb] = get_module_jit(_CU_SRC, _cubin_path(lb), [f"-DLB_MIN_BLOCKS={lb}"])
    return _modules[lb]


def _kname(bm, bn, bk, u, nw):
    return f"matmul_cuda_s6_bm{bm}_bn{bn}_bk{bk}_u{u}_nw{nw}"


def _block(nw):
    return (32, nw, 1)


def _grid(M, N, bm, bn):
    return ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)


def _launch(mod, kname, A, B, block, grid, smem_bytes):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    fn = mod.get_function(kname)
    if smem_bytes > 0:
        fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes)
    fn(np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(C.data_ptr()),
       np.int32(M), np.int32(K), np.int32(N),
       block=block, grid=grid, shared=smem_bytes)
    return C


_best: dict = {}


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    cfgs = [c for c in _CONFIGS if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0]
    best_t, best_cfg, n = float("inf"), cfgs[0], len(cfgs)

    for idx, cfg in enumerate(cfgs):
        bm, bn, bk, u, nw, lb = cfg
        mod   = _get_mod(lb)
        kn    = _kname(bm, bn, bk, u, nw)
        block = _block(nw)
        grid  = _grid(M, N, bm, bn)
        sb    = _smem(bm, bn, bk)
        try:
            _, ms_min, _ = triton.testing.do_bench(
                lambda: _launch(mod, kn, A, B, block, grid, sb),
                warmup=10, rep=50, quantiles=(0.5, 0.0, 1.0),
            )
        except Exception as e:
            print(f"  [{idx+1}/{n}] BM={bm} BN={bn} BK={bk} U={u} NW={nw} LB={lb}  FAILED: {e}")
            continue
        tflops = 2 * M * N * K / (ms_min / 1e3) / 1e12
        print(f"  [{idx+1:3d}/{n}] BM={bm:3d} BN={bn:3d} BK={bk:2d} U={u:2d} NW={nw} LB={lb}  {tflops:6.1f} TFLOPS")
        if ms_min < best_t:
            best_t, best_cfg = ms_min, cfg

    return best_cfg


def matmul_s6_lb(A, B):
    M, K = A.shape
    _, N = B.shape
    key = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0]
        print(f"[s6_lb] autotuning {M}x{N}x{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, u, nw, lb = _best[key]
        print(f"[s6_lb] best: BM={bm} BN={bn} BK={bk} U={u} NW={nw} LB={lb}")
    bm, bn, bk, u, nw, lb = _best[key]
    return _launch(_get_mod(lb), _kname(bm, bn, bk, u, nw), A, B,
                   _block(nw), _grid(M, N, bm, bn), _smem(bm, bn, bk))
