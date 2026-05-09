"""TC5_lb: TC5 autotuned over tile shape AND __launch_bounds__ min-blocks-per-SM.

Compiles _matmul_cuda_ext_tc5_lb.cu four times with -DLB_MIN_BLOCKS=1/2/3/4,
each to its own cubin.  Register budget = floor(65536 / (nw*32 * lb)).

  NW=4 (128 threads): LB=2→256 regs, LB=3→170 regs, LB=4→128 regs
  NW=8 (256 threads): LB=1→256 regs (baseline), LB=2→128 regs
    (LB=3→85 regs, LB=4→64 regs are too few for NW=8 — excluded)

Unlike -maxrregcount, __launch_bounds__ with two parameters is honoured even when
the kernel already has a single-arg __launch_bounds__ that would otherwise override it.
"""

import os
import numpy as np
import torch
import triton.testing
import pycuda.driver as drv

from .._pycuda_loader import get_module_jit

DTYPE = torch.bfloat16

_GPU_DIR   = os.path.dirname(os.path.abspath(__file__))
_TC5_LB_CU = os.path.join(_GPU_DIR, "_matmul_cuda_ext_tc5_lb.cu")
_SM_ARCH   = "sm_89"

_LB_BLOCKS = [1, 2, 3, 4]

_BMS = [64, 128, 256]
_BNS = [64, 128, 256]
_BKS = [16, 32, 64]
_NWS = [4, 8]
_MAX_SMEM = 100352

# LB_MIN_BLOCKS is register-budget knob: floor(65536 / (nw*32 * lb)) regs/thread.
# NW=4: LB=2→256 regs, LB=3→170 regs, LB=4→128 regs  (all viable)
# NW=8: LB=1→256 regs (baseline), LB=2→128 regs; LB=3→85, LB=4→64 — too few, heavy spills.
_LB_FOR_NW = {4: [1, 2, 3, 4], 8: [1, 2]}


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
    return os.path.join(_GPU_DIR, f"_matmul_cuda_ext_tc5_lb{lb}_{_SM_ARCH}.cubin")


def _get_mod(lb):
    return get_module_jit(_TC5_LB_CU, _cubin_path(lb), [f"-DLB_MIN_BLOCKS={lb}"])


def _kname(bm, bn, bk, nw):
    return f"matmul_cuda_tc5_bm{bm}_bn{bn}_bk{bk}_nw{nw}"


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
    best_t   = float("inf")
    best_cfg = cfgs[0]
    n = len(cfgs)

    for idx, cfg in enumerate(cfgs):
        bm, bn, bk, nw, lb = cfg
        kn    = _kname(bm, bn, bk, nw)
        block = _block(nw)
        grid  = _grid(M, N, bm, bn)
        sb    = _smem(bm, bn, bk)
        mod   = mods[lb]
        try:
            _, ms_min, _ = triton.testing.do_bench(
                lambda: _launch(mod, kn, A, B, block, grid, sb),
                warmup=10, rep=50, quantiles=(0.5, 0.0, 1.0),
            )
        except Exception as e:
            print(f"  [{idx+1}/{n}] BM={bm} BN={bn} BK={bk} NW={nw} LB={lb}  FAILED: {e}")
            continue

        tflops = 2 * M * N * K / (ms_min / 1e3) / 1e12
        print(f"  [{idx+1:3d}/{n}] BM={bm:3d} BN={bn:3d} BK={bk:2d} NW={nw} LB={lb}  {tflops:6.1f} TFLOPS")

        if ms_min < best_t:
            best_t   = ms_min
            best_cfg = cfg

    return best_cfg


def matmul_tc5_reg(A, B):
    M, K = A.shape
    _, N = B.shape
    key  = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0]
        print(f"[tc5_reg] autotuning {M}x{N}x{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, nw, lb = _best[key]
        print(f"[tc5_reg] best: BM={bm} BN={bn} BK={bk} NW={nw} LB={lb}")

    bm, bn, bk, nw, lb = _best[key]
    return _launch(_get_mod(lb), _kname(bm, bn, bk, nw), A, B,
                   _block(nw), _grid(M, N, bm, bn), _smem(bm, bn, bk))
