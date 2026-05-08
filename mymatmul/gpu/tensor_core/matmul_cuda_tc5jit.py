"""TC5jit: TC5 with M, K, N baked in as JIT compile-time constants.

One cubin is compiled per (M, K, N) with -DM_VAL=... -DK_VAL=... -DN_VAL=...
so the compiler sees those values as constants. Benefits vs TC5:
  - num_tiles = K / BK is compile-time → fully known loop trip count
  - Row strides (* K, * N) become constant multiplies (shifts for powers of 2)
  - 3 fewer int kernel arguments → 3 freed registers per thread
  - Write-back bounds checks eliminated (M/N are compile-time multiples of BM/BN)
"""

import os
import time
import torch
from .._pycuda_loader import get_module_jit, launch_matmul_raw, SM_ARCH

DTYPE = torch.bfloat16

_HERE    = os.path.dirname(os.path.abspath(__file__))
_CU_PATH = os.path.join(_HERE, "_matmul_cuda_ext_tc5jit.cu")

_BMS = [64, 128, 256]
_BNS = [64, 128, 256]
_BKS = [16, 32, 64]
_NWS = [4, 8]

_MAX_SMEM = 100352


def _smem(bm, bn, bk):
    return (2 * bm * bk + 2 * bk * bn) * 2


_CONFIGS = [
    (bm, bn, bk, nw)
    for bm in _BMS for bn in _BNS for bk in _BKS for nw in _NWS
    if _smem(bm, bn, bk) <= _MAX_SMEM
    and bm * bn <= 4096 * nw
]


def _kname(bm, bn, bk, nw):
    return f"matmul_cuda_tc5jit_bm{bm}_bn{bn}_bk{bk}_nw{nw}"


def _block(nw):
    return (32, nw, 1)


def _grid(M, N, bm, bn):
    return ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)


_modules: dict = {}
_best:    dict = {}


def _get_module(M, K, N):
    key = (M, K, N)
    if key not in _modules:
        cubin = os.path.join(_HERE, f"_tc5jit_m{M}_k{K}_n{N}_{SM_ARCH}.cubin")
        flags = [f"-DM_VAL={M}", f"-DK_VAL={K}", f"-DN_VAL={N}"]
        _modules[key] = get_module_jit(_CU_PATH, cubin, flags)
    return _modules[key]


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    mod = _get_module(M, K, N)

    cfgs = [c for c in _CONFIGS if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0]
    best_t  = float("inf")
    best_cfg = cfgs[0]
    n = len(cfgs)

    for idx, cfg in enumerate(cfgs):
        bm, bn, bk, nw = cfg
        kn    = _kname(*cfg)
        block = _block(nw)
        grid  = _grid(M, N, bm, bn)
        sb    = _smem(bm, bn, bk)
        try:
            for _ in range(2):
                launch_matmul_raw(mod, kn, A, B, block, grid, smem_bytes=sb)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                launch_matmul_raw(mod, kn, A, B, block, grid, smem_bytes=sb)
            torch.cuda.synchronize()
            t = (time.perf_counter() - t0) / 3
        except Exception as e:
            print(f"  [{idx+1}/{n}] BM={bm} BN={bn} BK={bk} NW={nw}  FAILED: {e}")
            continue

        gflops = 2 * M * N * K / t / 1e12
        print(f"  [{idx+1:3d}/{n}] BM={bm:3d} BN={bn:3d} BK={bk:2d} NW={nw}   {gflops:6.1f} TFLOPS")

        if t < best_t:
            best_t   = t
            best_cfg = cfg

    return best_cfg


def matmul_tc5jit(A, B):
    M, K = A.shape
    _, N = B.shape
    key  = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0]
        print(f"[tc5jit] autotuning {M}x{K}x{N} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, nw = _best[key]
        print(f"[tc5jit] best: BM={bm} BN={bn} BK={bk} NW={nw}")

    bm, bn, bk, nw = _best[key]
    mod = _get_module(M, K, N)
    return launch_matmul_raw(
        mod, _kname(bm, bn, bk, nw), A, B,
        _block(nw), _grid(M, N, bm, bn),
        smem_bytes=_smem(bm, bn, bk),
    )
