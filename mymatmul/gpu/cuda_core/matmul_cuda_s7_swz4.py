"""Stage 7: s6 with M, N, K baked as compile-time constants (JIT per shape).

For each (M, N, K) the template is compiled once with -DM_VAL=M -DN_VAL=N
-DK_VAL=K.  With the dimensions constexpr the compiler can:
  - treat num_tiles=K/BK as a known loop count (better scheduling / unrolling)
  - statically eliminate the bounds-check branch in the store epilog
  - simplify index arithmetic involving N and K

The compiled cubin is cached on disk next to the template source so subsequent
runs skip recompilation.
"""

import os
import time

import torch

from .._pycuda_loader import SM_ARCH, get_module_jit, launch_matmul_raw

_TEMPLATE_CU = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "_matmul_cuda_ext_s7_swz4_template.cu",
)

_BMS     = [64, 128, 256]
_BNS     = [64, 128, 256]
_BKS     = [16, 32]
_UNROLLS = [16, 8, 4, 2]
_NWS     = [4, 8]

_MAX_SMEM = 100352


def _smem(bm, bn, bk):
    return (2 * bm * bk + 2 * bk * bn) * 4


def _configs(M, N, K):
    return [
        (bm, bn, bk, u, nw)
        for bm in _BMS for bn in _BNS for bk in _BKS for u in _UNROLLS for nw in _NWS
        if _smem(bm, bn, bk) <= _MAX_SMEM
        and bm * bn <= 4096 * nw
        and M % bm == 0 and N % bn == 0 and K % bk == 0
    ]


def _kname(bm, bn, bk, u, nw):
    return f"matmul_cuda_s7_swz4_bm{bm}_bn{bn}_bk{bk}_u{u}_nw{nw}"


def _block(nw):
    return (32, nw, 1)


def _grid(M, N, bm, bn):
    return (M // bm * N // bn, 1, 1)  # 1D grid; swizzle in kernel


def _get_module(M, N, K):
    cubin = _TEMPLATE_CU[:-3] + f"_m{M}_n{N}_k{K}_{SM_ARCH}.cubin"
    flags = [f"-DM_VAL={M}", f"-DN_VAL={N}", f"-DK_VAL={K}"]
    return get_module_jit(_TEMPLATE_CU, cubin, flags)


_best: dict = {}


def _tune(M, N, K):
    mod = _get_module(M, N, K)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    cfgs = _configs(M, N, K)
    best_t = float("inf")
    best_cfg = cfgs[0]
    n = len(cfgs)

    for idx, cfg in enumerate(cfgs):
        bm, bn, bk, u, nw = cfg
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
            print(f"  [{idx+1}/{n}] BM={bm} BN={bn} BK={bk} U={u} NW={nw}  FAILED: {e}")
            continue

        gflops = 2 * M * N * K / t / 1e12
        print(f"  [{idx+1:3d}/{n}] BM={bm:3d} BN={bn:3d} BK={bk:2d} U={u:2d} NW={nw}   {gflops:6.1f} TFLOPS")

        if t < best_t:
            best_t   = t
            best_cfg = cfg

    return best_cfg


def matmul_s7_swz4(A, B):
    M, K = A.shape
    _, N = B.shape
    key  = (M, N, K)
    if key not in _best:
        cfgs = _configs(M, N, K)
        print(f"[s7_swz4] autotuning {M}x{K}x{N} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, u, nw = _best[key]
        print(f"[s7_swz4] best: BM={bm} BN={bn} BK={bk} U={u} NW={nw}")

    bm, bn, bk, u, nw = _best[key]
    mod = _get_module(M, N, K)
    return launch_matmul_raw(
        mod, _kname(bm, bn, bk, u, nw), A, B,
        _block(nw), _grid(M, N, bm, bn),
        smem_bytes=_smem(bm, bn, bk),
    )
