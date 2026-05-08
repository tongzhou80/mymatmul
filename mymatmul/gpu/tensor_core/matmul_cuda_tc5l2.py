"""TC5l2: TC5 with cp.async.cg.L2::128B prefetch hint on tile loads.

Uses cp.async.cg + L2::128B eviction hint instead of __pipeline_memcpy_async.
The hint tells the L2 to prefetch/retain the full 128-byte cache line, so
adjacent CTAs that touch the same line find it warm in L2.

L2::128B requires 16-byte (8-element bf16) copies; configs where A_ELEM or
B_ELEM is 4 (8 bytes) fall back to cp.async.ca without the hint.

Everything else (smem layout, XOR swizzle, COMPUTE_TILE, vectorized
bfloat162 write-back) is identical to TC5.
"""

import time
import torch
from .._pycuda_loader import launch_matmul, get_module

DTYPE = torch.bfloat16

_EXT = "_matmul_cuda_ext_tc5l2"

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
    return f"matmul_cuda_tc5l2_bm{bm}_bn{bn}_bk{bk}_nw{nw}"


def _block(nw):
    return (32, nw, 1)


def _grid(M, N, bm, bn):
    return ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)


_best: dict = {}


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    get_module(_EXT)

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
                launch_matmul(_EXT, kn, A, B, block, grid, smem_bytes=sb)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                launch_matmul(_EXT, kn, A, B, block, grid, smem_bytes=sb)
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


def matmul_tc5l2(A, B):
    M, K = A.shape
    _, N = B.shape
    key  = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0]
        print(f"[tc5l2] autotuning {M}x{K}x{N} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, nw = _best[key]
        print(f"[tc5l2] best: BM={bm} BN={bn} BK={bk} NW={nw}")

    bm, bn, bk, nw = _best[key]
    return launch_matmul(
        _EXT, _kname(bm, bn, bk, nw), A, B,
        _block(nw), _grid(M, N, bm, bn),
        smem_bytes=_smem(bm, bn, bk),
    )
