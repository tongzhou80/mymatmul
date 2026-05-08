"""TC5swz: TC5 + GROUP_M CTA swizzle for improved L2 B-tile reuse.

GROUP_M groups consecutive M-tiles into a super-group traversed M-first:
for a given N column all GROUP_M M-tiles are scheduled before the next N
column, so those CTAs all read the same B strip and share it in L2.

SW=1 is identity (no swizzle, identical to TC5).
SW=2/4/8 reuse the same B tile across 2/4/8 consecutive M-tiles.
"""

import time
import torch
from .._pycuda_loader import launch_matmul, get_module

DTYPE = torch.bfloat16

_EXT = "_matmul_cuda_ext_tc5swz"

_BMS      = [64, 128, 256]
_BNS      = [64, 128, 256]
_BKS      = [16, 32, 64]
_NWS      = [4, 8]
_SWIZZLES = [1, 2, 4, 8]

_MAX_SMEM = 100352


def _smem(bm, bn, bk):
    return (2 * bm * bk + 2 * bk * bn) * 2


_CONFIGS = [
    (bm, bn, bk, nw, sw)
    for bm in _BMS for bn in _BNS for bk in _BKS for nw in _NWS for sw in _SWIZZLES
    if _smem(bm, bn, bk) <= _MAX_SMEM
    and bm * bn <= 4096 * nw
]


def _kname(bm, bn, bk, nw, sw):
    return f"matmul_cuda_tc5swz_bm{bm}_bn{bn}_bk{bk}_nw{nw}_sw{sw}"


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
        bm, bn, bk, nw, sw = cfg
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
            print(f"  [{idx+1}/{n}] BM={bm} BN={bn} BK={bk} NW={nw} SW={sw}  FAILED: {e}")
            continue

        gflops = 2 * M * N * K / t / 1e12
        print(f"  [{idx+1:3d}/{n}] BM={bm:3d} BN={bn:3d} BK={bk:2d} NW={nw} SW={sw}   {gflops:6.1f} TFLOPS")

        if t < best_t:
            best_t   = t
            best_cfg = cfg

    return best_cfg


def matmul_tc5swz(A, B):
    M, K = A.shape
    _, N = B.shape
    key  = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0]
        print(f"[tc5swz] autotuning {M}x{K}x{N} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, nw, sw = _best[key]
        print(f"[tc5swz] best: BM={bm} BN={bn} BK={bk} NW={nw} SW={sw}")

    bm, bn, bk, nw, sw = _best[key]
    return launch_matmul(
        _EXT, _kname(bm, bn, bk, nw, sw), A, B,
        _block(nw), _grid(M, N, bm, bn),
        smem_bytes=_smem(bm, bn, bk),
    )
