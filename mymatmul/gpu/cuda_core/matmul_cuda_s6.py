"""Stage 6: unified warp-tiled register-double-buffered matmul.

NUM_WARPS=8 → 256 threads, 4×2 inter-warp (= s5_w4r)
NUM_WARPS=4 → 128 threads, 2×2 inter-warp (= s5_w4r2)
Search space extends to BM=32 and BN=32.
"""

import time
import torch
from .._pycuda_loader import launch_matmul, get_module

_EXT = "_matmul_cuda_ext_s6"

_BMS     = [64, 128, 256]
_BNS     = [64, 128, 256]
_BKS     = [16, 32]
_UNROLLS = [16, 8, 4, 2]
_NWS     = [4, 8]

_MAX_SMEM = 100352


def _smem(bm, bn, bk):
    return (2 * bm * bk + 2 * bk * bn) * 4


_CONFIGS = [
    (bm, bn, bk, u, nw)
    for bm in _BMS for bn in _BNS for bk in _BKS for u in _UNROLLS for nw in _NWS
    if _smem(bm, bn, bk) <= _MAX_SMEM
    and bm * bn <= 4096 * nw         # TM*TN <= 128: acc fits in registers
]


def _kname(bm, bn, bk, u, nw):
    return f"matmul_cuda_s6_bm{bm}_bn{bn}_bk{bk}_u{u}_nw{nw}"


def _block(nw):
    return (32, nw, 1)


def _grid(M, N, bm, bn):
    return ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)


_best: dict = {}


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    get_module(_EXT)

    best_t = float("inf")
    best_cfg = _CONFIGS[0]
    n = len(_CONFIGS)

    for idx, cfg in enumerate(_CONFIGS):
        bm, bn, bk, u, nw = cfg
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
            print(f"  [{idx+1}/{n}] BM={bm} BN={bn} BK={bk} U={u} NW={nw}  FAILED: {e}")
            continue

        gflops = 2 * M * N * K / t / 1e12
        print(f"  [{idx+1:3d}/{n}] BM={bm:3d} BN={bn:3d} BK={bk:2d} U={u:2d} NW={nw}   {gflops:6.1f} TFLOPS")

        if t < best_t:
            best_t   = t
            best_cfg = cfg

    return best_cfg


def matmul_s6(A, B):
    M, K = A.shape
    _, N = B.shape
    key  = (M, N, K)
    if key not in _best:
        print(f"[s6] autotuning {M}x{K}x{N} over {len(_CONFIGS)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, u, nw = _best[key]
        print(f"[s6] best: BM={bm} BN={bn} BK={bk} U={u} NW={nw}")

    bm, bn, bk, u, nw = _best[key]
    return launch_matmul(
        _EXT, _kname(bm, bn, bk, u, nw), A, B,
        _block(nw), _grid(M, N, bm, bn),
        smem_bytes=_smem(bm, bn, bk),
    )
