"""Stage 5 with L2 grouped block ordering; auto-tuned over BM/BN/BK/UNROLL/GROUP_M."""

import time
import torch
from .._pycuda_loader import launch_matmul, get_module

_EXT = "_matmul_cuda_ext_s5_l2"

_BMS     = [64, 128, 256]
_BNS     = [64, 128, 256]
_BKS     = [16, 32]
_UNROLLS = [2, 4, 8, 16]
_GMS     = [4, 8]          # GROUP_M values

_MAX_SMEM = 100352


def _smem(bm, bn, bk):
    return (2 * bm * bk + 2 * bk * bn) * 4


_CONFIGS = [
    (bm, bn, bk, u, gm)
    for bm in _BMS for bn in _BNS for bk in _BKS for u in _UNROLLS for gm in _GMS
    if _smem(bm, bn, bk) <= _MAX_SMEM
    and not (bm == 256 and bn == 256)
]  # 64 tile configs × 2 GROUP_M = 128 configs


def _kname(bm, bn, bk, u, gm):
    return f"matmul_cuda_s5_l2_bm{bm}_bn{bn}_bk{bk}_u{u}_gm{gm}"


def _block():
    return (32, 8, 1)


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
        bm, bn, bk, u, gm = cfg
        kn    = _kname(*cfg)
        block = _block()
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
            print(f"  [{idx+1}/{n}] BM={bm} BN={bn} BK={bk} U={u} GM={gm}  FAILED: {e}")
            continue

        gflops = 2 * M * N * K / t / 1e12
        print(f"  [{idx+1:3d}/{n}] BM={bm:3d} BN={bn:3d} BK={bk:2d} U={u:2d} GM={gm}"
              f"  {gflops:6.1f} TFLOPS")

        if t < best_t:
            best_t   = t
            best_cfg = cfg

    return best_cfg


def matmul_s5_l2(A, B):
    M, K = A.shape
    _, N = B.shape
    key  = (M, N, K)
    if key not in _best:
        print(f"[s5_l2] autotuning {M}x{K}x{N} over {len(_CONFIGS)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, u, gm = _best[key]
        print(f"[s5_l2] best: BM={bm} BN={bn} BK={bk} U={u} GM={gm}")

    bm, bn, bk, u, gm = _best[key]
    return launch_matmul(
        _EXT, _kname(bm, bn, bk, u, gm), A, B,
        _block(), _grid(M, N, bm, bn),
        smem_bytes=_smem(bm, bn, bk),
    )
