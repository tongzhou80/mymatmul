"""Stage 5 W4P: s5_w4 with 4-buffer paired loading to halve __syncthreads__ overhead."""

import time
import torch
from .._pycuda_loader import launch_matmul, get_module

_EXT = "_matmul_cuda_ext_s5_w4p"

_BMS     = [64, 128, 256]
_BNS     = [64, 128, 256]
_UNROLLS = [2, 4, 8, 16]

_MAX_SMEM = 100352


def _smem(bm, bn):
    return (4 * bm * 16 + 4 * 16 * bn) * 4   # 4 buffers of BK=16


_CONFIGS = [
    (bm, bn, u)
    for bm in _BMS for bn in _BNS for u in _UNROLLS
    if _smem(bm, bn) <= _MAX_SMEM
    and not (bm == 256 and bn == 256)
]  # 32 configs


def _kname(bm, bn, u):
    return f"matmul_cuda_s5_w4p_bm{bm}_bn{bn}_u{u}"


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
        bm, bn, u = cfg
        kn    = _kname(*cfg)
        block = _block()
        grid  = _grid(M, N, bm, bn)
        sb    = _smem(bm, bn)
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
            print(f"  [{idx+1}/{n}] BM={bm} BN={bn} U={u}  FAILED: {e}")
            continue

        gflops = 2 * M * N * K / t / 1e12
        print(f"  [{idx+1:3d}/{n}] BM={bm:3d} BN={bn:3d} U={u:2d}   {gflops:6.1f} TFLOPS")

        if t < best_t:
            best_t   = t
            best_cfg = cfg

    return best_cfg


def matmul_s5_w4p_bm256_bn128_u16(A, B):
    M, K = A.shape; _, N = B.shape
    bm, bn, u = 256, 128, 16
    get_module(_EXT)
    return launch_matmul(_EXT, _kname(bm, bn, u), A, B,
                         _block(), _grid(M, N, bm, bn), smem_bytes=_smem(bm, bn))


def matmul_s5_w4p(A, B):
    M, K = A.shape
    _, N = B.shape
    key  = (M, N, K)
    if key not in _best:
        print(f"[s5_w4p] autotuning {M}x{K}x{N} over {len(_CONFIGS)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, u = _best[key]
        print(f"[s5_w4p] best: BM={bm} BN={bn} U={u}")

    bm, bn, u = _best[key]
    return launch_matmul(
        _EXT, _kname(bm, bn, u), A, B,
        _block(), _grid(M, N, bm, bn),
        smem_bytes=_smem(bm, bn),
    )
