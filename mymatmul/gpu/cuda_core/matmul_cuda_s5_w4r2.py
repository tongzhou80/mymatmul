"""Stage 5 W4R2: s5_w4r with 128 threads (2×2 inter-warp, 4 warps).

BM=BN=128 gives TM=16, TN=8 — same per-thread tile as s5_w4r at BM=256/BN=128,
but 2 blocks fit per SM (32 KB smem, ~196 regs/thread) vs 1 block.
"""

import time
import torch
from .._pycuda_loader import launch_matmul, get_module

_EXT = "_matmul_cuda_ext_s5_w4r2"

_BMS     = [64, 128, 256]
_BNS     = [64, 128, 256]
_BKS     = [16, 32]
_UNROLLS = [2, 4, 8, 16]

_MAX_SMEM = 100352


def _smem(bm, bn, bk):
    return (2 * bm * bk + 2 * bk * bn) * 4


_CONFIGS = [
    (bm, bn, bk, u)
    for bm in _BMS for bn in _BNS for bk in _BKS for u in _UNROLLS
    if _smem(bm, bn, bk) <= _MAX_SMEM
    and bm * bn <= 16384          # TM*TN = BM*BN/128 <= 128 → acc fits in regs
]


def _kname(bm, bn, bk, u):
    return f"matmul_cuda_s5_w4r2_bm{bm}_bn{bn}_bk{bk}_u{u}"


def _block():
    return (32, 4, 1)   # 128 threads = 4 warps


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
        bm, bn, bk, u = cfg
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
            print(f"  [{idx+1}/{n}] BM={bm} BN={bn} BK={bk} U={u}  FAILED: {e}")
            continue

        gflops = 2 * M * N * K / t / 1e12
        print(f"  [{idx+1:3d}/{n}] BM={bm:3d} BN={bn:3d} BK={bk:2d} U={u:2d}   {gflops:6.1f} TFLOPS")

        if t < best_t:
            best_t   = t
            best_cfg = cfg

    return best_cfg


def matmul_s5_w4r2_bm128_bn128_bk16_u8(A, B):
    M, K = A.shape; _, N = B.shape
    bm, bn, bk, u = 128, 128, 16, 8
    get_module(_EXT)
    return launch_matmul(_EXT, _kname(bm, bn, bk, u), A, B,
                         _block(), _grid(M, N, bm, bn), smem_bytes=_smem(bm, bn, bk))


def matmul_s5_w4r2_bm128_bn128_bk16_u16(A, B):
    M, K = A.shape; _, N = B.shape
    bm, bn, bk, u = 128, 128, 16, 16
    get_module(_EXT)
    return launch_matmul(_EXT, _kname(bm, bn, bk, u), A, B,
                         _block(), _grid(M, N, bm, bn), smem_bytes=_smem(bm, bn, bk))


def matmul_s5_w4r2(A, B):
    M, K = A.shape
    _, N = B.shape
    key  = (M, N, K)
    if key not in _best:
        print(f"[s5_w4r2] autotuning {M}x{K}x{N} over {len(_CONFIGS)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, u = _best[key]
        print(f"[s5_w4r2] best: BM={bm} BN={bn} BK={bk} U={u}")

    bm, bn, bk, u = _best[key]
    return launch_matmul(
        _EXT, _kname(bm, bn, bk, u), A, B,
        _block(), _grid(M, N, bm, bn),
        smem_bytes=_smem(bm, bn, bk),
    )
