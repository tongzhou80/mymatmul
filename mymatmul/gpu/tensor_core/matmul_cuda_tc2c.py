"""TC2c: BF16 raw-PTX mma.sync, BK=64, symmetric A+B XOR swizzle.

At BK=64 every A-tile row aliases to bank 0 (row stride = 128 bytes = 32 banks),
identical to the B-tile pattern. Both tiles use the same per-row swizzle:
  physical_chunk = logical_chunk ^ (row % SWZ)
where SWZ = BK/8 = 8 for A and SWZ = BN/8 for B.
"""

import time
import torch
from .._pycuda_loader import launch_matmul, get_module

DTYPE = torch.bfloat16

_EXT = "_matmul_cuda_ext_tc2c"

_BK = 64

_BMS = [64, 128, 256]
_BNS = [64, 128, 256]
_NWS = [4, 8]

_MAX_SMEM = 100352


def _smem(bm, bn):
    return (2 * bm * _BK + 2 * _BK * bn) * 2  # = 256*(bm+bn)


_CONFIGS = [
    (bm, bn, nw)
    for bm in _BMS for bn in _BNS for nw in _NWS
    if _smem(bm, bn) <= _MAX_SMEM
    and bm * bn <= 4096 * nw
]


def _kname(bm, bn, nw):
    return f"matmul_cuda_tc2c_bm{bm}_bn{bn}_bk64_nw{nw}"


def _block(nw):
    return (32, nw, 1)


def _grid(M, N, bm, bn):
    return ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)


_best: dict = {}


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    get_module(_EXT)

    cfgs = [c for c in _CONFIGS if M % c[0] == 0 and N % c[1] == 0 and K % _BK == 0]
    best_t = float("inf")
    best_cfg = cfgs[0]
    n = len(cfgs)

    for idx, cfg in enumerate(cfgs):
        bm, bn, nw = cfg
        kn    = _kname(*cfg)
        block = _block(nw)
        grid  = _grid(M, N, bm, bn)
        sb    = _smem(bm, bn)
        try:
            for _ in range(2):
                launch_matmul(_EXT, kn, A, B, block, grid,
                              out_dtype=torch.float32, smem_bytes=sb)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                launch_matmul(_EXT, kn, A, B, block, grid,
                              out_dtype=torch.float32, smem_bytes=sb)
            torch.cuda.synchronize()
            t = (time.perf_counter() - t0) / 3
        except Exception as e:
            print(f"  [{idx+1}/{n}] BM={bm} BN={bn} NW={nw}  FAILED: {e}")
            continue

        gflops = 2 * M * N * K / t / 1e12
        print(f"  [{idx+1:2d}/{n}] BM={bm:3d} BN={bn:3d} NW={nw}   {gflops:6.1f} TFLOPS")

        if t < best_t:
            best_t   = t
            best_cfg = cfg

    return best_cfg


def matmul_tc2c(A, B):
    M, K = A.shape
    _, N = B.shape
    key  = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS if M % c[0] == 0 and N % c[1] == 0 and K % _BK == 0]
        print(f"[tc2c] autotuning {M}x{K}x{N} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, nw = _best[key]
        print(f"[tc2c] best: BM={bm} BN={bn} BK={_BK} NW={nw}")

    bm, bn, nw = _best[key]
    return launch_matmul(
        _EXT, _kname(bm, bn, nw), A, B,
        _block(nw), _grid(M, N, bm, bn),
        smem_bytes=_smem(bm, bn),
    )
