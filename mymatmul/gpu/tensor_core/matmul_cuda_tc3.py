"""TC3: TC2b generalised to a tunable NUM_STAGES smem pipeline.

NS=2 is a double-buffer pipeline identical in structure to TC2b.
NS=3/4/5 issue additional cp.async loads ahead of time to better overlap
global-memory latency with tensor-core computation.

Smem bytes: NS * (BM*BK + BK*BN) * 2
"""

import time
import torch
from .._pycuda_loader import launch_matmul, get_module

DTYPE = torch.bfloat16

_EXT = "_matmul_cuda_ext_tc3"

_BMS = [64, 128, 256]
_BNS = [64, 128, 256]
_BKS = [16, 32, 64]
_NWS = [4, 8]
_NSS = [2, 3, 4, 5]

_MAX_SMEM = 100352


def _smem(bm, bn, bk, ns):
    return ns * (bm * bk + bk * bn) * 2


_CONFIGS = [
    (bm, bn, bk, nw, ns)
    for bm in _BMS for bn in _BNS for bk in _BKS for nw in _NWS for ns in _NSS
    if _smem(bm, bn, bk, ns) <= _MAX_SMEM
    and bm * bn <= 4096 * nw
]


def _kname(bm, bn, bk, nw, ns):
    return f"matmul_cuda_tc3_bm{bm}_bn{bn}_bk{bk}_nw{nw}_ns{ns}"


def _block(nw):
    return (32, nw, 1)


def _grid(M, N, bm, bn):
    return ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)


_best: dict = {}


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    get_module(_EXT)

    cfgs = [
        c for c in _CONFIGS
        if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0
        and K // c[2] >= c[4]   # num_tiles >= NUM_STAGES
    ]
    best_t = float("inf")
    best_cfg = cfgs[0]
    n = len(cfgs)

    for idx, cfg in enumerate(cfgs):
        bm, bn, bk, nw, ns = cfg
        kn    = _kname(*cfg)
        block = _block(nw)
        grid  = _grid(M, N, bm, bn)
        sb    = _smem(bm, bn, bk, ns)
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
            print(f"  [{idx+1}/{n}] BM={bm} BN={bn} BK={bk} NW={nw} NS={ns}  FAILED: {e}")
            continue

        gflops = 2 * M * N * K / t / 1e12
        print(f"  [{idx+1:3d}/{n}] BM={bm:3d} BN={bn:3d} BK={bk:2d} NW={nw} NS={ns}   {gflops:6.1f} TFLOPS")

        if t < best_t:
            best_t   = t
            best_cfg = cfg

    return best_cfg


def matmul_tc3(A, B):
    M, K = A.shape
    _, N = B.shape
    key  = (M, N, K)
    if key not in _best:
        cfgs = [
            c for c in _CONFIGS
            if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0
            and K // c[2] >= c[4]
        ]
        print(f"[tc3] autotuning {M}x{K}x{N} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, nw, ns = _best[key]
        print(f"[tc3] best: BM={bm} BN={bn} BK={bk} NW={nw} NS={ns}")

    bm, bn, bk, nw, ns = _best[key]
    return launch_matmul(
        _EXT, _kname(bm, bn, bk, nw, ns), A, B,
        _block(nw), _grid(M, N, bm, bn),
        smem_bytes=_smem(bm, bn, bk, ns),
    )
