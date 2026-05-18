"""b2_ms: b1_tc5 extended with a tunable NUM_STAGES cp.async pipeline depth.

The kernel still runs synchronous `mma.sync m16n8k16` — so we keep the
simple "only one MMA in flight" design. NUM_STAGES sets how many tiles
worth of A+B SMEM buffers we keep, which sets how much DMA latency can
overlap with compute on the *load* side.

NS=2 is identical to b1_tc5 (one in-flight load + one being consumed).
Larger NS hides more cp.async latency at the cost of NS × tile bytes
of SMEM.

Config space adds NS on top of b1_tc5's (BM, BN, BK, NW, LB):
  SMEM:  NS × (BM·BK + BK·BN) × 2 bytes
  K:     K/BK ≥ NS (enforced at launch)
"""

import os
import numpy as np
import torch
import triton.testing
import pycuda.driver as drv

from .._pycuda_loader import get_module_jit, SM_ARCH

DTYPE = torch.bfloat16

_GPU_DIR = os.path.dirname(os.path.abspath(__file__))
_CU_PATH = os.path.join(_GPU_DIR, "_matmul_b2_ms.cu")

_BMS = [64, 128, 256]
_BNS = [64, 128, 256]
_BKS = [16, 32, 64]
_NWS = [4, 8]
_NSS = [2, 3, 4, 5]

# B200 has 228 KB dynamic SMEM/SM. Same headroom rule as Hopper: cap at 200 KB.
_MAX_SMEM = 200 * 1024

_LB_FOR_NW = {4: [1, 2, 3, 4], 8: [1, 2]}


def _smem(bm, bn, bk, ns):
    return (ns * bm * bk + ns * bk * bn) * 2


_CONFIGS = [
    (bm, bn, bk, nw, lb, ns)
    for bm in _BMS for bn in _BNS for bk in _BKS
    for nw in _NWS for ns in _NSS
    for lb in _LB_FOR_NW[nw]
    if _smem(bm, bn, bk, ns) <= _MAX_SMEM
    and bm * bn <= 4096 * nw
]


def _cubin_path(lb):
    return os.path.join(_GPU_DIR, f"_matmul_b2_ms_lb{lb}_{SM_ARCH}.cubin")


def _get_mod(lb):
    return get_module_jit(_CU_PATH, _cubin_path(lb), [f"-DLB_MIN_BLOCKS={lb}"])


def _kname(bm, bn, bk, nw, ns):
    return f"matmul_b2_ms_bm{bm}_bn{bn}_bk{bk}_nw{nw}_ns{ns}"


def _block(nw):
    return (32, nw, 1)


def _grid(M, N, bm, bn):
    return ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)


def _launch(mod, kname, A, B, block, grid, smem_bytes):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    fn = mod.get_function(kname)
    if smem_bytes > 0:
        fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes)
    fn(np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(C.data_ptr()),
       np.int32(M), np.int32(K), np.int32(N),
       block=block, grid=grid, shared=smem_bytes)
    return C


_best: dict = {}
_BEST_FIELDS = ("BM", "BN", "BK", "NW", "LB", "NS")


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    all_lbs = sorted({c[4] for c in _CONFIGS})
    mods = {lb: _get_mod(lb) for lb in all_lbs}

    cfgs = [c for c in _CONFIGS
            if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0
            and K // c[2] >= c[5]]   # num_tiles >= NS

    best_t = float("inf")
    best_cfg = cfgs[0]
    n = len(cfgs)
    for idx, cfg in enumerate(cfgs):
        bm, bn, bk, nw, lb, ns = cfg
        kn = _kname(bm, bn, bk, nw, ns)
        sb = _smem(bm, bn, bk, ns)
        try:
            ms_med, _, _ = triton.testing.do_bench(
                lambda bm=bm, bn=bn, bk=bk, nw=nw, lb=lb, ns=ns, kn=kn, sb=sb:
                    _launch(mods[lb], kn, A, B, _block(nw), _grid(M, N, bm, bn), sb),
                warmup=10, rep=50, quantiles=(0.5, 0.0, 1.0))
        except Exception as e:
            print(f"  [{idx+1}/{n}] BM={bm} BN={bn} BK={bk} NW={nw} LB={lb} NS={ns}  FAILED: {e}")
            continue

        tflops = 2 * M * N * K / (ms_med / 1e3) / 1e12
        print(f"  [{idx+1:3d}/{n}] BM={bm:3d} BN={bn:3d} BK={bk:2d} "
              f"NW={nw} LB={lb} NS={ns}  {tflops:6.1f} TFLOPS")
        if ms_med < best_t:
            best_t = ms_med
            best_cfg = cfg
    return best_cfg


def matmul_b2_ms(A, B):
    M, K = A.shape
    _, N = B.shape
    key = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS
                if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0
                and K // c[2] >= c[5]]
        print(f"[b2_ms] autotuning {M}x{N}x{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, nw, lb, ns = _best[key]
        print(f"[b2_ms] best: BM={bm} BN={bn} BK={bk} NW={nw} LB={lb} NS={ns}")

    bm, bn, bk, nw, lb, ns = _best[key]
    return _launch(_get_mod(lb), _kname(bm, bn, bk, nw, ns), A, B,
                   _block(nw), _grid(M, N, bm, bn), _smem(bm, bn, bk, ns))
