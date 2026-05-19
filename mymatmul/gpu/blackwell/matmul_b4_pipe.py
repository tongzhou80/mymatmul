"""b4_pipe: b3_tc05 + NS-deep cp.async pipeline + BK tunable + async MMA commit.

Fixed: BM=128, BN=128, NW=4. Tunable: BK ∈ {16,32,64,128}, NS ∈ {2..5}, LB ∈ {1..4}.
"""

import os
import numpy as np
import torch
import triton.testing
import pycuda.driver as drv

from .._pycuda_loader import get_module_jit, SM_ARCH

DTYPE = torch.bfloat16

_GPU_DIR = os.path.dirname(os.path.abspath(__file__))
_CU_PATH = os.path.join(_GPU_DIR, "_matmul_b4_pipe.cu")

BM, BN, NW = 128, 128, 4
_BKS = [16, 32, 64, 128]
_NSS = [2, 3, 4, 5]
_LBS = [1, 2, 3, 4]

# 200 KB cap on dynamic SMEM (matches b2_ms convention).
_MAX_SMEM = 200 * 1024


def _smem(bk, ns):
    # NS × 2 × BM × BK × 2 bytes for A+B, plus 32B for tmem_holder + mbarriers.
    return ns * 2 * BM * bk * 2 + 32


def _legal(bk, ns):
    if bk not in _BKS or ns not in _NSS:
        return False
    if _smem(bk, ns) > _MAX_SMEM:
        return False
    # Only BK=128 NS=2,3 are emitted in the .cu (NS=4,5 SMEM > cap).
    if bk == 128 and ns >= 4:
        return False
    return True


_CONFIGS = [
    (bk, ns, lb)
    for bk in _BKS for ns in _NSS for lb in _LBS
    if _legal(bk, ns)
]


def _cubin_path(lb):
    return os.path.join(_GPU_DIR, f"_matmul_b4_pipe_lb{lb}_{SM_ARCH}.cubin")


def _get_mod(lb):
    return get_module_jit(_CU_PATH, _cubin_path(lb), ["-arch=sm_100a", f"-DLB_MIN_BLOCKS={lb}"])


def _kname(bk, ns):
    return f"matmul_b4_pipe_bm{BM}_bn{BN}_bk{bk}_nw{NW}_ns{ns}"


def _launch(mod, kname, A, B, smem_bytes):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device="cuda", dtype=DTYPE)
    fn = mod.get_function(kname)
    if smem_bytes > 0:
        fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes)
    block = (32, NW, 1)
    grid = (N // BN, M // BM, 1)
    fn(np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(C.data_ptr()),
       np.int32(M), np.int32(K), np.int32(N),
       block=block, grid=grid, shared=smem_bytes)
    return C


_best: dict = {}


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=DTYPE)
    B = torch.randn(K, N, device="cuda", dtype=DTYPE)

    all_lbs = sorted({c[2] for c in _CONFIGS})
    mods = {lb: _get_mod(lb) for lb in all_lbs}

    cfgs = [c for c in _CONFIGS
            if M % BM == 0 and N % BN == 0 and K % c[0] == 0
            and K // c[0] >= c[1]]

    best_t = float("inf")
    best_cfg = cfgs[0]
    n = len(cfgs)
    for idx, cfg in enumerate(cfgs):
        bk, ns, lb = cfg
        kn = _kname(bk, ns)
        sb = _smem(bk, ns)
        try:
            ms_med, _, _ = triton.testing.do_bench(
                lambda bk=bk, ns=ns, lb=lb, kn=kn, sb=sb:
                    _launch(mods[lb], kn, A, B, sb),
                warmup=10, rep=50, quantiles=(0.5, 0.0, 1.0))
        except Exception as e:
            print(f"  [{idx+1}/{n}] BK={bk} NS={ns} LB={lb}  FAILED: {e}")
            continue

        tflops = 2 * M * N * K / (ms_med / 1e3) / 1e12
        print(f"  [{idx+1:3d}/{n}] BK={bk:3d} NS={ns} LB={lb}  {tflops:6.1f} TFLOPS")
        if ms_med < best_t:
            best_t = ms_med
            best_cfg = cfg
    return best_cfg


def matmul_b4_pipe(A, B):
    M, K = A.shape
    _, N = B.shape
    key = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS
                if M % BM == 0 and N % BN == 0 and K % c[0] == 0
                and K // c[0] >= c[1]]
        print(f"[b4_pipe] autotuning {M}x{N}x{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bk, ns, lb = _best[key]
        print(f"[b4_pipe] best: BK={bk} NS={ns} LB={lb}")

    bk, ns, lb = _best[key]
    return _launch(_get_mod(lb), _kname(bk, ns), A, B, _smem(bk, ns))
