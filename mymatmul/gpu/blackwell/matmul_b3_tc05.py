"""b3_tc05: first Blackwell tcgen05 kernel — single config (128x128x16, NW=4)."""

import os
import numpy as np
import torch
import pycuda.driver as drv

from .._pycuda_loader import get_module_jit, SM_ARCH

DTYPE = torch.bfloat16
_GPU_DIR = os.path.dirname(os.path.abspath(__file__))
_CU_PATH = os.path.join(_GPU_DIR, "_matmul_b3_tc05.cu")
_CUBIN   = os.path.join(_GPU_DIR, f"_matmul_b3_tc05_{SM_ARCH}.cubin")

BM, BN, BK, NW = 128, 128, 16, 4
KNAME = f"matmul_b3_tc05_bm{BM}_bn{BN}_bk{BK}_nw{NW}"


def _smem():
    # A[BM][BK] + B[BN][BK] (bf16) + tmem_holder (u32) + mbar (u64).
    return (BM + BN) * BK * 2 + 16 + 16  # pad to 16B


def _get_mod():
    # SM_100a is required for tcgen05 features; nvcc -arch=sm_100 won't enable
    # SM-arch-specific PTX. Use sm_100a (architecture-specific) on Blackwell.
    return get_module_jit(_CU_PATH, _CUBIN, ["-arch=sm_100a"])


def matmul_b3_tc05(A, B):
    M, K = A.shape
    _, N = B.shape
    assert M % BM == 0 and N % BN == 0 and K % BK == 0, \
        f"b3_tc05 requires M%{BM}==0, N%{BN}==0, K%{BK}==0; got {M},{N},{K}"
    C = torch.empty(M, N, device="cuda", dtype=DTYPE)
    mod = _get_mod()
    fn = mod.get_function(KNAME)
    smem = _smem()
    fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
    block = (32, NW, 1)
    grid = (N // BN, M // BM, 1)
    fn(np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(C.data_ptr()),
       np.int32(M), np.int32(K), np.int32(N),
       block=block, grid=grid, shared=smem)
    return C
