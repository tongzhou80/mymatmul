"""H2 Stage 1: TMA + mbarrier async loads; mma.sync compute; no SMEM swizzle.

Host side rewritten to use cuda-python (cuda.bindings.driver) instead of PyCUDA,
since Hopper host setup (TMA descriptors, future wgmma state) is cleaner via the
official CUDA driver bindings.

What changes vs tc5_regpruned
------------------------------
  Data movement : __pipeline_memcpy_async (per thread) → TMA bulk 2D copy (1 thread)
  Synchronisation: __pipeline_wait_prior + __syncthreads → mbarrier wait-parity
  SMEM layout   : XOR-swizzled → linear (bank conflicts present, intentional for S1)
  Compute       : unchanged — same ldmatrix + mma.sync as tc5_lb
"""

import ctypes
import os
import subprocess
import threading
import atexit

import numpy as np
import torch
import triton.testing

from cuda.bindings import driver as cudrvr

# ── CUDA context (shared with PyTorch) ───────────────────────────────────────

_ctx_ready = False
_ctx_lock  = threading.Lock()


def _ensure_ctx():
    global _ctx_ready
    with _ctx_lock:
        if _ctx_ready:
            return
        torch.cuda.init()
        (err,) = cudrvr.cuInit(0)
        _chk(err, "cuInit")
        err, ctx = cudrvr.cuDevicePrimaryCtxRetain(torch.cuda.current_device())
        _chk(err, "cuDevicePrimaryCtxRetain")
        (err,) = cudrvr.cuCtxSetCurrent(ctx)
        _chk(err, "cuCtxSetCurrent")
        atexit.register(lambda: cudrvr.cuDevicePrimaryCtxRelease(torch.cuda.current_device()))
        _ctx_ready = True


def _chk(err, op=""):
    if err != cudrvr.CUresult.CUDA_SUCCESS:
        _, name = cudrvr.cuGetErrorName(err)
        _, desc = cudrvr.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error in {op}: {name.decode()} — {desc.decode()}")


# ── Compilation ───────────────────────────────────────────────────────────────

_HOPPER_DIR = os.path.dirname(os.path.abspath(__file__))
_CU_PATH    = os.path.join(_HOPPER_DIR, "_matmul_h2_s1.cu")
_NVCC       = "/usr/local/cuda/bin/nvcc"
_ARCH       = "sm_90a"   # TMA + mbarrier require sm_90a

_mod_lock = threading.Lock()
_modules: dict[int, object] = {}   # lb → CUmodule


def _cubin_path(lb: int) -> str:
    return os.path.join(_HOPPER_DIR, f"_matmul_h2s1_lb{lb}_{_ARCH}.cubin")


def _compile(lb: int) -> None:
    cubin = _cubin_path(lb)
    if os.path.exists(cubin) and os.path.getmtime(_CU_PATH) <= os.path.getmtime(cubin):
        return
    print(f"[h2s1] compiling LB={lb} for {_ARCH} ...", end=" ", flush=True)
    cmd = [_NVCC, f"-arch={_ARCH}", "-O3", "--std=c++17", "--cubin",
           f"-DLB_MIN_BLOCKS={lb}", _CU_PATH, "-o", cubin]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{r.stderr}")
    print("done")


def _get_mod(lb: int):
    with _mod_lock:
        if lb not in _modules:
            _ensure_ctx()
            _compile(lb)
            err, mod = cudrvr.cuModuleLoad(_cubin_path(lb).encode())
            _chk(err, "cuModuleLoad")
            _modules[lb] = mod
    return _modules[lb]


# ── TMA descriptor construction ───────────────────────────────────────────────
#
# cuTensorMapEncodeTiled encodes tile shape, global tensor strides, swizzle, etc.
# into a 128-byte opaque descriptor.  The kernel receives the GPU address of this
# descriptor; it never looks inside it — the TMA hardware reads it.
#
# Dimension convention (innermost first):
#   For a row-major [rows, cols] BF16 tensor:
#     globalDim     = (cols, rows)          — innermost (cols) first
#     globalStrides = (cols * 2,)           — byte stride of the outer dim
#     boxDim        = (box_cols, box_rows)  — tile shape, innermost first

_DTYPE_BF16       = cudrvr.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
_SWIZZLE_NONE     = cudrvr.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
_SWIZZLE_128B     = cudrvr.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
_INTERLEAVE_NONE  = cudrvr.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE
_L2_PROMO_NONE    = cudrvr.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE
_OOB_FILL_NONE    = cudrvr.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE


def _make_tma_desc(ptr_int: int, nrows: int, ncols: int,
                   box_rows: int, box_cols: int,
                   swizzle=None) -> bytes:
    """Return the 128-byte TMA descriptor as a Python bytes object."""
    if swizzle is None:
        swizzle = _SWIZZLE_NONE
    u32 = cudrvr.cuuint32_t
    u64 = cudrvr.cuuint64_t
    err, tmap = cudrvr.cuTensorMapEncodeTiled(
        _DTYPE_BF16,
        2,                                      # rank
        ptr_int,                                # global base address
        [u64(ncols), u64(nrows)],               # globalDim:     [cols, rows]
        [u64(ncols * 2)],                       # globalStrides: bytes per row
        [u32(box_cols), u32(box_rows)],         # boxDim:        [cols, rows]
        [u32(1), u32(1)],                       # elementStrides: always 1
        _INTERLEAVE_NONE,
        swizzle,
        _L2_PROMO_NONE,
        _OOB_FILL_NONE,
    )
    _chk(err, "cuTensorMapEncodeTiled")
    # Extract the 16 × uint64 opaque words as raw bytes.
    return np.array([int(v) for v in tmap.opaque], dtype=np.uint64).tobytes()


# ── Config space (same constraints as tc5_regpruned) ─────────────────────────

_BMS = [64, 128, 256]
_BNS = [64, 128, 256]
_BKS = [16, 32, 64]
_NWS = [4, 8]
_LB_FOR_NW = {4: [1, 2, 3, 4], 8: [1, 2]}
_MAX_SMEM   = 100352


def _smem(bm, bn, bk):
    ab = (2 * bm * bk + 2 * bk * bn) * 2   # bf16 tiles, 2 pipeline stages
    return ab + 16                           # +16 for two 8-byte mbarriers


def _reg_estimate(bm, bn, bk, nw):
    wm = (bm * 2) // (nw * 16)
    wn = bn // 16
    kk = bk // 16
    return wm * wn * 4 + kk * (wm * 4 + wn * 2)


_CONFIGS = [
    (bm, bn, bk, nw, lb)
    for bm in _BMS for bn in _BNS for bk in _BKS for nw in _NWS
    for lb in _LB_FOR_NW[nw]
    if _smem(bm, bn, bk) <= _MAX_SMEM
    and bm * bn <= 4096 * nw
    and _reg_estimate(bm, bn, bk, nw) <= 65536 // (nw * 32 * lb)
]


def _kname(bm, bn, bk, nw):
    return f"matmul_h2s1_bm{bm}_bn{bn}_bk{bk}_nw{nw}"


# ── Kernel launch ─────────────────────────────────────────────────────────────

def _get_fn(mod, kname: str):
    err, fn = cudrvr.cuModuleGetFunction(mod, kname.encode())
    _chk(err, f"cuModuleGetFunction({kname})")
    return fn


def _set_max_smem(fn, smem_bytes: int):
    (err,) = cudrvr.cuFuncSetAttribute(
        fn,
        cudrvr.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        smem_bytes,
    )
    _chk(err, "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)")


def _launch(mod, kname, A, B, nw, bm, bn, bk, lb):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    fn = _get_fn(mod, kname)
    sb = _smem(bm, bn, bk)
    _set_max_smem(fn, sb)

    # Build TMA descriptors on the CPU (pure arithmetic, no GPU alloc/copy).
    # Pass as __grid_constant__ by value: kernelParams[i] points to the 128-byte
    # struct; the CUDA runtime copies it into per-CTA constant memory at launch.
    buf_a = ctypes.create_string_buffer(_make_tma_desc(A.data_ptr(), M, K, bm, bk), 128)
    buf_b = ctypes.create_string_buffer(_make_tma_desc(B.data_ptr(), K, N, bk, bn), 128)
    c_C   = ctypes.c_void_p(C.data_ptr())
    c_M   = ctypes.c_int(M)
    c_K   = ctypes.c_int(K)
    c_N   = ctypes.c_int(N)
    params = np.array([ctypes.addressof(buf_a), ctypes.addressof(buf_b),
                       ctypes.addressof(c_C), ctypes.addressof(c_M),
                       ctypes.addressof(c_K), ctypes.addressof(c_N)],
                      dtype=np.intp)

    grid = ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)
    (err,) = cudrvr.cuLaunchKernel(
        fn,
        grid[0], grid[1], grid[2],
        32, nw, 1,
        sb, 0,           # sharedMemBytes, stream=0 (default)
        params, 0,
    )
    _chk(err, f"cuLaunchKernel({kname})")
    return C


# ── Autotuner ─────────────────────────────────────────────────────────────────

_best: dict = {}
DTYPE = torch.bfloat16


def _tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    mods   = {lb: _get_mod(lb) for lb in [1, 2, 3, 4]}
    cfgs   = [c for c in _CONFIGS if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0]
    best_t, best_cfg = float("inf"), cfgs[0]

    for idx, (bm, bn, bk, nw, lb) in enumerate(cfgs):
        kn = _kname(bm, bn, bk, nw)
        try:
            _, ms, _ = triton.testing.do_bench(
                lambda bm=bm, bn=bn, bk=bk, nw=nw, lb=lb, kn=kn:
                    _launch(mods[lb], kn, A, B, nw, bm, bn, bk, lb),
                warmup=10, rep=50, quantiles=(0.5, 0.0, 1.0),
            )
        except Exception as e:
            print(f"  [{idx+1}/{len(cfgs)}] {kn} LB={lb} FAILED: {e}")
            continue
        tflops = 2 * M * N * K / (ms / 1e3) / 1e12
        print(f"  [{idx+1:3d}/{len(cfgs)}] BM={bm:3d} BN={bn:3d} BK={bk:2d} NW={nw} LB={lb}  {tflops:6.1f} TFLOPS")
        if ms < best_t:
            best_t, best_cfg = ms, (bm, bn, bk, nw, lb)
    return best_cfg


def matmul_h2_s1(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    _, N = B.shape
    key  = (M, N, K)
    if key not in _best:
        cfgs = [c for c in _CONFIGS if M % c[0] == 0 and N % c[1] == 0 and K % c[2] == 0]
        print(f"[h2_s1] autotuning {M}×{N}×{K} over {len(cfgs)} configs ...")
        _best[key] = _tune(M, N, K)
        bm, bn, bk, nw, lb = _best[key]
        print(f"[h2_s1] best: BM={bm} BN={bn} BK={bk} NW={nw} LB={lb}")

    bm, bn, bk, nw, lb = _best[key]
    return _launch(_get_mod(lb), _kname(bm, bn, bk, nw), A, B, nw, bm, bn, bk, lb)
