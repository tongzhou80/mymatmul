"""Measure per-phase cycle breakdown of h2_s7's main K-loop.

Instruments WAIT_SMEM, COMPUTE_TILE, WAIT_MMA, LOAD_TILE with clock64() reads
to see which phase is dominant. Useful for answering "is WAIT_SMEM ever
blocking the tensor pipe?".

The instrumented kernel is in `_matmul_h2_s7_timed.cu`. We launch with the
s7-best config (BM=128, BN=256, BK=64, WG=2, NS=3) at N=4096.
"""

import ctypes, os, subprocess, threading, atexit
import numpy as np, torch
from cuda.bindings import driver as cudrvr

_HOPPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/mymatmul/gpu/hopper"
_CU_PATH    = os.path.join(_HOPPER_DIR, "_matmul_h2_s7_timed.cu")
_CUBIN      = os.path.join(_HOPPER_DIR, "_matmul_h2_s7_timed_sm_90a.cubin")
_NVCC       = "/usr/local/cuda/bin/nvcc"


def _chk(err, op=""):
    if err != cudrvr.CUresult.CUDA_SUCCESS:
        _, name = cudrvr.cuGetErrorName(err)
        _, desc = cudrvr.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error in {op}: {name.decode()} — {desc.decode()}")


def main():
    torch.cuda.init()
    (err,) = cudrvr.cuInit(0); _chk(err, "cuInit")
    err, ctx = cudrvr.cuDevicePrimaryCtxRetain(torch.cuda.current_device())
    _chk(err, "primaryCtxRetain")
    (err,) = cudrvr.cuCtxSetCurrent(ctx); _chk(err, "ctxSetCurrent")

    # Compile if needed
    if not os.path.exists(_CUBIN) or os.path.getmtime(_CU_PATH) > os.path.getmtime(_CUBIN):
        print(f"[h2s7_timed] compiling ...", end=" ", flush=True)
        r = subprocess.run([_NVCC, "-arch=sm_90a", "-O3", "--std=c++17", "--cubin",
                            "-DLB_MIN_BLOCKS=1", _CU_PATH, "-o", _CUBIN],
                           capture_output=True, text=True)
        if r.returncode != 0: raise RuntimeError(f"nvcc failed:\n{r.stderr}")
        print("done")

    err, mod = cudrvr.cuModuleLoad(_CUBIN.encode()); _chk(err, "moduleLoad")

    bm, bn, bk, nwg, ns = 128, 256, 64, 2, 3
    M = N = K = 4096

    kname = f"matmul_h2s7_timed_bm{bm}_bn{bn}_bk{bk}_wg{nwg}_ns{ns}"
    err, fn = cudrvr.cuModuleGetFunction(mod, kname.encode())
    _chk(err, f"getFn({kname})")

    # SMEM
    smem_bytes = (ns * bm * bk + ns * bk * bn) * 2
    (err,) = cudrvr.cuFuncSetAttribute(
        fn, cudrvr.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        smem_bytes)
    _chk(err, "setAttribute")

    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    grid_x = N // bn
    grid_y = M // bm
    num_ctas = grid_x * grid_y

    # Output buffer: 6 uint64 per CTA
    timings = torch.zeros(num_ctas * 6, dtype=torch.int64, device="cuda")

    c_A = ctypes.c_void_p(A.data_ptr()); c_B = ctypes.c_void_p(B.data_ptr())
    c_C = ctypes.c_void_p(C.data_ptr()); c_T = ctypes.c_void_p(timings.data_ptr())
    c_M = ctypes.c_int(M); c_K = ctypes.c_int(K); c_N = ctypes.c_int(N)
    params = np.array([ctypes.addressof(x) for x in
                       [c_A, c_B, c_C, c_T, c_M, c_K, c_N]], dtype=np.intp)

    # Warmup
    for _ in range(5):
        (err,) = cudrvr.cuLaunchKernel(fn, grid_x, grid_y, 1, nwg * 128, 1, 1,
                                       smem_bytes, 0, params, 0)
        _chk(err, "launch")
    torch.cuda.synchronize()

    # Measured run
    (err,) = cudrvr.cuLaunchKernel(fn, grid_x, grid_y, 1, nwg * 128, 1, 1,
                                   smem_bytes, 0, params, 0)
    _chk(err, "launch"); torch.cuda.synchronize()

    t = timings.cpu().numpy().reshape(num_ctas, 6)
    num_main_iters = (K // bk) - (ns - 1)

    print(f"\nh2_s7 phase breakdown — best config BM={bm} BN={bn} BK={bk} WG={nwg} NS={ns}")
    print(f"N=K=M={N},  num CTAs={num_ctas},  main-loop iters per CTA={num_main_iters}\n")
    print(f"{'Phase':<18} {'cycles/iter (mean)':>22} {'(min)':>10} {'(max)':>10}")
    print("-" * 64)
    labels = ["WAIT_SMEM", "sync_pre_cmp", "COMPUTE_TILE",
              "WAIT_MMA(1)", "sync_pre_load", "LOAD_TILE"]
    for i, label in enumerate(labels):
        per_iter = t[:, i] / num_main_iters
        print(f"{label:<18} {per_iter.mean():>22.1f} {per_iter.min():>10.1f} {per_iter.max():>10.1f}")

    total = t.sum(axis=1) / num_main_iters
    print(f"\n{'TOTAL':<18} {total.mean():>22.1f} {total.min():>10.1f} {total.max():>10.1f}")
    print(f"\n(Reference: s7 unmodified runs ~1800 cycles/iter per CTA "
          f"at 600 TFLOPS — kernel is ~4 waves over 132 SMs.)")


if __name__ == "__main__":
    main()
