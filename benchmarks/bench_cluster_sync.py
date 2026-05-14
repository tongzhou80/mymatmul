"""Measure cluster.sync() cost in cycles on H800.

Compares pure loop / __syncthreads / cluster.sync overheads.
"""

import ctypes, os, subprocess, statistics
import numpy as np, torch
from cuda.bindings import driver as cudrvr

_HERE  = os.path.dirname(os.path.abspath(__file__))
_CU    = os.path.join(_HERE, "_cluster_sync_bench.cu")
_NVCC  = "/usr/local/cuda/bin/nvcc"
_ARCH  = "sm_90a"
_CUBIN = os.path.join(_HERE, f"_cluster_sync_bench_{_ARCH}.cubin")

def _chk(err, op=""):
    if err != cudrvr.CUresult.CUDA_SUCCESS:
        _, name = cudrvr.cuGetErrorName(err)
        _, desc = cudrvr.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error in {op}: {name.decode()} — {desc.decode()}")

torch.cuda.init()
(err,) = cudrvr.cuInit(0); _chk(err, "cuInit")
err, ctx = cudrvr.cuDevicePrimaryCtxRetain(torch.cuda.current_device())
_chk(err, "primaryCtxRetain")
(err,) = cudrvr.cuCtxSetCurrent(ctx); _chk(err, "ctxSetCurrent")

# Compile
if not os.path.exists(_CUBIN) or os.path.getmtime(_CU) > os.path.getmtime(_CUBIN):
    print(f"compiling for {_ARCH} ...", end=" ", flush=True)
    r = subprocess.run([_NVCC, f"-arch={_ARCH}", "-O3", "--std=c++17", "--cubin",
                       _CU, "-o", _CUBIN], capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(f"nvcc failed:\n{r.stderr}")
    print("done")

err, mod = cudrvr.cuModuleLoad(_CUBIN.encode()); _chk(err, "moduleLoad")


def run_kernel(kname, n_clusters, n_iters, cluster_kernel):
    """Launch kernel with (1 × n_clusters or 1 × n_clusters*2) CTAs.
    For cluster kernel: gridDim.y must be even (cluster Y=2).
    Returns array of cycles per CTA (size = num_ctas)."""
    err, fn = cudrvr.cuModuleGetFunction(mod, kname.encode())
    _chk(err, f"getFn({kname})")

    if cluster_kernel:
        grid = (1, n_clusters * 2, 1)
    else:
        grid = (1, n_clusters * 2, 1)   # same total CTAs for fair compare

    n_ctas = grid[0] * grid[1] * grid[2]
    out = torch.zeros(n_ctas, dtype=torch.int64, device="cuda")
    c_out = ctypes.c_void_p(out.data_ptr())
    c_n   = ctypes.c_int(n_iters)
    params = np.array([ctypes.addressof(c_out), ctypes.addressof(c_n)], dtype=np.intp)
    (err,) = cudrvr.cuLaunchKernel(fn, *grid, 128, 1, 1, 0, 0, params, 0)
    _chk(err, f"launch({kname})")
    torch.cuda.synchronize()
    return out.cpu().numpy()


def measure(kname, n_clusters, n_iters, cluster_kernel):
    """Return median cycles-per-iter across CTAs, plus the raw range."""
    # Run twice; take second to skip cold-start
    _ = run_kernel(kname, n_clusters, n_iters, cluster_kernel)
    cycles = run_kernel(kname, n_clusters, n_iters, cluster_kernel)
    per_iter = cycles / n_iters
    return float(np.median(per_iter)), float(per_iter.min()), float(per_iter.max())


def main():
    n_iters = 1000
    print(f"\nMeasurement: {n_iters} iterations of each barrier in a tight loop\n")
    print(f"{'config':<40} {'cycles/iter':>14} {'(min..max)':>20}")
    print("-" * 75)

    # Single cluster (1 cluster = 2 CTAs)
    for n_clusters, label in [(1, "1 cluster (2 CTAs, no contention)"),
                              (66, "66 clusters (132 CTAs, full SM occupancy)")]:
        m, lo, hi = measure("empty_loop",      n_clusters, n_iters, False)
        print(f"  empty_loop      ({label})\n    cycles/iter median: {m:9.1f}  ({lo:.1f} .. {hi:.1f})")
        m, lo, hi = measure("syncthreads_loop", n_clusters, n_iters, False)
        print(f"  __syncthreads   ({label})\n    cycles/iter median: {m:9.1f}  ({lo:.1f} .. {hi:.1f})")
        m, lo, hi = measure("cluster_sync_loop", n_clusters, n_iters, True)
        print(f"  cluster.sync()  ({label})\n    cycles/iter median: {m:9.1f}  ({lo:.1f} .. {hi:.1f})")
        print()


if __name__ == "__main__":
    main()
