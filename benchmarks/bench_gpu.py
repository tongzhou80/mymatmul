"""Benchmark GPU matmul implementations and write results to benchmarks/results_gpu.csv."""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import torch

# Registry: name -> (dotpath, max_size or None for no limit)
# max_size: skip sizes larger than this (leaves empty cells in the results table)
IMPLEMENTATIONS = {
    "torch_matmul": ("mymatmul.gpu.matmul_torch.matmul_torch",   None),
    "cuda_naive_ijk": ("mymatmul.gpu.matmul_cuda.matmul_cuda_naive_ijk", None),
    "cuda_naive_ijk_jx": ("mymatmul.gpu.matmul_cuda.matmul_cuda_naive_ijk_jx", None),
    "cuda_tiled_32x32": ("mymatmul.gpu.matmul_cuda.matmul_cuda_tiled_32x32", None),
}

SIZES = [64, 128, 256, 512, 1024, 2048, 4096]
WARMUP_RUNS = 1
TIMED_RUNS = 3

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results_gpu.csv")
FIELDNAMES = ["timestamp", "impl", "M", "N", "K", "gflops", "ms_mean", "ms_min"]


def gflops(M, N, K, seconds):
    return 2 * M * N * K / seconds / 1e9


def benchmark_fn(fn, A_gpu, B_gpu):
    # Warmup with GPU synchronization
    for _ in range(WARMUP_RUNS):
        fn(A_gpu, B_gpu)
        torch.cuda.synchronize()

    # Timed runs with synchronization
    times = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        fn(A_gpu, B_gpu)
        torch.cuda.synchronize()  # Wait for GPU to finish before stopping timer
        times.append(time.perf_counter() - t0)
    return times


def load_fn(dotpath: str):
    module_path, fn_name = dotpath.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, fn_name)


def validate_fn(fn, A_gpu, B_gpu, rtol=1e-2, atol=1e-1):
    result = fn(A_gpu, B_gpu).float()
    expected = torch.matmul(A_gpu.float(), B_gpu.float())

    diff = (result - expected).abs()
    rel = diff / expected.abs().clamp_min(1e-3)

    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    p99_abs = diff.quantile(0.99).item()
    max_rel = rel.max().item()
    mean_rel = rel.mean().item()

    if not torch.allclose(result, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"Mismatch: max_abs={max_abs:.3e}, p99_abs={p99_abs:.3e}, "
            f"mean_abs={mean_abs:.3e}, max_rel={max_rel:.3e}, mean_rel={mean_rel:.3e}"
        )
    return True


def run(impls, sizes):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []

    for name, (dotpath, max_size) in impls.items():
        print(f"\n[{name}] {dotpath}")
        fn = load_fn(dotpath)

        for sz in sizes:
            M = N = K = sz

            if max_size is not None and sz > max_size:
                print(f"  {M}x{N}x{K}: skipped (max_size={max_size})")
                continue

            # Create tensors directly on GPU
            A_gpu = torch.randn(M, K, dtype=torch.float16, device='cuda')
            B_gpu = torch.randn(K, N, dtype=torch.float16, device='cuda')

            # Validate result before benchmarking
            try:
                validate_fn(fn, A_gpu, B_gpu)
            except AssertionError as e:
                print(f"  {M}x{N}x{K}: ✗ validation FAILED: {e}")
                continue

            # Time only the GPU computation
            times = benchmark_fn(fn, A_gpu, B_gpu)
            ms_mean = np.mean(times) * 1e3
            ms_min = np.min(times) * 1e3
            gf = gflops(M, N, K, np.min(times))

            print(f"  {M}x{N}x{K}: ✓ {gf:.2f} GFLOPS  (mean {ms_mean:.1f} ms, best {ms_min:.1f} ms)")

            rows.append({
                "timestamp": timestamp,
                "impl": name,
                "M": M, "N": N, "K": K,
                "gflops": f"{gf:.4f}",
                "ms_mean": f"{ms_mean:.3f}",
                "ms_min": f"{ms_min:.3f}",
            })

    # Merge with existing results: new rows overwrite matching (impl, M, N, K) keys
    existing = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, newline="") as f:
            for row in csv.DictReader(f):
                existing[(row["impl"], row["M"], row["N"], row["K"])] = row
    for row in rows:
        existing[(row["impl"], row["M"], row["N"], row["K"])] = row

    merged = sorted(existing.values(), key=lambda r: (r["impl"], int(r["M"])))
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(merged)

    print(f"\nResults written to {RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impls", nargs="+", default=list(IMPLEMENTATIONS.keys()),
                        help="Which implementations to benchmark (default: all)")
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES,
                        help="Matrix sizes to benchmark (square MxMxM)")
    args = parser.parse_args()

    impls = {k: IMPLEMENTATIONS[k] for k in args.impls}
    run(impls, args.sizes)


if __name__ == "__main__":
    main()
