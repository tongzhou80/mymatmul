"""Benchmark all available matmul implementations and write results to benchmarks/results.csv."""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np

# Registry: add new versions here as they are implemented
IMPLEMENTATIONS = {
    "v0": "mymatmul.matmul_v0",
}

SIZES = [128, 256, 512, 1024, 2048]
WARMUP_RUNS = 1
TIMED_RUNS = 3

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.csv")
FIELDNAMES = ["timestamp", "impl", "M", "N", "K", "gflops", "ms_mean", "ms_min"]


def gflops(M, N, K, seconds):
    return 2 * M * N * K / seconds / 1e9


def benchmark_fn(fn, A, B):
    for _ in range(WARMUP_RUNS):
        fn(A, B)

    times = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        fn(A, B)
        times.append(time.perf_counter() - t0)
    return times


def load_fn(dotpath: str):
    module_path, fn_name = dotpath.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, fn_name)


def run(impls, sizes):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []

    for name, dotpath in impls.items():
        print(f"\n[{name}] {dotpath}")
        fn = load_fn(dotpath)

        for sz in sizes:
            M = N = K = sz
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)

            times = benchmark_fn(fn, A, B)
            ms_mean = np.mean(times) * 1e3
            ms_min = np.min(times) * 1e3
            gf = gflops(M, N, K, np.min(times))

            print(f"  {M}x{N}x{K}: {gf:.2f} GFLOPS  (mean {ms_mean:.1f} ms, best {ms_min:.1f} ms)")

            rows.append({
                "timestamp": timestamp,
                "impl": name,
                "M": M, "N": N, "K": K,
                "gflops": f"{gf:.4f}",
                "ms_mean": f"{ms_mean:.3f}",
                "ms_min": f"{ms_min:.3f}",
            })

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impls", nargs="+", default=list(IMPLEMENTATIONS.keys()),
                        help="Which implementations to benchmark (default: all)")
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES,
                        help="Matrix sizes (square, MxMxM)")
    args = parser.parse_args()

    impls = {k: IMPLEMENTATIONS[k] for k in args.impls}
    run(impls, args.sizes)


if __name__ == "__main__":
    main()
