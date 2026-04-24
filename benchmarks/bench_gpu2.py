"""Benchmark GPU matmul implementations using triton.testing.do_bench."""

import argparse
import csv
import os
from datetime import datetime

import torch
import triton.testing

from bench_gpu import IMPLEMENTATIONS, SIZES, RESULTS_FILE, FIELDNAMES, load_fn, validate_fn

WARMUP_MS = 100   # warmup budget in ms
REP_MS    = 500   # timed budget in ms  (more reps → tighter distribution)


def gflops(M, N, K, ms):
    return 2 * M * N * K / (ms / 1e3) / 1e9


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

            A_gpu = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            B_gpu = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')

            try:
                validate_fn(fn, A_gpu, B_gpu)
            except AssertionError as e:
                print(f"  {M}x{N}x{K}: ✗ validation FAILED: {e}")
                continue

            ms_median, ms_min, _ = triton.testing.do_bench(
                lambda: fn(A_gpu, B_gpu),
                warmup=WARMUP_MS,
                rep=REP_MS,
                quantiles=(0.5, 0.0, 1.0),
            )
            gf = gflops(M, N, K, ms_min)

            print(f"  {M}x{N}x{K}: ✓ {gf:.2f} GFLOPS  (median {ms_median:.2f} ms, best {ms_min:.2f} ms)")

            rows.append({
                "timestamp": timestamp,
                "impl": name,
                "M": M, "N": N, "K": K,
                "gflops": f"{gf:.4f}",
                "ms_mean": f"{ms_median:.3f}",
                "ms_min": f"{ms_min:.3f}",
            })

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
    parser.add_argument("--impls", nargs="+", default=list(IMPLEMENTATIONS.keys()))
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    args = parser.parse_args()

    impls = {k: IMPLEMENTATIONS[k] for k in args.impls}
    run(impls, args.sizes)


if __name__ == "__main__":
    main()
