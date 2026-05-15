"""Benchmark GPU matmul implementations using triton.testing.do_bench."""

import argparse
import csv
import os
from datetime import datetime

import torch
import triton.testing

from bench_gpu import IMPLEMENTATIONS, SIZES, RESULTS_FILE, FIELDNAMES, load_fn, validate_fn, get_impl_dtype, _all_impls

WARMUP_MS = 200   # warmup budget in ms
REP_MS    = 2000  # timed budget in ms  (more reps → tighter distribution)


def gflops(M, N, K, ms):
    return 2 * M * N * K / (ms / 1e3) / 1e9


def run(impl_names, shapes):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    all_i = _all_impls()

    for name in impl_names:
        entry    = all_i.get(name, (None, None))
        max_size = entry[1]
        dtype    = get_impl_dtype(name)
        fn       = load_fn(name)
        print(f"\n[{name}]")

        for M, K, N in shapes:
            if max_size is not None and max(M, N, K) > max_size:
                print(f"  {M}x{K}x{N}: skipped (max_size={max_size})")
                continue

            A_gpu = torch.randn(M, K, dtype=dtype, device='cuda')
            B_gpu = torch.randn(K, N, dtype=dtype, device='cuda')

            try:
                validate_fn(fn, A_gpu, B_gpu)
            except AssertionError as e:
                print(f"  {M}x{K}x{N}: ✗ validation FAILED: {e}")
                continue

            ms_median, ms_min, _ = triton.testing.do_bench(
                lambda: fn(A_gpu, B_gpu),
                warmup=WARMUP_MS,
                rep=REP_MS,
                quantiles=(0.5, 0.0, 1.0),
            )
            gf = gflops(M, N, K, ms_min)

            print(f"  {M}x{K}x{N}: ✓ {gf:.2f} GFLOPS  (median {ms_median:.2f} ms, best {ms_min:.2f} ms)")

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


def _parse_shape(s):
    """Parse MxKxN or a single integer (square) into (M, K, N)."""
    parts = s.split("x")
    if len(parts) == 1:
        n = int(parts[0])
        return (n, n, n)
    if len(parts) == 3:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    raise argparse.ArgumentTypeError(f"shape must be N or MxKxN, got {s!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impls", nargs="+", default=list(_all_impls().keys()))
    parser.add_argument("--sizes", nargs="+", type=int, default=None,
                        help="Square sizes (shorthand for --shapes NxNxN)")
    parser.add_argument("--shapes", nargs="+", type=_parse_shape, default=None,
                        help="Shapes as MxKxN (e.g. 64x16384x65536) or plain N for square")
    args = parser.parse_args()

    if args.shapes is not None:
        shapes = args.shapes
    elif args.sizes is not None:
        shapes = [(s, s, s) for s in args.sizes]
    else:
        shapes = [(s, s, s) for s in SIZES]

    run(args.impls, shapes)


if __name__ == "__main__":
    main()
