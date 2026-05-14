"""Sweep cluster shapes (CX, CY) at the s7-best config across square sizes.

For BM=128, BN=256, BK=64, WG=2, NS=3 (and NS=4), benchmark all 8 cluster
shapes generated in _matmul_h4_s2.cu and print a side-by-side table.

Usage:  python benchmarks/bench_cluster_shapes.py
"""

import argparse
import torch
import triton.testing
from mymatmul.gpu.hopper.matmul_h4_s2 import (
    _get_mod, _kname, _launch, _CONFIGS, _CLUSTERS,
)

# Focus on the configs that won in s7/h4 autotuning
FOCUS = [(128, 256, 64, 2, 3), (128, 256, 64, 2, 4)]


def gflops(M, N, K, ms):
    return 2 * M * N * K / (ms / 1e3) / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[4096, 5120, 6144, 7168, 8192, 9216, 10240])
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--rep", type=int, default=500)
    args = p.parse_args()

    mod = _get_mod()
    print(f"\n{'size':>6} {'BM/BN/BK/NS':>14} | " +
          " ".join(f"{f'({cx},{cy})':>10}" for cx, cy in _CLUSTERS))
    print("-" * (6 + 1 + 14 + 3 + 11 * len(_CLUSTERS)))

    for size in args.sizes:
        M = N = K = size
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

        for bm, bn, bk, nwg, ns in FOCUS:
            row = []
            for cx, cy in _CLUSTERS:
                # Skip cluster shapes that don't divide the grid
                grid_x = (N + bn - 1) // bn
                grid_y = (M + bm - 1) // bm
                if grid_x % cx != 0 or grid_y % cy != 0:
                    row.append("    -    ")
                    continue
                cfg = (bm, bn, bk, nwg, ns, cx, cy)
                kn = _kname(*cfg)
                try:
                    ms = triton.testing.do_bench(
                        lambda c=cfg, k=kn: _launch(mod, k, A, B, *c),
                        warmup=args.warmup, rep=args.rep, return_mode="min")
                    row.append(f"{gflops(M,N,K,ms)/1e3:9.1f}T")
                except Exception:
                    row.append("   FAIL  ")

            tag = f"{bm}/{bn}/{bk}/{ns}"
            print(f"{size:>6} {tag:>14} | " + " ".join(row))
        print()


if __name__ == "__main__":
    main()
