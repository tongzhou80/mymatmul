"""Numerical accuracy comparison of BF16 matmul implementations vs FP32 reference.

Inputs are scaled by 1/sqrt(K) so output variance stays ~1 regardless of K,
making absolute errors comparable across shapes. Reports max/mean absolute error,
max relative error, and cosine similarity.

Usage:
    python check_numerical.py                        # default shapes
    python check_numerical.py --shapes 4096x4096x4096 64x16384x65536
    python check_numerical.py --impls tc5_regpruned cublas_bf16
"""

import argparse
import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from bench import load_fn, get_impl_dtype

DEFAULT_IMPLS = ["tc5_regpruned", "cublas_bf16", "triton_bf16_autotuned"]

DEFAULT_SHAPES = [
    (1024,  1024,  1024),
    (2048,  2048,  2048),
    (4096,  4096,  4096),
    (8192,  8192,  8192),
    (64,    16384, 65536),
    (128,   8192,  65536),
]


def _parse_shape(s):
    parts = s.split("x")
    if len(parts) == 1:
        n = int(parts[0])
        return (n, n, n)
    if len(parts) == 3:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    raise argparse.ArgumentTypeError(f"shape must be N or MxKxN, got {s!r}")


def run(impl_names, shapes, seed=42):
    torch.manual_seed(seed)

    col_w = max(len(n) for n in impl_names) + 2
    hdr = (f"{'Shape':<24} | {'impl':<{col_w}} | "
           f"{'max abs':>8} | {'mean abs':>10} | {'max rel':>8} | {'cos sim':>9}")
    print(hdr)
    print("-" * len(hdr))

    for (M, K, N) in shapes:
        fns = {}
        for name in impl_names:
            dtype = get_impl_dtype(name)
            if dtype != torch.bfloat16:
                print(f"  skipping {name}: not a BF16 impl (dtype={dtype})")
                continue
            fns[name] = load_fn(name)

        scale = K ** -0.5
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * scale
        B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * scale
        ref = (A.float() @ B.float())

        for name, fn in fns.items():
            try:
                out = fn(A, B).float()
            except Exception as e:
                print(f"{str((M,K,N)):<24} | {name:<{col_w}} | FAILED: {e}")
                continue
            diff = (out - ref).abs()
            rel  = diff / (ref.abs() + 1e-5)
            cos  = F.cosine_similarity(out.flatten().unsqueeze(0),
                                       ref.flatten().unsqueeze(0)).item()
            print(
                f"{str((M,K,N)):<24} | {name:<{col_w}} | "
                f"{diff.max().item():>8.5f} | "
                f"{diff.mean().item():>10.7f} | "
                f"{rel.max().item():>8.5f} | "
                f"{cos:>9.7f}"
            )
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impls", nargs="+", default=DEFAULT_IMPLS)
    parser.add_argument("--shapes", nargs="+", type=_parse_shape, default=DEFAULT_SHAPES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.impls, args.shapes, args.seed)


if __name__ == "__main__":
    main()
