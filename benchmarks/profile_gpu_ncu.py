#!/usr/bin/env python3
"""Profile GPU matmul implementations using NVIDIA Compute Utilities (ncu).

Generates ncu profiling reports for each kernel to identify bottlenecks like
shared memory bank conflicts, memory bandwidth utilization, etc.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Registry: name -> dotpath
IMPLEMENTATIONS = {
    "cuda_naive_ijk": "mymatmul.gpu.matmul_cuda.matmul_cuda_naive_ijk",
    "cuda_naive_ijk_jx": "mymatmul.gpu.matmul_cuda.matmul_cuda_naive_ijk_jx",
    "cuda_tiled_32x32": "mymatmul.gpu.matmul_cuda.matmul_cuda_tiled_32x32",
    "cuda_tiled_32x32_16x16": "mymatmul.gpu.matmul_cuda.matmul_cuda_tiled_32x32_threads_16x16",
}

PROFILE_DIR = os.path.join(os.path.dirname(__file__), "profiles_ncu")


def load_fn(dotpath: str):
    module_path, fn_name = dotpath.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, fn_name)


def create_profile_script(fn_name, fn, M, N, K, output_path):
    """Create a standalone Python script that ncu can profile."""
    script = f'''
import torch
from mymatmul.gpu.matmul_cuda import {fn.__name__}

# Create input tensors on GPU
A_gpu = torch.randn({M}, {K}, dtype=torch.bfloat16, device='cuda')
B_gpu = torch.randn({K}, {N}, dtype=torch.bfloat16, device='cuda')

# Warm up
torch.cuda.synchronize()
{fn.__name__}(A_gpu, B_gpu)
torch.cuda.synchronize()

# Run kernel for profiling
for _ in range(5):
    {fn.__name__}(A_gpu, B_gpu)
    torch.cuda.synchronize()
'''
    with open(output_path, 'w') as f:
        f.write(script)


def profile_kernel(impl_name, fn_name, dotpath, matrix_size, ncu_config="default"):
    """Run ncu profiling on a single kernel."""
    print(f"\n[{impl_name}] Profiling at {matrix_size}³...")

    # Load the function to get the actual Python function name
    fn = load_fn(dotpath)

    # Create temporary script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        create_profile_script(impl_name, fn, matrix_size, matrix_size, matrix_size, f.name)
        script_path = f.name

    try:
        # Create output directory
        os.makedirs(PROFILE_DIR, exist_ok=True)

        # Output file for ncu results
        output_file = os.path.join(PROFILE_DIR, f"{impl_name}_{matrix_size}.ncu-rep")

        # Run ncu profiling
        cmd = [
            "ncu",
            "--set", ncu_config,
            "--export", output_file,
            "--force",  # Overwrite existing
            "python", script_path
        ]

        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"  ✗ Error: {result.stderr}")
            return False

        print(f"  ✓ Profile saved to {output_file}")
        return True

    except subprocess.TimeoutExpired:
        print(f"  ✗ Profiling timed out")
        return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False
    finally:
        # Clean up temporary script
        if os.path.exists(script_path):
            os.remove(script_path)


def main():
    parser = argparse.ArgumentParser(description="Profile GPU matmul kernels with ncu")
    parser.add_argument("--size", type=int, default=512,
                       help="Matrix size for profiling (NxNxN)")
    parser.add_argument("--config", default="default",
                       help="ncu metrics configuration (default, full, roofline, etc.)")
    parser.add_argument("--impls", nargs="+", default=list(IMPLEMENTATIONS.keys()),
                       help="Which implementations to profile")
    args = parser.parse_args()

    print(f"Profiling GPU matmul kernels at {args.size}³ with ncu")
    print(f"Configuration: {args.config}")
    print(f"Output directory: {PROFILE_DIR}")

    for impl_name in args.impls:
        if impl_name not in IMPLEMENTATIONS:
            print(f"  ✗ Unknown implementation: {impl_name}")
            continue

        dotpath = IMPLEMENTATIONS[impl_name]
        fn = load_fn(dotpath)
        success = profile_kernel(impl_name, fn.__name__, dotpath, args.size, args.config)

        if not success:
            print(f"  Failed to profile {impl_name}")

    print(f"\nProfiles saved to: {PROFILE_DIR}")
    print("\nTo view results, use:")
    print("  ncu-ui <profile_file>.ncu-rep")


if __name__ == "__main__":
    main()
