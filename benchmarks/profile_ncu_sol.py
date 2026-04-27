#!/usr/bin/env python3
"""
NCU Speed-of-Light profiling for fp32 CUDA matmul kernels.

Uses --kernel-name to select exactly the target CUDA function in NCU.
This fixes the wrong-kernel-selection bug in the previous version, which
used a "max SM throughput" heuristic that picked unrelated torch utility
kernels for most of the custom configs.

How it works
------------
Each kernel is profiled in a separate NCU invocation.  The target script
performs 3 warmup launches of the kernel, then 1 measurement launch.
NCU is invoked with:
    ncu --kernel-name <cuda_fn_name> --metrics <list> --csv python3 target.py

Because --kernel-name filters to only the matching CUDA function, the CSV
output contains rows only for our matmul kernel.  Metrics are averaged
across all captured invocations (the 3 warmup + 1 measurement launches all
give the same hardware counters for the same matrix size).

Kernel name derivation
----------------------
Our extern "C" __global__ entry point names follow a predictable pattern:
  impl "s3_X"   -> CUDA name "matmul_cuda_s3_X"
  impl "s4_X"   -> CUDA name "matmul_cuda_s4_X"
  impl "s4b_X"  -> CUDA name "matmul_cuda_s4b_X"
  impl "s4sw_X" -> CUDA name "matmul_cuda_s4sw_X"
  impl "s3w_X"  -> CUDA name "matmul_cuda_s3_warp_X"  (note: s3w -> s3_warp)
  stage 0/1     -> explicit dict mapping
  cuBLAS/Triton -> None (no known stable name; these are skipped)

Usage
-----
    # Default curated list at 4096^3
    sudo python profile_ncu_sol.py

    # Custom size
    sudo python profile_ncu_sol.py --size 8192

    # Select specific kernels (names from bench_gpu.IMPLEMENTATIONS)
    sudo python profile_ncu_sol.py s4_tm8_tn8_bm128_bn128_bk16_u16 s4sw_tm8_tn8_bm128_bn128_bk16_u8

    # Save results
    sudo python profile_ncu_sol.py --out results_sol.csv

NCU requires elevated privileges or a relaxed paranoia level:
    sudo python profile_ncu_sol.py
    # or:  sudo sh -c "echo 0 > /proc/sys/kernel/perf_event_paranoid"
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from io import StringIO

# ---------------------------------------------------------------------------
# Default profiling targets — curated cross-stage comparison
# ---------------------------------------------------------------------------
DEFAULT_KERNELS = [
    "s3_tm8_tn8_bm128_bn128_bk32_u8",
    "s3_tm8_tn8_bm128_bn64_bk32_u8",
    "s4_tm8_tn8_bm128_bn128_bk16_u1",
    "s4_tm8_tn8_bm128_bn128_bk16_u4",
    "s4_tm8_tn8_bm128_bn128_bk16_u8",
    "s4_tm8_tn8_bm128_bn128_bk16_u16",
    "s4_tm8_tn8_bm128_bn64_bk16_u8",
    "s4_tm8_tn8_bm64_bn64_bk16_u8",
    "s4b_tm8_tn8_bm128_bn128_bk16_u8",
    "s4b_tm8_tn8_bm128_bn128_bk16_u16",
    "s4sw_tm8_tn8_bm128_bn128_bk16_u1",
    "s4sw_tm8_tn8_bm128_bn128_bk16_u4",
    "s4sw_tm8_tn8_bm128_bn128_bk16_u8",
    "s4sw_tm8_tn8_bm128_bn128_bk16_u16",
    "s4sw_tm8_tn8_bm128_bn64_bk16_u8",
    "s4sw_tm8_tn8_bm64_bn64_bk16_u8",
    "s3w_tm8_tn8_bm128_bn128_bk32_wm64_wn32_u8",
    "s3w_tm8_tn8_bm128_bn128_bk32_wm32_wn64_u8",
]

# ---------------------------------------------------------------------------
# NCU metrics
# ---------------------------------------------------------------------------
METRICS = {
    "sm_sol_pct":        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_sol_pct":      "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_sol_pct":        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_sol_pct":        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "occupancy_pct":     "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smem_ld_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "smem_st_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
}

# ---------------------------------------------------------------------------
# Stage 0/1 kernel name mapping (impl name -> CUDA __global__ function name)
# ---------------------------------------------------------------------------
_S01_NAMES = {
    "cuda_naive_ijk":          "matmul_naive_ijk_2d_grid",
    "cuda_naive_ijk_jx":       "matmul_naive_ijk_2d_grid_jx",
    "cuda_tiled_32x32":        "matmul_tiled_32x32",
    "cuda_tiled_32x32_16x16":  "matmul_tiled_32x32_16x16_threads",
    "cuda_tiled_32x32_32x8":   "matmul_tiled_32x32_32x8_threads",
    "cuda_tiled_32x32_32x4":   "matmul_tiled_32x32_32x4_threads",
    "cuda_tiled_32x64_32x4":   "matmul_tiled_32x64_32x4_threads",
    "cuda_tiled_32x64_tm4_tn4":"matmul_tiled_32x64_tm4_tn4",
}


def cuda_kernel_name(impl_name: str) -> str | None:
    """Return the exact CUDA __global__ function name for a bench impl name.

    Returns None for cuBLAS / Triton implementations where no stable name
    is known.
    """
    if impl_name.startswith("s3w_"):
        # s3w_X -> matmul_cuda_s3_warp_X
        return "matmul_cuda_s3_warp_" + impl_name[4:]
    if impl_name.startswith(("s3_", "s4_", "s4b_", "s4sw_", "s4st_")):
        return "matmul_cuda_" + impl_name
    return _S01_NAMES.get(impl_name)  # None for cuBLAS / Triton


# ---------------------------------------------------------------------------
# Target script generation
# ---------------------------------------------------------------------------

def make_target_script(dotpath: str, size: int, path: str) -> None:
    """Write a Python script that warms up the kernel 3x then runs it once."""
    module_path, fn_name = dotpath.rsplit(".", 1)
    script = f"""\
import sys
sys.path.insert(0, {repr(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))})
import torch, importlib
fn = getattr(importlib.import_module({repr(module_path)}), {repr(fn_name)})
A = torch.randn({size}, {size}, dtype=torch.float32, device='cuda')
B = torch.randn({size}, {size}, dtype=torch.float32, device='cuda')
# 3 warmup launches
for _ in range(3):
    fn(A, B)
torch.cuda.synchronize()
# 1 measurement launch (captured by NCU)
fn(A, B)
torch.cuda.synchronize()
"""
    with open(path, "w") as f:
        f.write(script)


# ---------------------------------------------------------------------------
# NCU invocation
# ---------------------------------------------------------------------------

def find_ncu() -> str:
    ncu = os.environ.get("NCU", "/usr/local/cuda/bin/ncu")
    if os.path.exists(ncu):
        return ncu
    for candidate in ["/usr/local/cuda-12.8/bin/ncu", "/usr/local/cuda-12.3/bin/ncu"]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("ncu not found; set NCU env var to point to it")


def run_ncu(script_path: str, cuda_name: str | None) -> str | None:
    """Run NCU and return raw CSV text, or None on failure."""
    ncu = find_ncu()
    metric_str = ",".join(METRICS.values())
    cmd = ["sudo", "-E", ncu, "--metrics", metric_str, "--csv"]
    if cuda_name is not None:
        # Filter to exactly our kernel — the key fix over the previous version.
        cmd += ["--kernel-name", cuda_name]
    cmd += [sys.executable, script_path]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"    ncu error:\n{result.stderr.strip()}", file=sys.stderr)
        return None

    # NCU emits ==PROF== / ==ERROR== progress lines mixed into stdout alongside CSV.
    csv_lines = [l for l in result.stdout.splitlines() if not l.startswith("==")]
    return "\n".join(csv_lines)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_ncu_csv(csv_text: str, cuda_name: str | None) -> dict:
    """Parse NCU CSV and return averaged metrics.

    When cuda_name is given (our custom kernels), every row in the CSV is for
    our matmul kernel — we average across all captured invocations.

    When cuda_name is None (cuBLAS / Triton), fall back to selecting the
    invocation with the highest SM throughput (legacy heuristic).
    """
    from collections import defaultdict

    # invocation_id -> {metric_name -> float}
    by_id: dict[str, dict] = defaultdict(dict)
    reader = csv.DictReader(StringIO(csv_text))
    for row in reader:
        kid   = row.get("ID", "").strip('"')
        name  = row.get("Metric Name", "").strip('"')
        value = row.get("Metric Value", "").strip('"').replace(",", "")
        if kid and name and value:
            try:
                by_id[kid][name] = float(value)
            except ValueError:
                pass

    if not by_id:
        return {}

    if cuda_name is not None:
        # All rows are for our kernel; average across invocations.
        all_vals: dict[str, list] = defaultdict(list)
        for metrics in by_id.values():
            for k, v in metrics.items():
                all_vals[k].append(v)
        return {k: sum(vs) / len(vs) for k, vs in all_vals.items()}
    else:
        # Fallback: pick invocation with highest SM throughput.
        sm_key = METRICS["sm_sol_pct"]
        return max(by_id.values(), key=lambda m: m.get(sm_key, 0.0))


# ---------------------------------------------------------------------------
# Per-kernel profiling
# ---------------------------------------------------------------------------

def profile_one(name: str, dotpath: str, size: int) -> dict | None:
    cuda_name = cuda_kernel_name(name)
    if cuda_name is None:
        print(f"    (no stable CUDA name for {name!r}; skipping)", file=sys.stderr)
        return None

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        tmp = f.name
    try:
        make_target_script(dotpath, size, tmp)
        csv_text = run_ncu(tmp, cuda_name)
        if csv_text is None:
            return None
        raw = parse_ncu_csv(csv_text, cuda_name)
        if not raw:
            print(f"    warning: no metrics parsed (kernel name not found in NCU output?)",
                  file=sys.stderr)
            return None
        return {k: raw.get(v) for k, v in METRICS.items()}
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(rows: list[tuple[str, dict]]) -> None:
    hdr = (
        f"{'Kernel':<48} {'SM%':>5} {'DRAM%':>6} {'L1%':>5} {'L2%':>5}"
        f" {'Occ%':>5}  {'LD-conf':>10} {'ST-conf':>10}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, m in rows:
        def v(k):
            return m[k] if m.get(k) is not None else float("nan")
        print(
            f"  {name:<46} "
            f"{v('sm_sol_pct'):5.1f} "
            f"{v('dram_sol_pct'):6.1f} "
            f"{v('l1_sol_pct'):5.1f} "
            f"{v('l2_sol_pct'):5.1f} "
            f"{v('occupancy_pct'):5.1f}  "
            f"{v('smem_ld_conflicts'):10.0f} "
            f"{v('smem_st_conflicts'):10.0f}"
        )


def save_csv(rows: list[tuple[str, dict]], path: str) -> None:
    fieldnames = ["kernel"] + list(METRICS.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, m in rows:
            w.writerow({"kernel": name,
                        **{k: (m[k] if m.get(k) is not None else "") for k in METRICS}})
    print(f"\nSaved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "impls", nargs="*", default=DEFAULT_KERNELS,
        help="Kernel names from bench_gpu.IMPLEMENTATIONS (default: curated set)",
    )
    parser.add_argument(
        "--size", type=int, default=4096, metavar="N",
        help="Square matrix size (default: 4096)",
    )
    parser.add_argument(
        "--out", default=None, metavar="FILE",
        help="Save results to CSV file",
    )
    args = parser.parse_args()

    # Load IMPLEMENTATIONS from bench_gpu (sibling module)
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, bench_dir)
    from bench_gpu import IMPLEMENTATIONS

    rows = []
    for name in args.impls:
        if name not in IMPLEMENTATIONS:
            print(f"  [{name}] not in IMPLEMENTATIONS — skipping")
            continue

        dotpath, _ = IMPLEMENTATIONS[name]
        cuda_name = cuda_kernel_name(name)
        if cuda_name is None:
            print(f"  [{name}] no stable CUDA kernel name — skipping"
                  f" (cuBLAS/Triton kernels not supported)")
            continue

        print(f"  [{name}] cuda_fn={cuda_name}  size={args.size}³ ...",
              end=" ", flush=True)
        result = profile_one(name, dotpath, args.size)
        if result is None:
            print("FAILED")
        else:
            print("done")
            rows.append((name, result))

    if rows:
        print_table(rows)
        if args.out:
            save_csv(rows, args.out)
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
