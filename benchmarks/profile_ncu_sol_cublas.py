#!/usr/bin/env python3
"""
NCU Speed-of-Light profiling for cuBLAS and Triton fp32 matmul.

cuBLAS and Triton use internally-named CUDA kernels whose names are not
stable across driver/library versions.  We therefore do NOT pass
--kernel-name to NCU; instead we capture all kernels and pick the one
with the highest SM throughput — which reliably identifies the main
compute kernel for large square matrices.

Usage
-----
    # Default: cuBLAS + all Triton fp32simt configs at 4096³
    sudo python profile_ncu_sol_cublas.py

    # Custom size
    sudo python profile_ncu_sol_cublas.py --size 8192

    # cuBLAS only
    sudo python profile_ncu_sol_cublas.py --impls cublas_fp32_notf32

    # Save results
    sudo python profile_ncu_sol_cublas.py --out results_cublas_sol.csv

    # Compare alongside custom kernels (reads results_ncu_sol_4096.csv)
    sudo python profile_ncu_sol_cublas.py --compare benchmarks/results_ncu_sol_4096.csv

NCU requires elevated privileges:
    sudo python profile_ncu_sol_cublas.py
    # or:  sudo sh -c "echo 0 > /proc/sys/kernel/perf_event_paranoia"
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from io import StringIO

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
DEFAULT_IMPLS = [
    "cublas_fp32_notf32",
    "triton_fp32simt_bm128_bn128_bk16",
    "triton_fp32simt_bm128_bn64_bk16",
    "triton_fp32simt_bm64_bn64_bk16",
    "triton_fp32simt_bm128_bn128_bk32",
    "triton_fp32simt_bm128_bn64_bk32",
    "triton_fp32simt_bm64_bn64_bk32",
]

# ---------------------------------------------------------------------------
# NCU metrics (same set as profile_ncu_sol.py for direct comparison)
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
# Target script generation
# ---------------------------------------------------------------------------

def make_target_script(dotpath: str, size: int, path: str) -> None:
    """Write a warmup-then-measure script for the given impl dotpath."""
    module_path, fn_name = dotpath.rsplit(".", 1)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = f"""\
import sys
sys.path.insert(0, {repr(repo_root)})
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
    raise FileNotFoundError("ncu not found; set NCU env var")


def run_ncu(script_path: str) -> str | None:
    """Run NCU without --kernel-name (captures all kernels) and return raw CSV."""
    ncu = find_ncu()
    metric_str = ",".join(METRICS.values())
    cmd = ["sudo", "-E", ncu, "--metrics", metric_str, "--csv",
           sys.executable, script_path]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"    ncu error:\n{result.stderr.strip()}", file=sys.stderr)
        return None

    csv_lines = [l for l in result.stdout.splitlines() if not l.startswith("==")]
    return "\n".join(csv_lines)


# ---------------------------------------------------------------------------
# CSV parsing — max SM throughput heuristic
# ---------------------------------------------------------------------------

def parse_ncu_csv_max_sm(csv_text: str) -> dict:
    """Return metrics for the invocation with the highest SM throughput.

    This is the best available heuristic for cuBLAS/Triton where the kernel
    name is unknown.  For a pure compute workload on a large matrix the main
    matmul kernel dominates SM utilisation.
    """
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

    sm_key = METRICS["sm_sol_pct"]
    best = max(by_id.values(), key=lambda m: m.get(sm_key, 0.0))
    return best


# ---------------------------------------------------------------------------
# Per-impl profiling
# ---------------------------------------------------------------------------

def profile_one(dotpath: str, size: int) -> dict | None:
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        tmp = f.name
    try:
        make_target_script(dotpath, size, tmp)
        csv_text = run_ncu(tmp)
        if csv_text is None:
            return None
        raw = parse_ncu_csv_max_sm(csv_text)
        if not raw:
            print("    warning: no metrics parsed", file=sys.stderr)
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
        def v(k, _m=m):
            return _m[k] if _m.get(k) is not None else float("nan")
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


def load_comparison_csv(path: str) -> list[tuple[str, dict]]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.pop("kernel")
            metrics = {}
            for k in METRICS:
                try:
                    metrics[k] = float(row[k]) if row.get(k) else None
                except (ValueError, KeyError):
                    metrics[k] = None
            rows.append((name, metrics))
    return rows


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
        "--impls", nargs="+", default=DEFAULT_IMPLS,
        metavar="IMPL",
        help="Impl names from bench_gpu.IMPLEMENTATIONS (default: cuBLAS + Triton configs)",
    )
    parser.add_argument(
        "--size", type=int, default=4096, metavar="N",
        help="Square matrix size (default: 4096)",
    )
    parser.add_argument(
        "--out", default=None, metavar="FILE",
        help="Save results to CSV file",
    )
    parser.add_argument(
        "--compare", default=None, metavar="FILE",
        help="Also print rows from a results CSV (e.g. results_ncu_sol_4096.csv)"
             " for side-by-side comparison",
    )
    args = parser.parse_args()

    bench_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, bench_dir)
    from bench_gpu import IMPLEMENTATIONS

    rows: list[tuple[str, dict]] = []
    for name in args.impls:
        if name not in IMPLEMENTATIONS:
            print(f"  [{name}] not in IMPLEMENTATIONS — skipping")
            continue
        dotpath, _ = IMPLEMENTATIONS[name]
        print(f"  [{name}]  size={args.size}³  (max-SM heuristic) ...",
              end=" ", flush=True)
        result = profile_one(dotpath, args.size)
        if result is None:
            print("FAILED")
        else:
            print("done")
            rows.append((name, result))

    if not rows:
        print("No results collected.")
        return

    # Optionally prepend comparison rows from an existing CSV.
    if args.compare and os.path.exists(args.compare):
        cmp_rows = load_comparison_csv(args.compare)
        print(f"\n--- comparison rows from {args.compare} ---")
        all_rows = cmp_rows + [("---", {k: None for k in METRICS})] + rows
    else:
        all_rows = rows

    print_table(all_rows)

    if args.out:
        save_csv(rows, args.out)


if __name__ == "__main__":
    main()
