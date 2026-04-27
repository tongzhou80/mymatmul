#!/usr/bin/env python3
"""
NCU Speed-of-Light profiling for matmul kernels.

Collects SM throughput, DRAM bandwidth, L1/L2 cache utilization, achieved
occupancy, and shared-memory bank conflicts for each kernel.  Results are
printed as a table and optionally saved to a CSV.

Usage:
    # Profile the default curated kernel list
    python profile_ncu_sol.py

    # Profile specific kernels from IMPLEMENTATIONS
    python profile_ncu_sol.py s4_tm8_tn8_bm128_bn128_bk16_u16 cublas_fp32_notf32

    # Set matrix size (default 4096)
    python profile_ncu_sol.py --size 4096

NCU may require elevated privileges:
    sudo python profile_ncu_sol.py
    # or: sudo sh -c "echo 0 > /proc/sys/kernel/perf_event_paranoid"
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from io import StringIO

# ---------------------------------------------------------------------------
# Default kernel selection — curated for bottleneck comparison
# ---------------------------------------------------------------------------
DEFAULT_KERNELS = [
    "cublas_fp32_notf32",
    "triton_fp32simt_bm128_bn64_bk32",
    "s4_tm8_tn8_bm128_bn128_bk16_u16",
    "s4_tm8_tn8_bm128_bn64_bk16_u16",
    "s4_tm8_tn8_bm64_bn64_bk16_u16",
    "s4b_tm8_tn8_bm128_bn128_bk16_u16",
    "s4sw_tm8_tn8_bm128_bn128_bk16_u8",
    "s3_tm8_tn8_bm128_bn128_bk32_u8",
]

# ---------------------------------------------------------------------------
# NCU metrics to collect
# ---------------------------------------------------------------------------
METRICS = {
    "sm_sol_pct":     "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_sol_pct":   "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_sol_pct":     "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_sol_pct":     "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "occupancy_pct":  "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smem_ld_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "smem_st_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
}


def create_kernel_script(dotpath: str, size: int, path: str) -> None:
    module_path, fn_name = dotpath.rsplit(".", 1)
    script = f"""\
import torch, importlib
fn = getattr(importlib.import_module("{module_path}"), "{fn_name}")
A = torch.randn({size}, {size}, dtype=torch.float32, device='cuda')
B = torch.randn({size}, {size}, dtype=torch.float32, device='cuda')
for _ in range(3):
    fn(A, B)
torch.cuda.synchronize()
fn(A, B)
torch.cuda.synchronize()
"""
    with open(path, "w") as f:
        f.write(script)


def run_ncu(script_path: str) -> str | None:
    ncu = os.environ.get("NCU", "/usr/local/cuda/bin/ncu")
    if not os.path.exists(ncu):
        # Fall back to versioned path
        for candidate in ["/usr/local/cuda-12.8/bin/ncu", "/usr/local/cuda-12.3/bin/ncu"]:
            if os.path.exists(candidate):
                ncu = candidate
                break
    cmd = [
        "sudo", "-E", ncu,
        "--metrics", ",".join(METRICS.values()),
        "--csv",
        sys.executable, script_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"    ncu error: {result.stderr.strip()}", file=sys.stderr)
        return None
    # NCU mixes ==PROF== / ==ERROR== progress lines into stdout alongside CSV;
    # strip them before handing off to the CSV parser.
    csv_lines = [l for l in result.stdout.splitlines() if not l.startswith("==")]
    return "\n".join(csv_lines)


def parse_ncu_csv(csv_text: str) -> dict:
    """Return {metric_name: float_value} for the kernel with highest SM throughput.

    NCU profiles every kernel in the script (warmup, randn, etc.).  The matmul
    kernel always has the highest SM throughput, so we select that one.
    """
    from collections import defaultdict
    # kernel_id -> {metric_name -> value}
    by_kernel: dict[str, dict] = defaultdict(dict)
    reader = csv.DictReader(StringIO(csv_text))
    for row in reader:
        kid   = row.get("ID", "").strip('"')
        name  = row.get("Metric Name", "").strip('"')
        value = row.get("Metric Value", "").strip('"').replace(",", "")
        if kid and name and value:
            try:
                by_kernel[kid][name] = float(value)
            except ValueError:
                pass
    if not by_kernel:
        return {}
    sm_key = METRICS["sm_sol_pct"]
    best = max(by_kernel.values(), key=lambda m: m.get(sm_key, 0.0))
    return best


def profile_one(name: str, dotpath: str, size: int) -> dict | None:
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        tmp = f.name
    try:
        create_kernel_script(dotpath, size, tmp)
        csv_text = run_ncu(tmp)
        if csv_text is None:
            return None
        raw = parse_ncu_csv(csv_text)
        return {k: raw.get(v) for k, v in METRICS.items()}
    finally:
        os.unlink(tmp)


def print_table(rows: list[tuple[str, dict]]) -> None:
    hdr = f"{'Kernel':<46} {'SM%':>5} {'DRAM%':>6} {'L1%':>5} {'L2%':>5} {'Occ%':>5}  {'LD-conflicts':>13} {'ST-conflicts':>13}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, m in rows:
        def v(k): return m[k] if m[k] is not None else float("nan")
        print(
            f"  {name:<44} "
            f"{v('sm_sol_pct'):5.1f} "
            f"{v('dram_sol_pct'):6.1f} "
            f"{v('l1_sol_pct'):5.1f} "
            f"{v('l2_sol_pct'):5.1f} "
            f"{v('occupancy_pct'):5.1f}  "
            f"{v('smem_ld_conflicts'):13.0f} "
            f"{v('smem_st_conflicts'):13.0f}"
        )


def save_csv(rows: list[tuple[str, dict]], path: str) -> None:
    fieldnames = ["kernel"] + list(METRICS.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, m in rows:
            w.writerow({"kernel": name, **{k: (m[k] if m[k] is not None else "") for k in METRICS}})
    print(f"\nSaved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("impls", nargs="*", default=DEFAULT_KERNELS,
                        help="Kernel names from IMPLEMENTATIONS (default: curated set)")
    parser.add_argument("--size", type=int, default=4096, metavar="N",
                        help="Square matrix size (default: 4096)")
    parser.add_argument("--out", default=None, metavar="FILE",
                        help="Save results to CSV file")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench_gpu import IMPLEMENTATIONS

    rows = []
    for name in args.impls:
        if name not in IMPLEMENTATIONS:
            print(f"  [{name}] not in IMPLEMENTATIONS, skipping")
            continue
        dotpath, _ = IMPLEMENTATIONS[name]
        print(f"  [{name}] profiling {args.size}³...", end=" ", flush=True)
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
