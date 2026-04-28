#!/usr/bin/env python3
"""
NCU occupancy analysis: registers, shared memory, and limiting factors.

For each kernel, collects:
  - threads per block (launch config)
  - registers per thread
  - static shared memory per block (bytes)
  - theoretical occupancy (warps/SM at full utilisation)
  - achieved occupancy (%)
  - the resource that limits occupancy: registers / shared_mem / block_size / warps

Usage:
    python profile_ncu_occupancy.py
    python profile_ncu_occupancy.py s4_tm8_tn8_bm128_bn128_bk16_u16 s3_tm8_tn8_bm128_bn128_bk32_u8
    python profile_ncu_occupancy.py --size 4096 --out results_occupancy.csv

NCU may require elevated privileges:
    sudo python profile_ncu_occupancy.py
    # or: sudo sh -c "echo 0 > /proc/sys/kernel/perf_event_paranoid"
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from io import StringIO

DEFAULT_KERNELS = [
    "cublas_fp32_notf32",
    "triton_fp32simt_bm128_bn64_bk32",
    "s3_tm8_tn8_bm128_bn128_bk32_u8",
    "s4_tm8_tn8_bm128_bn128_bk16_u8",
    "s4_tm8_tn8_bm128_bn128_bk16_u16",
    "s4_tm8_tn8_bm128_bn64_bk16_u16",
    "s4_tm8_tn8_bm64_bn64_bk16_u16",
    "s4b_tm8_tn8_bm128_bn128_bk16_u16",
    "s4sw_tm8_tn8_bm128_bn128_bk16_u8",
    "s3w_tm8_tn8_bm128_bn128_bk32_wm64_wn32_u8",
]

# SM throughput is collected only to identify the matmul kernel (not reported).
_SM_THROUGHPUT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"

# NCU launch/occupancy metrics
METRICS = {
    # launch config
    "threads_per_block":    "launch__block_size",
    "registers_per_thread": "launch__registers_per_thread",
    "smem_static_bytes":    "launch__shared_mem_per_block_static",
    "smem_dynamic_bytes":   "launch__shared_mem_per_block_dynamic",
    # occupancy
    "theoretical_occ_pct":  "sm__maximum_warps_per_active_cycle_pct",
    "achieved_occ_pct":     "sm__warps_active.avg.pct_of_peak_sustained_active",
    # limiters (these report the warps-per-SM ceiling imposed by each resource)
    "limit_registers":      "launch__occupancy_limit_registers",
    "limit_shared_mem":     "launch__occupancy_limit_shared_mem",
    "limit_block_size":     "launch__occupancy_limit_blocks",
    "limit_warps":          "launch__occupancy_limit_warps",
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
    for candidate in [ncu, "/usr/local/cuda-12.8/bin/ncu", "/usr/local/cuda-12.3/bin/ncu"]:
        if os.path.exists(candidate):
            ncu = candidate
            break
    all_metrics = list(METRICS.values()) + [_SM_THROUGHPUT]
    cmd = [
        "sudo", "-E", ncu,
        "--metrics", ",".join(all_metrics),
        "--csv",
        sys.executable, script_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"    ncu error: {result.stderr.strip()}", file=sys.stderr)
        return None
    csv_lines = [l for l in result.stdout.splitlines() if not l.startswith("==")]
    return "\n".join(csv_lines)


def parse_ncu_csv(csv_text: str) -> dict:
    """Return metrics for the kernel with the highest SM throughput (our matmul)."""
    from collections import defaultdict
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
                by_kernel[kid][name] = value  # keep strings (e.g. limiter names)
    if not by_kernel:
        return {}
    # Pick the kernel with the highest SM throughput — that's always the matmul.
    # Many torch helper kernels also use 256 threads, so block_size is not a
    # reliable selector.
    best = max(by_kernel.values(), key=lambda m: float(m.get(_SM_THROUGHPUT, 0) or 0))
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


def _bottleneck(m: dict) -> str:
    """Return which resource is the binding occupancy limiter."""
    limits = {
        "regs":  m.get("limit_registers"),
        "smem":  m.get("limit_shared_mem"),
        "block": m.get("limit_block_size"),
        "warps": m.get("limit_warps"),
    }
    valid = {k: float(v) for k, v in limits.items() if v is not None}
    if not valid:
        return "?"
    return min(valid, key=valid.__getitem__)


def print_table(rows: list[tuple[str, dict]]) -> None:
    hdr = (
        f"{'Kernel':<46} "
        f"{'threads':>7} "
        f"{'regs/thr':>8} "
        f"{'smem(KB)':>8} "
        f"{'th-occ%':>7} "
        f"{'ac-occ%':>7} "
        f"{'limiter':>7}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, m in rows:
        def v(k, fmt=".1f"):
            val = m.get(k)
            return f"{float(val):{fmt}}" if val is not None else "  n/a"

        smem_kb = (float(m["smem_static_bytes"] or 0) + float(m["smem_dynamic_bytes"] or 0)) / 1024 \
            if m.get("smem_static_bytes") is not None else float("nan")

        print(
            f"  {name:<44} "
            f"{v('threads_per_block', '.0f'):>7} "
            f"{v('registers_per_thread', '.0f'):>8} "
            f"{smem_kb:8.1f} "
            f"{v('theoretical_occ_pct'):>7} "
            f"{v('achieved_occ_pct'):>7} "
            f"{_bottleneck(m):>7}"
        )


def save_csv(rows: list[tuple[str, dict]], path: str) -> None:
    fieldnames = ["kernel"] + list(METRICS.keys()) + ["bottleneck"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, m in rows:
            w.writerow({
                "kernel": name,
                **{k: (m[k] if m[k] is not None else "") for k in METRICS},
                "bottleneck": _bottleneck(m),
            })
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
