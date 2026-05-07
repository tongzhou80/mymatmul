#!/usr/bin/env python3
"""
NCU occupancy analysis: registers, shared memory, and limiting factors.

Pass one or more impl names (from _all_impls()) or CUDA kernel names.

For each kernel, collects:
  - threads per block (launch config)
  - registers per thread
  - static + dynamic shared memory per block (bytes)
  - theoretical occupancy (warps/SM at full utilisation)
  - achieved occupancy (%)
  - the resource that limits occupancy: registers / shared_mem / block_size / warps

Usage:
    sudo python profile_ncu_occupancy.py tc1 tc4 cublas_bf16 --size 4096
    sudo python profile_ncu_occupancy.py matmul_cuda_tc1_bm256_bn64_bk32_nw4 --size 4096
    sudo python profile_ncu_occupancy.py --size 4096 --out results_occupancy.csv
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from io import StringIO

# SM throughput identifies the main matmul kernel among all kernels launched.
_SM_THROUGHPUT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"

METRICS = {
    "threads_per_block":    "launch__block_size",
    "registers_per_thread": "launch__registers_per_thread",
    "smem_static_bytes":    "launch__shared_mem_per_block_static",
    "smem_dynamic_bytes":   "launch__shared_mem_per_block_dynamic",
    "theoretical_occ_pct":  "sm__maximum_warps_per_active_cycle_pct",
    "achieved_occ_pct":     "sm__warps_active.avg.pct_of_peak_sustained_active",
    "limit_registers":      "launch__occupancy_limit_registers",
    "limit_shared_mem":     "launch__occupancy_limit_shared_mem",
    "limit_block_size":     "launch__occupancy_limit_blocks",
    "limit_warps":          "launch__occupancy_limit_warps",
}


def infer_impl(name: str, all_i: dict) -> tuple[str, str, str] | None:
    """Accept either an impl name or a CUDA kernel name; return (impl_name, dotpath, dtype_str)."""
    import torch
    if name in all_i:
        dotpath, _, dtype = all_i[name]
        dtype_str = "torch.bfloat16" if dtype == torch.bfloat16 else "torch.float32"
        return name, dotpath, dtype_str
    # Try stripping matmul_cuda_ prefix and matching progressively shorter suffixes
    prefix = "matmul_cuda_"
    stem = name[len(prefix):] if name.startswith(prefix) else name
    parts = stem.split("_")
    for n in range(len(parts), 0, -1):
        candidate = "_".join(parts[:n])
        if candidate in all_i:
            dotpath, _, dtype = all_i[candidate]
            dtype_str = "torch.bfloat16" if dtype == torch.bfloat16 else "torch.float32"
            return candidate, dotpath, dtype_str
    return None


def make_target_script(dotpath: str, size: int, path: str, dtype: str = "torch.float32") -> None:
    module_path, fn_name = dotpath.rsplit(".", 1)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = f"""\
import sys
sys.path.insert(0, {repr(repo_root)})
import torch, importlib
fn = getattr(importlib.import_module({repr(module_path)}), {repr(fn_name)})
A = torch.randn({size}, {size}, dtype={dtype}, device='cuda')
B = torch.randn({size}, {size}, dtype={dtype}, device='cuda')
for _ in range(3):
    fn(A, B)
torch.cuda.synchronize()
fn(A, B)
torch.cuda.synchronize()
"""
    with open(path, "w") as f:
        f.write(script)


def find_ncu() -> str:
    ncu = os.environ.get("NCU", "/usr/local/cuda/bin/ncu")
    if os.path.exists(ncu):
        return ncu
    for candidate in ["/usr/local/cuda-12.8/bin/ncu", "/usr/local/cuda-12.3/bin/ncu"]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("ncu not found; set NCU env var")


def run_ncu(script_path: str) -> str | None:
    ncu = find_ncu()
    all_metrics = list(METRICS.values()) + [_SM_THROUGHPUT]
    cmd = ["sudo", "-E", ncu,
           "--metrics", ",".join(all_metrics),
           "--csv", sys.executable, script_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"    ncu error: {result.stderr.strip()}", file=sys.stderr)
        return None
    csv_lines = [l for l in result.stdout.splitlines() if not l.startswith("==")]
    return "\n".join(csv_lines)


def parse_ncu_csv(csv_text: str) -> dict:
    """Return metrics for the kernel with the highest SM throughput."""
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
                by_kernel[kid][name] = value
    if not by_kernel:
        return {}
    best = max(by_kernel.values(), key=lambda m: float(m.get(_SM_THROUGHPUT, 0) or 0))
    return best


def profile_one(dotpath: str, size: int, dtype: str) -> dict | None:
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        tmp = f.name
    try:
        make_target_script(dotpath, size, tmp, dtype=dtype)
        csv_text = run_ncu(tmp)
        if csv_text is None:
            return None
        raw = parse_ncu_csv(csv_text)
        if not raw:
            print("    warning: no metrics parsed", file=sys.stderr)
            return None
        return {k: raw.get(v) for k, v in METRICS.items()}
    finally:
        os.unlink(tmp)


def _bottleneck(m: dict) -> str:
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

        static  = float(m.get("smem_static_bytes")  or 0)
        dynamic = float(m.get("smem_dynamic_bytes") or 0)
        smem_kb = (static + dynamic) / 1024

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
                **{k: (m[k] if m.get(k) is not None else "") for k in METRICS},
                "bottleneck": _bottleneck(m),
            })
    print(f"\nSaved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("impls", nargs="*",
                        help="Impl names or CUDA kernel names (e.g. tc1, matmul_cuda_tc1_bm256_bn64_bk32_nw4)")
    parser.add_argument("--size", type=int, default=4096, metavar="N")
    parser.add_argument("--out", default=None, metavar="FILE")
    args = parser.parse_args()

    bench_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, bench_dir)
    from bench_gpu import _all_impls
    all_i = _all_impls()

    if not args.impls:
        print("No impls specified. Available tensor-core impls:")
        for k in sorted(all_i):
            if k.startswith("tc"):
                print(f"  {k}")
        return

    rows = []
    for name in args.impls:
        inferred = infer_impl(name, all_i)
        if inferred is None:
            print(f"  [{name}] not found in _all_impls() — skipping")
            continue
        impl_name, dotpath, dtype_str = inferred
        print(f"  [{name}] impl={impl_name}  dtype={dtype_str}  size={args.size}³ ...",
              end=" ", flush=True)
        result = profile_one(dotpath, args.size, dtype_str)
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
