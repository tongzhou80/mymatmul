#!/usr/bin/env python3
"""
NCU warp-stall profiling for CUDA matmul kernels.

Pass one or more CUDA kernel names directly.  The impl dotpath and dtype are
inferred from the kernel name via _all_impls().

Usage
-----
    sudo -E env PATH="$PATH" PYTHONPATH="$PYTHONPATH" python3 profile_ncu_stalls.py \\
        matmul_cuda_tc1_bm256_bn64_bk32_nw4 --size 2048

    sudo ... python3 profile_ncu_stalls.py \\
        matmul_cuda_tc1_bm256_bn64_bk32_nw4 matmul_cuda_s6_bm256_bn64_bk32_nw4 --size 4096
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from io import StringIO

METRICS = {
    "barrier":    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "membar":     "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "long_sb":    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "short_sb":   "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "math_throt": "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "mio_throt":  "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "not_sel":    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "no_inst":    "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
    "wait":       "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "misc":       "smsp__warp_issue_stalled_misc_per_warp_active.pct",
}


def infer_impl(cuda_kernel_name: str, all_i: dict) -> tuple[str, str, str] | None:
    """Return (impl_name, dotpath, dtype_str) by matching cuda_kernel_name against _all_impls()."""
    import torch
    # strip mandatory prefix
    prefix = "matmul_cuda_"
    stem = cuda_kernel_name[len(prefix):] if cuda_kernel_name.startswith(prefix) else cuda_kernel_name
    # progressively strip trailing _<token> until we find a match
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
    script = f"""\
import sys
sys.path.insert(0, {repr(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))})
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


def run_ncu(script_path: str, cuda_name: str) -> str | None:
    ncu = find_ncu()
    metric_str = ",".join(METRICS.values())
    cmd = ["sudo", "-E", ncu, "--kernel-name", cuda_name,
           "--metrics", metric_str, "--csv", sys.executable, script_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"    ncu error:\n{result.stderr.strip()}", file=sys.stderr)
        return None
    # keep only CSV lines (start with '"'); sudo merges all output into stdout
    csv_lines = [l for l in result.stdout.splitlines() if l.startswith('"')]
    return "\n".join(csv_lines)


def parse_ncu_csv(csv_text: str) -> dict:
    from collections import defaultdict
    by_id: dict[str, dict] = defaultdict(dict)
    reader = csv.DictReader(StringIO(csv_text),
                            fieldnames=["ID","Process ID","Process Name","Host Name",
                                        "Kernel Name","Context","Stream","Block Size",
                                        "Grid Size","Device","CC","Section Name",
                                        "Metric Name","Metric Unit","Metric Value"])
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
    all_vals: dict[str, list] = defaultdict(list)
    for metrics in by_id.values():
        for k, v in metrics.items():
            all_vals[k].append(v)
    return {k: sum(vs) / len(vs) for k, vs in all_vals.items()}


def profile_one(dotpath: str, cuda_name: str, size: int, dtype: str) -> dict | None:
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        tmp = f.name
    try:
        make_target_script(dotpath, size, tmp, dtype=dtype)
        csv_text = run_ncu(tmp, cuda_name)
        if csv_text is None:
            return None
        raw = parse_ncu_csv(csv_text)
        if not raw:
            print("    warning: no metrics parsed", file=sys.stderr)
            return None
        return {k: raw.get(v) for k, v in METRICS.items()}
    finally:
        os.unlink(tmp)


def print_table(rows: list[tuple[str, dict]]) -> None:
    hdr = (f"{'Kernel':<44} {'barrier':>8} {'membar':>7} {'long_sb':>8} "
           f"{'short_sb':>9} {'math_th':>8} {'mio_th':>7} "
           f"{'not_sel':>8} {'no_inst':>8} {'wait':>6} {'misc':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, m in rows:
        def v(k):
            x = m.get(k)
            return x if x is not None else float("nan")
        print(
            f"  {name:<42} "
            f"{v('barrier'):8.1f} "
            f"{v('membar'):7.1f} "
            f"{v('long_sb'):8.1f} "
            f"{v('short_sb'):9.1f} "
            f"{v('math_throt'):8.1f} "
            f"{v('mio_throt'):7.1f} "
            f"{v('not_sel'):8.1f} "
            f"{v('no_inst'):8.1f} "
            f"{v('wait'):6.1f} "
            f"{v('misc'):6.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cuda_kernels", nargs="+",
                        help="CUDA kernel name(s) to profile (e.g. matmul_cuda_tc1_bm256_bn64_bk32_nw4)")
    parser.add_argument("--size", type=int, default=4096, metavar="N")
    args = parser.parse_args()

    bench_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, bench_dir)
    from bench_gpu import _all_impls
    all_i = _all_impls()

    rows = []
    for cuda_name in args.cuda_kernels:
        inferred = infer_impl(cuda_name, all_i)
        if inferred is None:
            print(f"  [{cuda_name}] could not infer impl from _all_impls() — skipping")
            continue
        impl_name, dotpath, dtype_str = inferred
        print(f"  [{cuda_name}] impl={impl_name}  dtype={dtype_str}  size={args.size}³ ...",
              end=" ", flush=True)
        result = profile_one(dotpath, cuda_name, args.size, dtype_str)
        if result is None:
            print("FAILED")
        else:
            print("done")
            rows.append((cuda_name, result))

    if rows:
        print_table(rows)
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
