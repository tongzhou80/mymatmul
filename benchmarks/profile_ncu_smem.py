#!/usr/bin/env python3
"""
NCU shared memory traffic profiling.

Measures actual smem load/store wavefronts, local memory (register spill)
traffic, and shuffle instruction counts — useful for diagnosing whether a
kernel reduction in smem reads actually lands or is offset by spills/shuffles.

Usage
-----
    # Default targets at 4096³
    sudo python profile_ncu_smem.py

    # Custom kernels
    sudo python profile_ncu_smem.py s4st_tm8_tn8_bm128_bn128_bk16_u16 s4st_shfl_tm8_tn8_bm128_bn128_bk16

    # Different size
    sudo python profile_ncu_smem.py --size 8192
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from io import StringIO

DEFAULT_KERNELS = [
    "s4st_tm8_tn8_bm128_bn128_bk16_u16",
    "s4st_shfl_tm8_tn8_bm128_bn128_bk16",
]

METRICS = {
    # shared memory wavefronts (1 wavefront = 1 bank-conflict-free smem access)
    "smem_ld_wf":   "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "smem_st_wf":   "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    # shared memory instructions issued
    "smem_ld_inst": "smsp__inst_executed_op_shared_ld.sum",
    "smem_st_inst": "smsp__inst_executed_op_shared_st.sum",
    # local memory (register spill) wavefronts
    "lmem_ld_wf":   "l1tex__data_pipe_lsu_wavefronts_mem_local_op_ld.sum",
    "lmem_st_wf":   "l1tex__data_pipe_lsu_wavefronts_mem_local_op_st.sum",
    # shuffle instructions
    "shfl_inst":    "smsp__inst_executed_op_shuffle.sum",
}

_S4ST_SHFL_CUDA_NAME = "matmul_cuda_s4st_shfl_tm8_tn8_bm128_bn128_bk16"


def cuda_kernel_name(impl_name: str) -> str | None:
    if impl_name.startswith("s3w_"):
        return "matmul_cuda_s3_warp_" + impl_name[4:]
    if impl_name.startswith(("s3_", "s4_", "s4b_", "s4sw_", "s4st_", "s4st2_", "s4stp_")):
        return "matmul_cuda_" + impl_name
    if impl_name.startswith("triton_"):
        return "_matmul_kernel"
    return None


def make_script(dotpath: str, size: int, path: str) -> None:
    mod, fn = dotpath.rsplit(".", 1)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(path, "w") as f:
        f.write(f"""\
import sys; sys.path.insert(0, {repr(repo)})
import torch, importlib
fn = getattr(importlib.import_module({repr(mod)}), {repr(fn)})
A = torch.randn({size}, {size}, dtype=torch.float32, device='cuda')
B = torch.randn({size}, {size}, dtype=torch.float32, device='cuda')
for _ in range(3): fn(A, B)
torch.cuda.synchronize()
fn(A, B)
torch.cuda.synchronize()
""")


def run_ncu(script: str, cuda_name: str | None) -> str | None:
    ncu = os.environ.get("NCU", "/usr/local/cuda/bin/ncu")
    metric_str = ",".join(METRICS.values())
    cmd = ["sudo", "-E", ncu, "--metrics", metric_str, "--csv"]
    if cuda_name:
        cmd += ["--kernel-name", cuda_name]
    cmd += [sys.executable, script]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"    ncu error: {r.stderr.strip()}", file=sys.stderr)
        return None
    lines = [l for l in r.stdout.splitlines() if not l.startswith("==")]
    return "\n".join(lines)


def parse(csv_text: str, cuda_name: str | None) -> dict:
    by_id: dict[str, dict] = defaultdict(dict)
    for row in csv.DictReader(StringIO(csv_text)):
        kid  = row.get("ID", "").strip('"')
        name = row.get("Metric Name", "").strip('"')
        val  = row.get("Metric Value", "").strip('"').replace(",", "")
        if kid and name and val:
            try:
                by_id[kid][name] = float(val)
            except ValueError:
                pass
    if not by_id:
        return {}
    if cuda_name:
        agg: dict[str, list] = defaultdict(list)
        for m in by_id.values():
            for k, v in m.items():
                agg[k].append(v)
        return {k: sum(vs) / len(vs) for k, vs in agg.items()}
    sm_key = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    return max(by_id.values(), key=lambda m: m.get(sm_key, 0.0))


def profile_one(name: str, dotpath: str, size: int) -> dict | None:
    cuda_name = cuda_kernel_name(name)
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        tmp = f.name
    try:
        make_script(dotpath, size, tmp)
        csv_text = run_ncu(tmp, cuda_name)
        if not csv_text:
            return None
        raw = parse(csv_text, cuda_name)
        if not raw:
            print("    warning: no metrics parsed", file=sys.stderr)
            return None
        return {k: raw.get(v) for k, v in METRICS.items()}
    finally:
        os.unlink(tmp)


def print_table(rows: list[tuple[str, dict]]) -> None:
    col = 16
    keys = list(METRICS.keys())
    hdr = f"{'Kernel':<48}" + "".join(f"{k:>{col}}" for k in keys)
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, m in rows:
        row = f"  {name:<46}"
        for k in keys:
            v = m.get(k)
            row += f"{(f'{v:.0f}' if v is not None else 'n/a'):>{col}}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("impls", nargs="*", default=DEFAULT_KERNELS)
    parser.add_argument("--size", type=int, default=4096)
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench_gpu import IMPLEMENTATIONS

    rows = []
    for name in args.impls:
        if name not in IMPLEMENTATIONS:
            print(f"  [{name}] not in IMPLEMENTATIONS — skipping")
            continue
        dotpath, _ = IMPLEMENTATIONS[name]
        print(f"  [{name}] size={args.size}³ ...", end=" ", flush=True)
        result = profile_one(name, dotpath, args.size)
        if result is None:
            print("FAILED")
        else:
            print("done")
            rows.append((name, result))

    if rows:
        print_table(rows)
    else:
        print("No results.")


if __name__ == "__main__":
    main()
