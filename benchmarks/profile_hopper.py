#!/usr/bin/env python3
"""
Profile a specific CUDA kernel by name using NCU.

Parses the kernel name to determine which module and launch parameters to use.
No autotuning — kernels are launched directly by the name you provide.

Usage (requires sudo for ncu):
    sudo -E env PATH="$PATH" PYTHONPATH="$PYTHONPATH" python3 benchmarks/profile_hopper.py \\
        matmul_h2s3_wg2_bn256_bk64  \\
        matmul_cuda_tc5_bm128_bn128_bk32_nw4 \\
        --size 4096 [--lb 1]

Supported kernel name formats
------------------------------
  matmul_h2s3_wg{N}_bn{N}_bk{N}          H2-S3 (cuda-python, multi-warpgroup)
  matmul_h2s2_wg{N}_bn{N}_bk{N}          H2-S2 (cuda-python, single-warpgroup)
  matmul_h2s1_bm{N}_bn{N}_bk{N}_nw{N}    H2-S1 (cuda-python, mma.sync)
  matmul_cuda_tc5_bm{N}_bn{N}_bk{N}_nw{N} tc5 / tc5_regpruned  (pycuda, use --lb)
  cublas                                   cuBLAS bf16 reference
"""

import argparse, csv, os, re, subprocess, sys, tempfile
from io import StringIO

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── NCU metrics ───────────────────────────────────────────────────────────────

METRICS = {
    "sm_sol_pct":           "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_sol_pct":         "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_sol_pct":           "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_sol_pct":           "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "achieved_occ_pct":     "sm__warps_active.avg.pct_of_peak_sustained_active",
    "theoretical_occ_pct":  "sm__maximum_warps_per_active_cycle_pct",
    "threads_per_block":    "launch__block_size",
    "registers_per_thread": "launch__registers_per_thread",
    "smem_static_bytes":    "launch__shared_mem_per_block_static",
    "smem_dynamic_bytes":   "launch__shared_mem_per_block_dynamic",
    "limit_registers":      "launch__occupancy_limit_registers",
    "limit_shared_mem":     "launch__occupancy_limit_shared_mem",
    "limit_blocks":         "launch__occupancy_limit_blocks",
    "limit_warps":          "launch__occupancy_limit_warps",
    "smem_ld_conflicts":    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "smem_st_conflicts":    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "stall_math_throt":     "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "stall_long_sb":        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "stall_not_sel":        "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "stall_membar":         "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "stall_barrier":        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "stall_short_sb":       "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "stall_mio_throt":      "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "stall_wait":           "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "stall_no_inst":        "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
}

# ── Target script builders ────────────────────────────────────────────────────

def _header(size):
    return (f"import sys; sys.path.insert(0, {repr(REPO)})\n"
            f"import torch\n"
            f"S = {size}\n"
            f"A = torch.randn(S, S, device='cuda', dtype=torch.bfloat16)\n"
            f"B = torch.randn(S, S, device='cuda', dtype=torch.bfloat16)\n")


def build_script(kname: str, size: int, lb: int) -> str:
    """Return a Python script body that launches the named kernel once."""

    # ── h2s3: matmul_h2s3_wg{N}_bn{N}_bk{N} ─────────────────────────────────
    m = re.fullmatch(r"matmul_h2s3_wg(\d+)_bn(\d+)_bk(\d+)", kname)
    if m:
        nwg, bn, bk = int(m[1]), int(m[2]), int(m[3])
        return _header(size) + f"""\
from mymatmul.gpu.hopper.matmul_h2_s3 import _launch, _get_mod, _kname
_launch(_get_mod(), {repr(kname)}, A, B, {nwg}, {bn}, {bk})
torch.cuda.synchronize()
"""

    # ── h2s2: matmul_h2s2_wg{N}_bn{N}_bk{N} ─────────────────────────────────
    m = re.fullmatch(r"matmul_h2s2_wg(\d+)_bn(\d+)_bk(\d+)", kname)
    if m:
        nwg, bn, bk = int(m[1]), int(m[2]), int(m[3])
        return _header(size) + f"""\
from mymatmul.gpu.hopper.matmul_h2_s2 import _launch, _get_mod
_launch(_get_mod(), {repr(kname)}, A, B, {nwg}, {bn}, {bk})
torch.cuda.synchronize()
"""

    # ── h2s1: matmul_h2s1_bm{N}_bn{N}_bk{N}_nw{N} ───────────────────────────
    m = re.fullmatch(r"matmul_h2s1_bm(\d+)_bn(\d+)_bk(\d+)_nw(\d+)", kname)
    if m:
        bm, bn, bk, nw = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        return _header(size) + f"""\
from mymatmul.gpu.hopper.matmul_h2_s1 import _launch, _get_mod
_launch(_get_mod({lb}), {repr(kname)}, A, B, {nw}, {bm}, {bn}, {bk}, {lb})
torch.cuda.synchronize()
"""

    # ── tc5: matmul_cuda_tc5_bm{N}_bn{N}_bk{N}_nw{N} ────────────────────────
    m = re.fullmatch(r"matmul_cuda_tc5_bm(\d+)_bn(\d+)_bk(\d+)_nw(\d+)", kname)
    if m:
        bm, bn, bk, nw = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        return _header(size) + f"""\
from mymatmul.gpu.tensor_core.matmul_cuda_tc5_regpruned import _launch, _get_mod, _block, _grid, _smem
_launch(_get_mod({lb}), {repr(kname)}, A, B,
        _block({nw}), _grid(S, S, {bm}, {bn}), _smem({bm}, {bn}, {bk}))
torch.cuda.synchronize()
"""

    # ── cublas ────────────────────────────────────────────────────────────────
    if kname == "cublas":
        return _header(size) + "torch.mm(A, B)\ntorch.cuda.synchronize()\n"

    raise ValueError(
        f"Unrecognised kernel name: {repr(kname)}\n"
        "Supported prefixes: matmul_h2s3_wg*, matmul_h2s2_wg*, matmul_h2s1_bm*, "
        "matmul_cuda_tc5_bm*, cublas"
    )

# ── NCU runner ────────────────────────────────────────────────────────────────

def _find_ncu():
    for p in ["/usr/local/cuda/bin/ncu", "/usr/bin/ncu"]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("ncu not found")


def run_ncu(script: str) -> str | None:
    ncu = _find_ncu()
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script); fname = f.name
    cmd = ["sudo", "-E", ncu,
           "--metrics", ",".join(METRICS.values()),
           "--csv", sys.executable, fname]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    os.unlink(fname)
    if r.returncode != 0:
        print(f"\n  ncu stderr: {r.stderr.strip()[:400]}", file=sys.stderr)
        return None
    return "\n".join(l for l in r.stdout.splitlines() if not l.startswith("=="))


def parse_ncu(csv_text: str) -> dict:
    result = {}
    for row in csv.DictReader(StringIO(csv_text)):
        metric = row.get("Metric Name", "").strip('"')
        value  = row.get("Metric Value", "").strip('"').replace(",", "")
        if not metric or not value:
            continue
        try:    result[metric] = float(value)
        except: result[metric] = value
    return result

# ── Pretty printer ────────────────────────────────────────────────────────────

def _v(data, key, fmt="{:.1f}"):
    v = data.get(METRICS.get(key, key))
    if v is None: return "N/A"
    return fmt.format(v) if isinstance(v, float) else str(v)


def print_results(rows, max_warps_sm):
    labels = [r[0] for r in rows]
    w = max(len(l) for l in labels) + 2

    hdr = f"  {'Metric':<30}" + "".join(f"{l:>{w}}" for l in labels)
    print(f"\n{hdr}")
    print("  " + "─" * len(hdr))

    def row(name, key, fmt="{:.1f}", unit=""):
        vals = "".join(f"{_v(r[1], key, fmt):>{w}}" for r in rows)
        print(f"  {name:<30}{vals}  {unit}")

    print("  Speed-of-Light (%)")
    row("SM SoL",              "sm_sol_pct",          unit="%")
    row("DRAM SoL",            "dram_sol_pct",         unit="%")
    row("L1 SoL",              "l1_sol_pct",           unit="%")
    row("L2 SoL",              "l2_sol_pct",           unit="%")
    print("  Occupancy")
    row("Achieved",            "achieved_occ_pct",     unit="% warps active")
    row("Theoretical",         "theoretical_occ_pct",  unit="%")
    print("  Launch config")
    row("Threads/block",       "threads_per_block",    "{:.0f}")
    row("Registers/thread",    "registers_per_thread", "{:.0f}")
    row("Dynamic SMEM",        "smem_dynamic_bytes",   "{:.0f}",  "bytes")
    print(f"  Occupancy limiters (max {max_warps_sm} warps/SM)")
    row("  Registers",         "limit_registers",      "{:.0f}",  "warps/SM")
    row("  Shared mem",        "limit_shared_mem",     "{:.0f}",  "warps/SM")
    row("  Blocks",            "limit_blocks",         "{:.0f}",  "warps/SM")
    print("  Bank conflicts")
    row("SMEM load conflicts",  "smem_ld_conflicts",   "{:.0f}")
    row("SMEM store conflicts", "smem_st_conflicts",   "{:.0f}")
    print("  Warp stalls (% of active cycles)")
    row("  Math throttle",     "stall_math_throt")
    row("  Long scoreboard",   "stall_long_sb")
    row("  Not selected",      "stall_not_sel")
    row("  Membar",            "stall_membar")
    row("  Barrier",           "stall_barrier")
    row("  Short scoreboard",  "stall_short_sb")
    row("  MIO throttle",      "stall_mio_throt")
    row("  Wait",              "stall_wait")
    row("  No instruction",    "stall_no_inst")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import torch
    p = argparse.ArgumentParser(
        description="Profile specific CUDA kernels by name using NCU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("kernels", nargs="+",
                   help="Kernel names, e.g. matmul_h2s3_wg2_bn256_bk64")
    p.add_argument("--size", type=int, default=4096,
                   help="Square matrix size (default 4096)")
    p.add_argument("--lb", type=int, default=1,
                   help="LB_MIN_BLOCKS for tc5/h2s1 kernels (default 1)")
    args = p.parse_args()

    props = torch.cuda.get_device_properties(0)
    max_warps_sm = props.max_threads_per_multi_processor // 32
    print(f"GPU: {props.name}  (sm_{props.major}{props.minor})"
          f"  max {max_warps_sm} warps/SM  size={args.size}³")

    rows = []
    for kname in args.kernels:
        print(f"  profiling {kname} ...", end=" ", flush=True)
        try:
            script = build_script(kname, args.size, args.lb)
        except ValueError as e:
            print(f"SKIPPED: {e}"); continue

        csv_text = run_ncu(script)
        if csv_text is None:
            print("FAILED"); continue
        data = parse_ncu(csv_text)
        if not data:
            print("FAILED (no metrics parsed)"); continue
        print("done")
        rows.append((kname, data))

    if rows:
        print_results(rows, max_warps_sm)
    else:
        print("No results.")


if __name__ == "__main__":
    main()
