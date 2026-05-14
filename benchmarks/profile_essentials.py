#!/usr/bin/env python3
"""
NCU essential metrics in one pass: SoL, occupancy, resources, bank conflicts.

Usage
-----
    sudo -E env PATH="$PATH" PYTHONPATH="$PYTHONPATH" python3 profile_essentials.py \\
        matmul_cuda_tc4_bm64_bn256_bk32_nw4 --size 4096

    sudo ... python3 profile_essentials.py \\
        matmul_cuda_tc4_bm64_bn256_bk32_nw4 matmul_cuda_tc2b_bm128_bn64_bk64_nw4 --size 8192 --out results.csv
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from io import StringIO

METRICS = {
    # Speed-of-light
    "sm_sol_pct":           "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_sol_pct":         "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_sol_pct":           "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_sol_pct":           "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    # Occupancy
    "achieved_occ_pct":     "sm__warps_active.avg.pct_of_peak_sustained_active",
    "theoretical_occ_pct":  "sm__maximum_warps_per_active_cycle_pct",
    # Launch config
    "threads_per_block":    "launch__block_size",
    "registers_per_thread": "launch__registers_per_thread",
    "smem_static_bytes":    "launch__shared_mem_per_block_static",
    "smem_dynamic_bytes":   "launch__shared_mem_per_block_dynamic",
    # Occupancy limiters (each reports max warps/SM that factor allows)
    "limit_registers":      "launch__occupancy_limit_registers",
    "limit_shared_mem":     "launch__occupancy_limit_shared_mem",
    "limit_blocks":         "launch__occupancy_limit_blocks",
    "limit_warps":          "launch__occupancy_limit_warps",
    # Bank conflicts
    "smem_ld_conflicts":    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "smem_st_conflicts":    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    # Pipe utilization — what % of active SM cycles each pipe is busy
    "tensor_pipe_pct":      "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "tensor_hmma_pct":      "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
    "alu_pipe_pct":         "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "fma_pipe_pct":         "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "lsu_pipe_pct":         "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    # Warp stalls (% of warp-active cycles)
    "stall_math_throt":     "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "stall_long_sb":        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "stall_not_sel":        "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "stall_membar":         "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "stall_barrier":        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "stall_short_sb":       "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "stall_mio_throt":      "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "stall_wait":           "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "stall_no_inst":        "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
    "stall_misc":           "smsp__warp_issue_stalled_misc_per_warp_active.pct",
}


# Mapping from CUDA kernel name prefix → bench_gpu impl name, for Hopper kernels
# that don't use the matmul_cuda_<prefix>_* convention.
_HOPPER_PREFIX_MAP = {
    "matmul_h1ms": "h1_ms",
    "matmul_h2s1": "h2_s1",
    "matmul_h2s2": "h2_s2",
    "matmul_h2s3": "h2_s3",
    "matmul_h2s4": "h2_s4",
    "matmul_h2s5": "h2_s5",
    "matmul_h2s6": "h2_s6",
    "matmul_h2s7": "h2_s7",
    "matmul_h2s8": "h2_s8",
    "matmul_h4s2": "h4_s2",
    "matmul_h4":   "h4",
    "matmul_h5":   "h5",
    "matmul_h6":   "h6",
    "matmul_h7":   "h7",
    "matmul_h8":   "h8",
    "matmul_h3":   "h3",
    # Exact-match entries (no config suffix in the CUDA kernel name)
    "matmul_kernel": "triton_ptx",
}


def _hopper_impl_name(kname: str) -> str | None:
    """Return the bench_gpu impl name for a Hopper kernel, or None."""
    for prefix, impl in _HOPPER_PREFIX_MAP.items():
        if kname == prefix or kname.startswith(prefix + "_"):
            return impl
    return None


def find_impl(kname: str, all_i: dict):
    """Return (impl_name, module_path, dtype_str) for a kernel name or CUDA kernel name."""
    import torch

    # Hopper kernels: matmul_h1ms_*, matmul_h2s3_*, etc.
    hopper_impl = _hopper_impl_name(kname)
    if hopper_impl:
        entry = all_i.get(hopper_impl)
        if entry is None:
            return None
        dotpath, _, dtype = entry
        module_path = dotpath.rsplit(".", 1)[0]
        dtype_str = "torch.bfloat16" if dtype == torch.bfloat16 else "torch.float32"
        return hopper_impl, module_path, dtype_str

    # Legacy kernels: matmul_cuda_<prefix>_* or bare <prefix>
    stem = kname[len("matmul_cuda_"):] if kname.startswith("matmul_cuda_") else kname
    prefix = stem.split("_")[0]
    entry = all_i.get(prefix)
    if entry is None:
        return None
    dotpath, _, dtype = entry
    module_path = dotpath.rsplit(".", 1)[0]
    dtype_str = "torch.bfloat16" if dtype == torch.bfloat16 else "torch.float32"
    return prefix, module_path, dtype_str


def make_target_script(module_path: str, kname: str, size: int, path: str, dtype: str) -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Hopper-style modules expose _launch_by_name(kname, M, N, K) — use that.
    # This is detected by checking if the kernel name has a Hopper prefix.
    if _hopper_impl_name(kname):
        script = f"""\
import sys
sys.path.insert(0, {repr(repo_root)})
import torch, importlib
_mod = importlib.import_module({repr(module_path)})
_mod._launch_by_name({repr(kname)}, {size}, {size}, {size})
torch.cuda.synchronize()
"""
    else:
        # Legacy pycuda path (tc5, tc6, tc7, s6_lb, …)
        script = f"""\
import sys, inspect
sys.path.insert(0, {repr(repo_root)})
import torch, importlib
_mod = importlib.import_module({repr(module_path)})
from mymatmul.gpu._pycuda_loader import launch_matmul, get_module
get_module(_mod._EXT)

kname = {repr(kname)}
cfg = next((c for c in _mod._CONFIGS if _mod._kname(*c) == kname), None)
if cfg is None:
    raise ValueError(f"Kernel {{kname!r}} not found in _CONFIGS")

bm, bn, bk, nw = cfg[0], cfg[1], cfg[2], cfg[3]
block = _mod._block(nw)
grid  = _mod._grid({size}, {size}, bm, bn)

_smem_n = len(inspect.signature(_mod._smem).parameters)
_smem_args = cfg[:3] + cfg[4:]
smem = _mod._smem(*_smem_args[:_smem_n])

A = torch.randn({size}, {size}, dtype={dtype}, device='cuda')
B = torch.randn({size}, {size}, dtype={dtype}, device='cuda')
launch_matmul(_mod._EXT, kname, A, B, block, grid, out_dtype={dtype}, smem_bytes=smem)
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
    cmd = ["sudo", "-E", ncu,
           "--metrics", ",".join(METRICS.values()),
           "--csv", sys.executable, script_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"    ncu error: {result.stderr.strip()}", file=sys.stderr)
        return None
    csv_lines = [l for l in result.stdout.splitlines() if not l.startswith("==")]
    return "\n".join(csv_lines)


def parse_ncu_csv(csv_text: str, kname: str) -> dict:
    """Return metrics from the single profiled invocation of kname."""
    result: dict = {}
    reader = csv.DictReader(StringIO(csv_text))
    for row in reader:
        kn     = row.get("Kernel Name", "").strip('"')
        metric = row.get("Metric Name", "").strip('"')
        value  = row.get("Metric Value", "").strip('"').replace(",", "")
        if kn != kname or not metric or not value:
            continue
        try:
            result[metric] = float(value)
        except ValueError:
            result[metric] = value
    return result


def profile_one(module_path: str, kname: str, size: int, dtype: str) -> dict | None:
    # Hopper kernels already carry their full name; legacy kernels need "matmul_cuda_" prepended.
    if _hopper_impl_name(kname):
        full_kname = kname
    else:
        full_kname = kname if kname.startswith("matmul_cuda_") else f"matmul_cuda_{kname}"
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        tmp = f.name
    try:
        make_target_script(module_path, full_kname, size, tmp, dtype)
        csv_text = run_ncu(tmp)
        if csv_text is None:
            return None
        raw = parse_ncu_csv(csv_text, full_kname)
        if not raw:
            print("    warning: no metrics parsed", file=sys.stderr)
            return None
        return {k: raw.get(v) for k, v in METRICS.items()}
    finally:
        os.unlink(tmp)


def _bottleneck(m: dict) -> str:
    limits = {
        "regs":   m.get("limit_registers"),
        "smem":   m.get("limit_shared_mem"),
        "blocks": m.get("limit_blocks"),
        "warps":  m.get("limit_warps"),
    }
    valid = {k: float(v) for k, v in limits.items() if v is not None}
    if not valid:
        return "?"
    return min(valid, key=valid.__getitem__)


def _fv(m: dict, key: str, fmt: str = ".1f") -> str:
    val = m.get(key)
    if val is None:
        return "n/a"
    try:
        return f"{float(val):{fmt}}"
    except (ValueError, TypeError):
        return str(val)


def print_results(rows: list[tuple[str, dict]], max_warps_per_sm: int) -> None:
    for name, m in rows:
        smem_kb = (float(m.get("smem_static_bytes") or 0)
                   + float(m.get("smem_dynamic_bytes") or 0)) / 1024

        achieved_occ = m.get("achieved_occ_pct")
        warps_per_sm = (float(achieved_occ) / 100 * max_warps_per_sm
                        if achieved_occ is not None else None)
        warps_str = (f"{warps_per_sm:.1f}/{max_warps_per_sm}"
                     if warps_per_sm is not None else "n/a")

        limiter = _bottleneck(m)
        _limiter_key = {"regs": "limit_registers", "smem": "limit_shared_mem",
                        "blocks": "limit_blocks", "warps": "limit_warps"}
        limit_val = m.get(_limiter_key.get(limiter, ""))
        limit_str = f"{limiter}({int(float(limit_val))}w)" if limit_val is not None else limiter

        print(f"\n{name}")
        print(f"  SoL:  SM={_fv(m,'sm_sol_pct')}%  DRAM={_fv(m,'dram_sol_pct')}%"
              f"  L1={_fv(m,'l1_sol_pct')}%  L2={_fv(m,'l2_sol_pct')}%")
        print(f"  Occ:  achieved={_fv(m,'achieved_occ_pct')}%"
              f"  theoretical={_fv(m,'theoretical_occ_pct')}%"
              f"  warps/SM={warps_str}")
        print(f"  Rsrc: regs={_fv(m,'registers_per_thread','.0f')}/thr"
              f"  smem={smem_kb:.1f}KB"
              f"  limiter={limit_str}"
              f"  LD-cf={_fv(m,'smem_ld_conflicts','.0f')}"
              f"  ST-cf={_fv(m,'smem_st_conflicts','.0f')}")
        print(f"  Stall: math={_fv(m,'stall_math_throt')}%"
              f"  long_sb={_fv(m,'stall_long_sb')}%"
              f"  not_sel={_fv(m,'stall_not_sel')}%"
              f"  membar={_fv(m,'stall_membar')}%"
              f"  barrier={_fv(m,'stall_barrier')}%"
              f"  short_sb={_fv(m,'stall_short_sb')}%"
              f"  mio={_fv(m,'stall_mio_throt')}%"
              f"  wait={_fv(m,'stall_wait')}%"
              f"  no_inst={_fv(m,'stall_no_inst')}%"
              f"  misc={_fv(m,'stall_misc')}%")
        print(f"  Pipe: tensor={_fv(m,'tensor_pipe_pct')}%"
              f"  hmma={_fv(m,'tensor_hmma_pct')}%"
              f"  alu={_fv(m,'alu_pipe_pct')}%"
              f"  fma={_fv(m,'fma_pipe_pct')}%"
              f"  lsu={_fv(m,'lsu_pipe_pct')}%")


def save_csv(rows: list[tuple[str, dict]], max_warps_per_sm: int, path: str) -> None:
    extra = ["warps_per_sm", "smem_total_kb", "bottleneck"]
    fieldnames = ["kernel"] + list(METRICS.keys()) + extra
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, m in rows:
            smem_kb = (float(m.get("smem_static_bytes") or 0)
                       + float(m.get("smem_dynamic_bytes") or 0)) / 1024
            occ = m.get("achieved_occ_pct")
            warps = (float(occ) / 100 * max_warps_per_sm if occ is not None else "")
            w.writerow({
                "kernel": name,
                **{k: (m[k] if m.get(k) is not None else "") for k in METRICS},
                "warps_per_sm": f"{warps:.1f}" if warps != "" else "",
                "smem_total_kb": f"{smem_kb:.1f}",
                "bottleneck": _bottleneck(m),
            })
    print(f"\nSaved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kernels", nargs="+",
                        help="CUDA kernel name(s) to profile")
    parser.add_argument("--size", type=int, default=4096, metavar="N")
    parser.add_argument("--out", default=None, metavar="FILE")
    args = parser.parse_args()

    bench_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, bench_dir)
    from bench_gpu import _all_impls
    all_i = _all_impls()

    import torch
    props = torch.cuda.get_device_properties(0)
    max_warps_per_sm = props.max_threads_per_multi_processor // 32
    print(f"GPU: {props.name}  (sm_{props.major}{props.minor})"
          f"  max {max_warps_per_sm} warps/SM  size={args.size}³")

    rows = []
    for kname in args.kernels:
        found = find_impl(kname, all_i)
        if found is None:
            print(f"  [{kname}] impl not found in _all_impls() — skipping")
            continue
        impl_name, module_path, dtype_str = found
        print(f"  profiling {kname}  (impl={impl_name}  dtype={dtype_str}) ...",
              end=" ", flush=True)
        result = profile_one(module_path, kname, args.size, dtype_str)
        if result is None:
            print("FAILED")
        else:
            print("done")
            rows.append((kname, result))

    if rows:
        print_results(rows, max_warps_per_sm)
        if args.out:
            save_csv(rows, max_warps_per_sm, args.out)
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
