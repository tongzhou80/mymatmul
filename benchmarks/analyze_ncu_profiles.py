#!/usr/bin/env python3
"""Analyze ncu profiling reports and extract key metrics."""

import os
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

PROFILE_DIR = os.path.join(os.path.dirname(__file__), "profiles_ncu")

# Key metrics to extract from ncu
METRICS = [
    "l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_st.sum",
    "l1tex__data_bank_conflicts_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_shared_op_st.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__inst_executed.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_active",
    "gpu__time_duration.sum",
    "smsp__thread_inst_executed.sum",
    "dram__throughput.avg.pct_of_peak_sustained",
]


def extract_metrics(ncu_file):
    """Extract metrics from ncu report file."""
    metrics = {}

    if not os.path.exists(ncu_file):
        print(f"File not found: {ncu_file}")
        return metrics

    # Use ncu to export CSV format
    try:
        result = subprocess.run(
            ["ncu", "--csv", ncu_file],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"Error reading {ncu_file}: {result.stderr}")
            return metrics

        # Parse CSV output
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            # CSV format: metric_name, value, unit
            for line in lines[1:]:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    metric_name = parts[0]
                    try:
                        value = float(parts[1])
                        metrics[metric_name] = value
                    except (ValueError, IndexError):
                        pass

        return metrics

    except subprocess.TimeoutExpired:
        print(f"Timeout reading {ncu_file}")
        return metrics
    except Exception as e:
        print(f"Error: {e}")
        return metrics


def format_metrics(metrics):
    """Format metrics for display."""
    formatted = {}

    # Bank conflicts
    if "l1tex__data_bank_conflicts_shared_op_ld.sum" in metrics:
        formatted["Shared L1 Bank Conflicts (LD)"] = int(metrics["l1tex__data_bank_conflicts_shared_op_ld.sum"])
    if "l1tex__data_bank_conflicts_shared_op_st.sum" in metrics:
        formatted["Shared L1 Bank Conflicts (ST)"] = int(metrics["l1tex__data_bank_conflicts_shared_op_st.sum"])
    if "l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_ld.sum" in metrics:
        formatted["Global L1 Bank Conflicts (LD)"] = int(metrics["l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_ld.sum"])

    # Memory throughput
    if "dram__throughput.avg.pct_of_peak_sustained" in metrics:
        formatted["DRAM Throughput (% of peak)"] = f"{metrics['dram__throughput.avg.pct_of_peak_sustained']:.1f}%"

    # SM utilization
    if "sm__throughput.avg.pct_of_peak_sustained_active" in metrics:
        formatted["SM Throughput (% of peak)"] = f"{metrics['sm__throughput.avg.pct_of_peak_sustained_active']:.1f}%"
    if "sm__warps_active.avg.pct_of_peak_sustained_active" in metrics:
        formatted["SM Warp Activity (% of peak)"] = f"{metrics['sm__warps_active.avg.pct_of_peak_sustained_active']:.1f}%"

    # GPU time
    if "gpu__time_duration.sum" in metrics:
        time_us = metrics["gpu__time_duration.sum"]
        formatted["GPU Time"] = f"{time_us:.0f} µs ({time_us/1000:.3f} ms)"

    return formatted


def main():
    if not os.path.exists(PROFILE_DIR):
        print(f"No profiles directory found: {PROFILE_DIR}")
        sys.exit(1)

    # Find all ncu-rep files
    profile_files = sorted(Path(PROFILE_DIR).glob("*.ncu-rep"))

    if not profile_files:
        print(f"No .ncu-rep files found in {PROFILE_DIR}")
        sys.exit(1)

    print("=" * 80)
    print("NCU PROFILING ANALYSIS")
    print("=" * 80)

    all_results = {}

    for profile_file in profile_files:
        print(f"\n{profile_file.stem}")
        print("-" * 80)

        metrics = extract_metrics(str(profile_file))

        if not metrics:
            print("No metrics extracted")
            continue

        formatted = format_metrics(metrics)
        all_results[profile_file.stem] = (metrics, formatted)

        for key, value in formatted.items():
            print(f"  {key:.<40} {value}")

    # Comparison summary
    print("\n" + "=" * 80)
    print("SUMMARY: Key Bottleneck Indicators")
    print("=" * 80)

    print("\n1. Shared Memory Bank Conflicts (Lower is Better):")
    for name, (metrics, _) in sorted(all_results.items()):
        conflicts_ld = metrics.get("l1tex__data_bank_conflicts_shared_op_ld.sum", 0)
        conflicts_st = metrics.get("l1tex__data_bank_conflicts_shared_op_st.sum", 0)
        total = int(conflicts_ld + conflicts_st)
        print(f"  {name:.<40} {total:>12,} conflicts")

    print("\n2. SM Throughput Utilization (Higher is Better):")
    for name, (metrics, _) in sorted(all_results.items()):
        throughput = metrics.get("sm__throughput.avg.pct_of_peak_sustained_active", 0)
        print(f"  {name:.<40} {throughput:>11.1f}%")

    print("\n3. DRAM Throughput Utilization (Higher is Better):")
    for name, (metrics, _) in sorted(all_results.items()):
        throughput = metrics.get("dram__throughput.avg.pct_of_peak_sustained", 0)
        print(f"  {name:.<40} {throughput:>11.1f}%")


if __name__ == "__main__":
    main()
