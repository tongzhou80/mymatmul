#!/usr/bin/env python3
"""Extract metrics from ncu reports using ncu's metric query capabilities."""

import subprocess
import os
from pathlib import Path
import re

PROFILE_DIR = os.path.join(os.path.dirname(__file__), "profiles_ncu")

# Core metrics we want to extract
METRIC_QUERIES = {
    "l1tex_shared_ld_bank_conflicts": "l1tex__data_bank_conflicts_shared_op_ld.sum",
    "l1tex_shared_st_bank_conflicts": "l1tex__data_bank_conflicts_shared_op_st.sum",
    "dram_read_bytes": "dram__bytes_read.sum",
    "dram_write_bytes": "dram__bytes_write.sum",
    "l1tex_requests": "l1tex__t_requests.sum",
    "elapsed_cycles": "gpu__time_duration.sum",
    "sm_instruction_throughput": "sm__throughput.avg.pct_of_peak_sustained_active",
    "dram_throughput_pct": "dram__throughput.avg.pct_of_peak_sustained",
}


def run_ncu_metric_query(report_file, metric_name):
    """Run ncu to query a specific metric from a report."""
    try:
        # Use ncu to print the report with metrics
        result = subprocess.run(
            ["ncu", "--metrics", metric_name, report_file],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            # Parse the output to find the metric value
            # Format typically looks like: "metric_name,value,unit"
            for line in result.stdout.split('\n'):
                if metric_name in line or ',' in line:
                    # Try to extract the numeric value
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            return float(parts[1].strip())
                        except (ValueError, IndexError):
                            pass
        return None
    except Exception as e:
        return None


def extract_all_metrics(report_file):
    """Extract all metrics from a single report file."""
    metrics = {}

    print(f"Extracting metrics from {Path(report_file).name}...")

    for friendly_name, metric_name in METRIC_QUERIES.items():
        value = run_ncu_metric_query(report_file, metric_name)
        if value is not None:
            metrics[friendly_name] = value

    return metrics


def display_kernel_metrics(kernel_name, metrics):
    """Display metrics for a single kernel."""
    print(f"\n{kernel_name}")
    print("-" * 70)

    if not metrics:
        print("  No metrics extracted")
        return

    # Shared memory bank conflicts
    if "l1tex_shared_ld_bank_conflicts" in metrics:
        ld_conflicts = int(metrics.get("l1tex_shared_ld_bank_conflicts", 0))
        st_conflicts = int(metrics.get("l1tex_shared_st_bank_conflicts", 0))
        total_conflicts = ld_conflicts + st_conflicts
        print(f"  Shared L1 Bank Conflicts")
        print(f"    Load:  {ld_conflicts:>12,}")
        print(f"    Store: {st_conflicts:>12,}")
        print(f"    Total: {total_conflicts:>12,}")

    # Memory throughput
    print(f"\n  Memory Throughput")
    if "dram_throughput_pct" in metrics:
        print(f"    DRAM: {metrics['dram_throughput_pct']:>11.1f}% of peak")

    # SM utilization
    print(f"\n  Compute Utilization")
    if "sm_instruction_throughput" in metrics:
        print(f"    SM Throughput: {metrics['sm_instruction_throughput']:>7.1f}% of peak")

    # Memory statistics
    print(f"\n  Memory Accesses")
    if "l1tex_requests" in metrics:
        print(f"    L1 Requests: {int(metrics['l1tex_requests']):>14,}")
    if "dram_read_bytes" in metrics:
        read_gb = metrics["dram_read_bytes"] / 1e9
        print(f"    DRAM Reads:  {read_gb:>14.2f} GB")
    if "dram_write_bytes" in metrics:
        write_gb = metrics["dram_write_bytes"] / 1e9
        print(f"    DRAM Writes: {write_gb:>14.2f} GB")


def main():
    profile_files = sorted(Path(PROFILE_DIR).glob("*.ncu-rep"))

    if not profile_files:
        print(f"No profile files found in {PROFILE_DIR}")
        return

    print("=" * 70)
    print("NCU PROFILE ANALYSIS")
    print("=" * 70)

    all_results = {}

    for profile_file in profile_files:
        kernel_name = profile_file.stem
        metrics = extract_all_metrics(str(profile_file))
        all_results[kernel_name] = metrics
        display_kernel_metrics(kernel_name, metrics)

    # Comparative analysis
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)

    print("\nShared Memory Bank Conflicts (Lower is Better)")
    print("-" * 70)
    conflicts_data = []
    for kernel_name, metrics in sorted(all_results.items()):
        if "l1tex_shared_ld_bank_conflicts" in metrics:
            total = int(metrics.get("l1tex_shared_ld_bank_conflicts", 0) +
                       metrics.get("l1tex_shared_st_bank_conflicts", 0))
            conflicts_data.append((kernel_name, total))

    for kernel_name, total in sorted(conflicts_data, key=lambda x: x[1]):
        print(f"  {kernel_name:.<45} {total:>12,}")

    print("\nSM Throughput Utilization (Higher is Better)")
    print("-" * 70)
    throughput_data = []
    for kernel_name, metrics in sorted(all_results.items()):
        if "sm_instruction_throughput" in metrics:
            throughput_data.append((kernel_name, metrics["sm_instruction_throughput"]))

    for kernel_name, throughput in sorted(throughput_data, key=lambda x: x[1], reverse=True):
        print(f"  {kernel_name:.<45} {throughput:>11.1f}%")


if __name__ == "__main__":
    main()
