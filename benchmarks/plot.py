"""Plot GFLOPS vs matrix size for all implementations in results.csv."""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.csv")
PLOT_FILE = os.path.join(os.path.dirname(__file__), "results.png")

# Theoretical peak GFLOPS
# AMD EPYC 7773X: 128 physical cores, AVX2 (8 FP32/vec), 2 FMA units, FMA=2 ops, 3527.7 MHz
_CORES = 128
_GHZ = 3.5277
_FP32_PER_VEC = 8   # AVX2 256-bit
_FMA_OPS = 2        # FMA counts as 2 floating-point ops (multiply + add)
_FMA_UNITS = 2      # 2 FMA units per core (superscalar)
CPU_SINGLE_CORE_PEAK = _FP32_PER_VEC * _FMA_OPS * _FMA_UNITS * _GHZ          # ~112.9 GFLOPS
CPU_ALL_CORE_PEAK    = _CORES * _FP32_PER_VEC * _FMA_OPS * _FMA_UNITS * _GHZ  # ~14438 GFLOPS
GPU_4090_PEAK        = 82_580.0                                          # GFLOPS (FP32)


EXCLUDE = {"numba_ijk", "numba_ikj", "cpp_ikj_vec"}


def load_results(path):
    data = defaultdict(lambda: {"sizes": [], "gflops": []})
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            impl = row["impl"]
            if impl in EXCLUDE:
                continue
            size = int(row["M"])
            if size == 64:  # skip 64
                continue
            data[impl]["sizes"].append(size)
            data[impl]["gflops"].append(float(row["gflops"]))
    return data


def plot(data):
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "v", "P", "*"]
    for i, (impl, d) in enumerate(sorted(data.items())):
        sizes = d["sizes"]
        gflops = d["gflops"]
        ax.plot(sizes, gflops, marker=markers[i % len(markers)],
                label=impl, linewidth=2, markersize=7)

    all_sizes = sorted({s for d in data.values() for s in d["sizes"]})
    x_min, x_max = all_sizes[0] * 0.8, all_sizes[-1] * 1.2

    ax.axhline(CPU_SINGLE_CORE_PEAK, color="steelblue", linestyle="--", linewidth=1.2,
               label=f"CPU single-core peak  ({CPU_SINGLE_CORE_PEAK:.0f} GFLOPS)")
    ax.axhline(CPU_ALL_CORE_PEAK, color="darkorange", linestyle="--", linewidth=1.2,
               label=f"CPU all-core peak  ({CPU_ALL_CORE_PEAK:.0f} GFLOPS)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: str(int(v))))
    ax.set_xlabel("Matrix size (N×N×N)", fontsize=12)
    ax.set_ylabel("GFLOPS (log scale)", fontsize=12)
    ax.set_title("Matrix multiplication: attained GFLOPS vs size", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)
    print(f"Saved {PLOT_FILE}")


if __name__ == "__main__":
    data = load_results(RESULTS_FILE)
    plot(data)
