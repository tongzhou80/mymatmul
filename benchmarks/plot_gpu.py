"""Plot GFLOPS vs matrix size for GPU implementations in results_gpu.csv."""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results_gpu.csv")
PLOT_FILE = os.path.join(os.path.dirname(__file__), "results_gpu.png")

# Theoretical peak GFLOPS
# RTX 4090: 16384 CUDA cores, 2.505 GHz boost, 2 FP32 ops per clock per core
GPU_4090_PEAK = 82_580.0  # GFLOPS (FP32)


def load_results(path):
    data = defaultdict(lambda: {"sizes": [], "gflops": []})
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            impl = row["impl"]
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

    ax.axhline(GPU_4090_PEAK, color="mediumseagreen", linestyle="--", linewidth=1.2,
               label=f"RTX 4090 peak  ({GPU_4090_PEAK:,.0f} GFLOPS)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: str(int(v))))
    ax.set_xlabel("Matrix size (N×N×N)", fontsize=12)
    ax.set_ylabel("GFLOPS (log scale)", fontsize=12)
    ax.set_title("Matrix multiplication: GPU attained GFLOPS vs size (FP32)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)
    print(f"Saved {PLOT_FILE}")


if __name__ == "__main__":
    data = load_results(RESULTS_FILE)
    plot(data)
