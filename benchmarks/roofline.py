"""H800 roofline model — measures DRAM and L2 bandwidth, then plots.

H800 SXM specs:
  HBM3:        3.35 TB/s theoretical
  L2 cache:    50 MB
  BF16 tensor: 989 TFLOPS dense
  SMs:         132
"""

import numpy as np
import torch
import triton.testing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── DRAM bandwidth ─────────────────────────────────────────────────────────────

def measure_dram_bw():
    """Triad (A = B + s*C) with 512 MB arrays — forces every access to HBM."""
    N = 512 * 1024 * 1024 // 4   # float32 elements
    A = torch.empty(N, dtype=torch.float32, device="cuda")
    B = torch.ones (N, dtype=torch.float32, device="cuda")
    C = torch.ones (N, dtype=torch.float32, device="cuda")
    ms = triton.testing.do_bench(lambda: torch.add(B, C, alpha=2.0, out=A),
                                 warmup=10, rep=50, return_mode="min")
    bw = 3 * N * 4 / (ms * 1e-3) / 1e12
    print(f"DRAM triad  512 MB/array:  {bw:.3f} TB/s  ({ms:.2f} ms)")
    return bw

# ── L2 bandwidth ───────────────────────────────────────────────────────────────
# Same approach as bench_bandwidth.py: Python loop with CUDA events.
# Events measure GPU time only — no Python overhead in the timestamps.
# The GPU executes all 1000 queued copies back-to-back.
# Buffer (8 MB) << L2 (50 MB): after first warmup pass, data is L2-resident.

def measure_l2_bw():
    L2_MB = 8
    REPS  = 1000

    N = L2_MB * 1024 * 1024 // 4   # float32 elements, 8 MB
    a = torch.ones (N, dtype=torch.float32, device="cuda")
    b = torch.empty(N, dtype=torch.float32, device="cuda")

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    # Warmup: populate L2
    for _ in range(20):
        b.copy_(a)
    torch.cuda.synchronize()

    start.record()
    for _ in range(REPS):
        b.copy_(a)
    end.record()
    torch.cuda.synchronize()

    elapsed_s = start.elapsed_time(end) / 1e3
    bw = 2 * N * 4 * REPS / elapsed_s / 1e12   # read + write, REPS times
    print(f"L2   copy   {L2_MB} MB × {REPS} reps:  {bw:.3f} TB/s  ({elapsed_s*1e3:.1f} ms total)")
    return bw

# ── Roofline plot ──────────────────────────────────────────────────────────────

PEAK_BF16_TFLOPS = 989.0

def plot_roofline(dram_bw_tbs, l2_bw_tbs, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.logspace(-2, 5, 2000)   # 0.01 … 100000 FLOP/byte

    dram_perf = np.minimum(PEAK_BF16_TFLOPS, dram_bw_tbs * x)
    l2_perf   = np.minimum(PEAK_BF16_TFLOPS, l2_bw_tbs   * x)

    ax.loglog(x, dram_perf, color="#E53935", lw=2.5, label=f"DRAM  {dram_bw_tbs:.2f} TB/s (measured)")
    ax.loglog(x, l2_perf,   color="#8E24AA", lw=2.5, linestyle="--",
              label=f"L2    {l2_bw_tbs:.1f} TB/s (literature)")
    ax.axhline(PEAK_BF16_TFLOPS, color="#43A047", lw=2.5, linestyle=":",
               label=f"Compute  {PEAK_BF16_TFLOPS:.0f} TFLOPS BF16 (spec)")

    # Ridge points
    ridge_dram = PEAK_BF16_TFLOPS / dram_bw_tbs
    ridge_l2   = PEAK_BF16_TFLOPS / l2_bw_tbs
    for ridge, color, name in [(ridge_dram, "#E53935", "DRAM"), (ridge_l2, "#8E24AA", "L2")]:
        ax.axvline(ridge, color=color, lw=1, linestyle=":", alpha=0.6)
        ax.text(ridge * 1.08, 1.5, f"{name}\nridge\n{ridge:.0f} F/B",
                color=color, fontsize=8, va="bottom")

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=12)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=12)
    ax.set_title("H800 SXM Roofline Model", fontsize=14)
    ax.set_xlim(0.01, 100000)
    ax.set_ylim(0.01, 3000)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved → {out_path}")

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    dram_bw = measure_dram_bw()
    l2_bw   = 7.5   # TB/s — H800 L2, literature value

    print(f"\nDRAM efficiency vs 3.35 TB/s spec: {dram_bw/3.35*100:.1f}%")

    out = Path(__file__).parent / "roofline_h800.png"
    plot_roofline(dram_bw, l2_bw, str(out))
