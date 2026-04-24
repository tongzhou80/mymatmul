"""GPU memory bandwidth benchmark (STREAM-style).

Measures peak achievable HBM/GDDR bandwidth using Copy, Scale, Add, and Triad
kernels — all with buffers much larger than L2 cache to force DRAM access.

RTX 4090 specs:
  DRAM:        GDDR6X, 1008 GB/s theoretical peak
  L2 cache:    72 MB
  Buffer size: 512 MB per array (well above L2)
"""

import time
import numpy as np
import torch

WARMUP  = 5
TRIALS  = 20
# 512 MB per array of float32 (4 bytes) = 128M elements
N       = 512 * 1024 * 1024 // 4

def timed(fn, trials=TRIALS, warmup=WARMUP):
    """Returns best elapsed time in seconds using CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    ms_list = []
    for _ in range(trials):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ms_list.append(start.elapsed_time(end))

    return np.min(ms_list) / 1e3, np.mean(ms_list) / 1e3   # seconds


def main():
    print(f"Buffer size per array: {N * 4 / 1024**3:.2f} GB  ({N} float32 elements)")
    print(f"Warmup: {WARMUP}  Trials: {TRIALS}\n")

    a = torch.ones (N, dtype=torch.float32, device='cuda')
    b = torch.empty(N, dtype=torch.float32, device='cuda')
    c = torch.empty(N, dtype=torch.float32, device='cuda')
    scalar = 2.0

    # ---------- Copy: b = a   (1 read + 1 write = 2N bytes) ----------
    best, mean = timed(lambda: b.copy_(a))
    bw_best = 2 * N * 4 / best  / 1e9
    bw_mean = 2 * N * 4 / mean  / 1e9
    print(f"Copy   (2 arrays, 2N bytes):  best {bw_best:6.1f} GB/s  mean {bw_mean:6.1f} GB/s")

    # ---------- Scale: b = scalar * a  (1 read + 1 write = 2N bytes) ----------
    best, mean = timed(lambda: torch.mul(a, scalar, out=b))
    bw_best = 2 * N * 4 / best  / 1e9
    bw_mean = 2 * N * 4 / mean  / 1e9
    print(f"Scale  (2 arrays, 2N bytes):  best {bw_best:6.1f} GB/s  mean {bw_mean:6.1f} GB/s")

    # ---------- Add: c = a + b  (2 reads + 1 write = 3N bytes) ----------
    best, mean = timed(lambda: torch.add(a, b, out=c))
    bw_best = 3 * N * 4 / best  / 1e9
    bw_mean = 3 * N * 4 / mean  / 1e9
    print(f"Add    (3 arrays, 3N bytes):  best {bw_best:6.1f} GB/s  mean {bw_mean:6.1f} GB/s")

    # ---------- Triad: a = b + scalar * c  (2 reads + 1 write = 3N bytes) ----------
    best, mean = timed(lambda: torch.add(b, c, alpha=scalar, out=a))
    bw_best = 3 * N * 4 / best  / 1e9
    bw_mean = 3 * N * 4 / mean  / 1e9
    print(f"Triad  (3 arrays, 3N bytes):  best {bw_best:6.1f} GB/s  mean {bw_mean:6.1f} GB/s")

    # ---------- Read-only: sum(a)  (1 read = N bytes) ----------
    best, mean = timed(lambda: a.sum())
    bw_best = N * 4 / best  / 1e9
    bw_mean = N * 4 / mean  / 1e9
    print(f"Read   (1 array,  1N bytes):  best {bw_best:6.1f} GB/s  mean {bw_mean:6.1f} GB/s")

    print(f"\nTheoretical peak: 1008 GB/s")


def measure_l2_bandwidth():
    """Measure L2 cache bandwidth by repeatedly accessing a small buffer.

    RTX 4090: L1 = 128KB/SM, L2 = 72MB total.
    Buffer must be: >> L1 (to avoid L1 hits) AND << L2 (to stay cached).
    We use 8MB: well above any single SM's L1, well below 72MB L2.

    We repeat the access REPS times. After the first pass populates L2,
    subsequent passes are served entirely from L2. Total bytes / time = L2 BW.
    """
    L2_BUFFER_MB = 8
    REPS = 1000
    WARMUP = 10

    N = L2_BUFFER_MB * 1024 * 1024 // 4   # float32 elements
    a = torch.ones (N, dtype=torch.float32, device='cuda')
    b = torch.empty(N, dtype=torch.float32, device='cuda')

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    # Warmup: populate L2
    for _ in range(WARMUP):
        for _ in range(10):
            b.copy_(a)
    torch.cuda.synchronize()

    # Timed: all accesses served from L2
    start.record()
    for _ in range(REPS):
        b.copy_(a)
    end.record()
    torch.cuda.synchronize()

    elapsed_s = start.elapsed_time(end) / 1e3
    total_bytes = 2 * N * 4 * REPS          # read + write, REPS times
    bw = total_bytes / elapsed_s / 1e12      # TB/s

    print(f"\nL2 bandwidth measurement")
    print(f"  Buffer: {L2_BUFFER_MB} MB  ({N} float32, well inside 72MB L2)")
    print(f"  Reps:   {REPS}  x  2 arrays  =  {total_bytes/1e9:.1f} GB total")
    print(f"  Time:   {elapsed_s*1e3:.1f} ms")
    print(f"  L2 bandwidth (copy): {bw:.2f} TB/s")


if __name__ == "__main__":
    main()
    measure_l2_bandwidth()
