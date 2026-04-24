"""Run NCU DRAM bandwidth profiling for s4 configs at a given size."""
import subprocess, re, sys

NCU = "/usr/local/cuda-12.8/bin/ncu"
METRICS = ",".join([
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
])

SZ = int(sys.argv[1]) if len(sys.argv) > 1 else 4096

CONFIGS = [
    "s4_tm8_tn8_bm128_bn64_bk16_u16",
    "s4_tm8_tn8_bm128_bn128_bk16_u16",
    "s4_tm8_tn8_bm64_bn64_bk16_u16",
    "s4_tm8_tn8_bm128_bn64_bk16_u8",
    "s4_tm8_tn8_bm128_bn128_bk16_u8",
    "s4_tm8_tn8_bm64_bn64_bk16_u8",
    "s4b_tm8_tn8_bm128_bn128_bk16_u16",
]

def parse_last(output, metric):
    """Return the last occurrence of a metric value from NCU output."""
    val = None
    for line in output.splitlines():
        if metric in line:
            parts = line.split()
            # value is second-to-last (before unit or after metric name)
            try:
                val = float(parts[-2].replace(",", ""))
            except (ValueError, IndexError):
                try:
                    val = float(parts[-1].replace(",", ""))
                except (ValueError, IndexError):
                    pass
    return val

print(f"\nN={SZ}  (theoretical: A+B = {2*SZ*SZ*2/1e6:.0f} MB, C = {SZ*SZ*2/1e6:.0f} MB)\n")
print(f"{'config':<42} {'DRAM_r':>8} {'DRAM_w':>8} {'DRAM%':>7} {'FMA%':>7} {'SM%':>7}")
print(f"{'------':<42} {'(MB)':>8} {'(MB)':>8} {'':>7} {'':>7} {'':>7}")

for cfg in CONFIGS:
    kname = "matmul_s4b" if "s4b" in cfg else "matmul_s4"
    cmd = [NCU, "--metrics", METRICS, "--kernel-name", kname,
           "python", "benchmarks/profile_dram.py", cfg, str(SZ)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout + result.stderr

    dr   = parse_last(out, "dram__bytes_read.sum")
    dw   = parse_last(out, "dram__bytes_write.sum")
    dbw  = parse_last(out, "dram__throughput.avg.pct_of_peak_sustained_elapsed")
    fma  = parse_last(out, "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed")
    sm   = parse_last(out, "sm__throughput.avg.pct_of_peak_sustained_elapsed")

    def fmt(v): return f"{v:8.1f}" if v is not None else f"{'?':>8}"
    print(f"{cfg:<42} {fmt(dr)} {fmt(dw)} {fmt(dbw) if dbw else '?':>7} {fmt(fma) if fma else '?':>7} {fmt(sm) if sm else '?':>7}")
