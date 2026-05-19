"""Empirical TMEM layout prober.

Runs the b3_tc05 kernel with a known A, B (so reference D = A@B is unique-valued
per (m, n)), then captures every TMEM register the kernel reads via
tcgen05.ld.32x32b.x32, and reverse-matches each register value to a (m, n) in D
to discover the lane/reg → TMEM (m, n) mapping.
"""

import os
import numpy as np
import torch
import pycuda.driver as drv

from .._pycuda_loader import get_module_jit, SM_ARCH

_GPU_DIR = os.path.dirname(os.path.abspath(__file__))
_CU = os.path.join(_GPU_DIR, "_tmem_probe.cu")
_CUBIN = os.path.join(_GPU_DIR, f"_tmem_probe_{SM_ARCH}.cubin")


def main():
    M, N, K = 128, 128, 16
    # Pick A, B so D[m, n] = 4096*m + n — a bijection over (m, n) in [0,128)^2.
    # All values bf16-exact (powers of 2 × small ints).  No collisions, no fp drift.
    # D[m, n] = m + 128*n + 1. All operands and partial sums fit bf16 exactly.
    # +1 so value 0 unambiguously means "uninitialized".
    A = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")
    A[:, 0] = (torch.arange(0, M, dtype=torch.float32) + 1).to(torch.bfloat16)  # m+1
    A[:, 1] = 1
    B = torch.zeros(K, N, dtype=torch.bfloat16, device="cuda")
    B[0, :] = 1
    B[1, :] = (128 * torch.arange(0, N, dtype=torch.float32)).to(torch.bfloat16)
    m_idx = torch.arange(0, M, dtype=torch.float32).unsqueeze(1)
    n_idx = torch.arange(0, N, dtype=torch.float32).unsqueeze(0)
    D_ref = m_idx + 128 * n_idx + 1

    # 4 warps × 32 lanes × 4 calls × 64 regs = 32768 fp32 = 128 KB.
    out = torch.zeros(32768, dtype=torch.float32, device="cuda")

    mod = get_module_jit(_CU, _CUBIN, ["-arch=sm_100a"])
    fn = mod.get_function("tmem_probe_bm128_bn128_bk16_nw4")
    smem = (128 + 128) * 16 * 2 + 32
    fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
    fn(np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(out.data_ptr()),
       np.int32(M), np.int32(K), np.int32(N),
       block=(32, 4, 1), grid=(1, 1, 1), shared=smem)
    torch.cuda.synchronize()

    # Layout: [4 warps][32 lanes][4 calls (offsets 0,4,16,20)][64 regs]
    dbg = out.view(4, 32, 4, 64).cpu().numpy()
    D = D_ref.cpu().numpy()                       # [128, 128]
    D_flat = D.reshape(-1)                        # 16384 unique-ish values

    # For each (warp, lane, chunk, reg), find which (m, n) in D matches.
    print(f"D range: min={D.min():.3f} max={D.max():.3f}")
    print(f"dbg range: min={dbg.min():.3f} max={dbg.max():.3f}")
    print()

    def find(val, atol=1e-2):
        diff = np.abs(D - val)
        idx = np.argwhere(diff < atol)
        return idx

    # Probe warp 0, lane 0, chunk 0, regs 0..7
    print("=== warp 0, lane 0, chunk 0 ===")
    for r in range(8):
        v = dbg[0, 0, 0, r]
        hits = find(v)
        s = ",".join(f"({m},{n})" for m, n in hits[:5])
        print(f"  reg[{r:2d}] = {v:8.3f}  → D positions: {s}")

    print("=== warp 0, lane 1, chunk 0 ===")
    for r in range(8):
        v = dbg[0, 1, 0, r]
        hits = find(v)
        s = ",".join(f"({m},{n})" for m, n in hits[:5])
        print(f"  reg[{r:2d}] = {v:8.3f}  → D positions: {s}")

    print("=== warp 0, lane 0, chunk 1 (base col 32) ===")
    for r in range(8):
        v = dbg[0, 0, 1, r]
        hits = find(v)
        s = ",".join(f"({m},{n})" for m, n in hits[:5])
        print(f"  reg[{r:2d}] = {v:8.3f}  → D positions: {s}")

    print("=== warp 1, lane 0, chunk 0 ===")
    for r in range(8):
        v = dbg[1, 0, 0, r]
        hits = find(v)
        s = ",".join(f"({m},{n})" for m, n in hits[:5])
        print(f"  reg[{r:2d}] = {v:8.3f}  → D positions: {s}")

    # Sum-decomposition decoder: v = m + 128n + 1; v == 0 means uninitialized.
    def best_match(v):
        if abs(v) < 0.5:
            return "ZERO"
        iv = int(round(v)) - 1
        n, m = divmod(iv, 128)
        if 0 <= m < 128 and 0 <= n < 128 and abs(m + 128*n + 1 - v) < 0.5:
            return (m, n)
        return None

    # 1. Lane → TMEM row, for warp 0, call 0, reg 0
    print("\n=== Discover lane → row mapping (warp 0, call 0, reg 0) ===")
    for L in range(32):
        v = dbg[0, L, 0, 0]
        m = best_match(v)
        print(f"  lane {L:2d}: v={v:8.3f}  →  {m}")

    # 2. Same for warp 1
    print("\n=== Discover lane → row mapping (warp 1, call 0, reg 0) ===")
    for L in range(32):
        v = dbg[1, L, 0, 0]
        m = best_match(v)
        print(f"  lane {L:2d}: v={v:8.3f}  →  {m}")

    # 3. Reg → (m,n) for warp 0, lane 0, all 4 calls
    print("\n=== reg → (m,n) for warp 0, lane 0 (all 4 calls, offsets {0,4,16,20}) ===")
    for c in range(4):
        line = f"  call {c}: "
        for r in range(64):
            v = dbg[0, 0, c, r]
            m = best_match(v)
            if m is not None and m != "ZERO":
                line += f"r{r:02d}=({m[0]:3d},{m[1]:3d}) "
            elif m == "ZERO":
                line += f"r{r:02d}=ZERO       "
            else:
                line += f"r{r:02d}=?         "
            if (r+1) % 8 == 0:
                print(line); line = "           "

    # 4. lane 1 to see if odd lanes hold valid data
    print("\n=== reg → (m,n) for warp 0, lane 1, call 0 ===")
    line = ""
    for r in range(64):
        v = dbg[0, 1, 0, r]
        m = best_match(v)
        if m is not None and m != "ZERO":
            line += f"r{r:02d}=({m[0]:3d},{m[1]:3d}) "
        elif m == "ZERO":
            line += f"r{r:02d}=ZERO       "
        else:
            line += f"r{r:02d}=?         "
        if (r+1) % 8 == 0:
            print(line); line = ""


if __name__ == "__main__":
    main()
