"""Benchmark GPU matmul implementations and write results to benchmarks/results_gpu.csv."""
"""We should use bench_gpu2.py which uses triton's timing API."""
import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import torch

# Registry: name -> (dotpath, max_size or None for no limit)
# max_size: skip sizes larger than this (leaves empty cells in the results table)
IMPLEMENTATIONS = {
    "triton_fp32simt_autotuned": ("mymatmul.gpu.matmul_triton.triton_fp32simt_autotuned", None),
    "triton_fp32simt_bm128_bn128_bk32_w8_s4": ("mymatmul.gpu.matmul_triton.triton_fp32simt_bm128_bn128_bk32_w8_s4", None),
    "triton_fp32simt_bm128_bn256_bk16": ("mymatmul.gpu.matmul_triton.triton_fp32simt_bm128_bn256_bk16", None),
    "torch_matmul": ("mymatmul.gpu.matmul_torch.matmul_torch",   None),
    # cuBLAS FP32 with TF32 disabled (pure FP32 SIMT, comparable to our s4 kernels)
    "cublas_fp32_notf32": ("mymatmul.gpu.matmul_torch.matmul_torch_fp32_notf32", None),
    # Triton FP32 SIMT (allow_tf32=False) — directly comparable to our s4 CUDA kernels
    "triton_fp32simt_bm128_bn128_bk16": ("mymatmul.gpu.matmul_triton.triton_fp32simt_bm128_bn128_bk16", None),
    "triton_fp32simt_bm128_bn64_bk16":  ("mymatmul.gpu.matmul_triton.triton_fp32simt_bm128_bn64_bk16",  None),
    "triton_fp32simt_bm64_bn64_bk16":   ("mymatmul.gpu.matmul_triton.triton_fp32simt_bm64_bn64_bk16",   None),
    "triton_fp32simt_bm128_bn128_bk32": ("mymatmul.gpu.matmul_triton.triton_fp32simt_bm128_bn128_bk32", None),
    "triton_fp32simt_bm128_bn64_bk32":  ("mymatmul.gpu.matmul_triton.triton_fp32simt_bm128_bn64_bk32",  None),
    "triton_fp32simt_bm64_bn64_bk32":   ("mymatmul.gpu.matmul_triton.triton_fp32simt_bm64_bn64_bk32",   None),
    "cuda_naive_ijk": ("mymatmul.gpu.cuda_core.matmul_cuda.matmul_cuda_naive_ijk", None),
    "cuda_naive_ijk_jx": ("mymatmul.gpu.cuda_core.matmul_cuda.matmul_cuda_naive_ijk_jx", None),
    "cuda_tiled_32x32": ("mymatmul.gpu.cuda_core.matmul_cuda.matmul_cuda_tiled_32x32", None),
    "cuda_tiled_32x32_16x16": ("mymatmul.gpu.cuda_core.matmul_cuda.matmul_cuda_tiled_32x32_threads_16x16", None),
    "cuda_tiled_32x32_32x8": ("mymatmul.gpu.cuda_core.matmul_cuda.matmul_cuda_tiled_32x32_threads_32x8", None),
    "cuda_tiled_32x32_32x4": ("mymatmul.gpu.cuda_core.matmul_cuda.matmul_cuda_tiled_32x32_threads_32x4", None),
    "cuda_tiled_32x64_32x4": ("mymatmul.gpu.cuda_core.matmul_cuda.matmul_cuda_tiled_32x64_threads_32x4", None),
    "cuda_tiled_32x64_tm4_tn4": ("mymatmul.gpu.cuda_core.matmul_cuda.matmul_cuda_tiled_32x64_tm4_tn4", None),
    # Stage 3: BK=32, unroll=8
    **{f"s3_{k}_bk32_u8": (f"mymatmul.gpu.cuda_core.matmul_cuda_s3.matmul_s3_{k}_bk32_u8", None)
       for k in ["tm4_tn4_bm32_bn64","tm4_tn4_bm64_bn64","tm8_tn4_bm64_bn64","tm8_tn8_bm128_bn64","tm8_tn8_bm128_bn128"]},
    # Stage 3: BK=16, unroll=1,2,4,8
    **{f"s3_{k}_bk16_u{u}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s3.matmul_s3_{k}_bk16_u{u}", None)
       for u in [1, 2, 4, 8]
       for k in ["tm4_tn4_bm32_bn64","tm4_tn4_bm64_bn64","tm8_tn4_bm64_bn64","tm8_tn8_bm128_bn64","tm8_tn8_bm128_bn128"]},
    # Stage 4: double-buffered with cp.async, BK=16 (small configs: fixed full unroll)
    **{f"s4_{k}_bk16": (f"mymatmul.gpu.cuda_core.matmul_cuda_s4.matmul_s4_{k}_bk16", None)
       for k in ["tm4_tn4_bm32_bn64","tm4_tn4_bm64_bn64","tm8_tn4_bm64_bn64"]},
    # Stage 4: large configs, sweep compute-loop unroll 1,2,4,8,16
    **{f"s4_{k}_bk16_u{u}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s4.matmul_s4_{k}_bk16_u{u}", None)
       for u in [1, 2, 4, 8, 16]
       for k in ["tm8_tn8_bm128_bn64", "tm8_tn8_bm128_bn128", "tm8_tn8_bm64_bn64"]},
    # Stage 4 Strided: strided output assignment → consecutive B reads per warp step
    **{f"s4st_{k}_bk16_u{u}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s4st.matmul_s4st_{k}_bk16_u{u}", None)
       for u in [1, 4, 8, 16]
       for k in ["tm8_tn8_bm64_bn64", "tm8_tn8_bm128_bn128", "tm8_tn8_bm128_bn64", "tm8_tn8_bm64_bn128"]},
    # Stage 4 Strided BK=32: halves tile count but adds 2-way A bank conflicts
    # (bm128_bn128_bk32 needs 64KB smem → exceeds 48KB limit, not instantiated)
    **{f"s4st_{k}_bk32_u{u}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s4st.matmul_s4st_{k}_bk32_u{u}", None)
       for u in [1, 4, 8, 16, 32]
       for k in ["tm8_tn8_bm64_bn64", "tm8_tn8_bm128_bn64"]},
    # Stage 4b: Stage 4 + A_shared bank-conflict fix (BK+4 padding), BN=128 only
    **{f"s4b_tm8_tn8_bm128_bn128_bk16_u{u}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s4b.matmul_s4b_tm8_tn8_bm128_bn128_bk16_u{u}", None)
       for u in [8, 16]},
    # Stage 4 + A-swizzle: XOR swizzle on A_shared to eliminate bank conflicts
    **{f"s4sw_{k}_bk16_u{u}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s4sw.matmul_s4sw_{k}_bk16_u{u}", None)
       for u in [1, 2, 4, 8, 16]
       for k in ["tm8_tn8_bm128_bn128", "tm8_tn8_bm128_bn64", "tm8_tn8_bm64_bn64"]},
    # s4st BK=32 dynamic smem: bm128_bn128_bk32 needs 64 KB (> 48 KB static limit)
    **{f"s4st_bk32_tm8_tn8_bm128_bn128_bk32_u{u}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s4st_bk32.matmul_s4st_bk32_tm8_tn8_bm128_bn128_bk32_u{u}", None)
       for u in [1, 4, 8, 16, 32]},
    # Stage 4 Strided-2: 2-contiguous output assignment → float2 B smem loads, zero conflicts
    **{f"s4st2_tm8_tn8_bm128_bn128_bk16_u{u}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s4st2.matmul_s4st2_tm8_tn8_bm128_bn128_bk16_u{u}", None)
       for u in [1, 4, 8, 16]},
    # s4st2 BK=32 dynamic smem: float2 B loads + larger tile (64 KB)
    **{f"s4st2_bk32_tm8_tn8_bm128_bn128_bk32_u{u}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s4st2_bk32.matmul_s4st2_bk32_tm8_tn8_bm128_bn128_bk32_u{u}", None)
       for u in [1, 4, 8, 16, 32]},
    # Stage 4 Strided-4: float4 B reads + 8×4 warp layout, 2-way A conflict (no swizzle)
    **{f"s4st4_tm8_tn8_bm128_bn128_bk16_u{u}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s4st4.matmul_s4st4_tm8_tn8_bm128_bn128_bk16_u{u}", None)
       for u in [1, 4, 8, 16]},
    # Stage 4 Strided-4 XOR: float4 B reads + 8×4 warp layout + XOR A swizzle (zero conflicts)
    **{f"s4st4_xor_tm8_tn8_bm128_bn128_bk16_u{u}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s4st4_xor.matmul_s4st4_xor_tm8_tn8_bm128_bn128_bk16_u{u}", None)
       for u in [1, 4, 8, 16]},
    # s4st + intra-warp shuffle to reduce smem reads (experimental)
    "s4st_shfl_tm8_tn8_bm128_bn128_bk16": (
        "mymatmul.gpu.cuda_core.matmul_cuda_s4st_shfl.matmul_s4st_shfl_tm8_tn8_bm128_bn128_bk16", None),
    # s4st TN=16 PTX: inline PTX immediate-offset smem loads; higher FMA:load ratio
    **{f"s4st_tn16_ptx_bm128_bn256_bk16_u{u}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s4st_tn16_ptx.matmul_s4st_tn16_ptx_tm8_tn16_bm128_bn256_bk16_u{u}", None)
       for u in [1, 2, 4, 8, 16]},
    # s4st TN=16 pure C++ (no inline PTX): compiler has full scheduling freedom
    **{f"s4st_tn16_bm128_bn256_bk16_u{u}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s4st_tn16.matmul_s4st_tn16_tm8_tn16_bm128_bn256_bk16_u{u}", None)
       for u in [1, 2, 4, 8]},
    # s4st TN=16 m2: hand-crafted 2-way interleaving (loads×2 → FMAs×2), unroll 1 outer loop
    "s4st_tn16_m2_bm128_bn256_bk16": (
        "mymatmul.gpu.cuda_core.matmul_cuda_s4st_tn16_m2.matmul_s4st_tn16_m2_tm8_tn16_bm128_bn256_bk16", None),
    # s4st TN=16 p1: register-prefetch software pipeline (193 regs, below 255 cliff)
    "s4st_tn16_p1_bm128_bn256_bk16": (
        "mymatmul.gpu.cuda_core.matmul_cuda_s4st_tn16_p1.matmul_s4st_tn16_p1_tm8_tn16_bm128_bn256_bk16", None),
    # s4st TN=16 f2: float2 B smem loads (u1=168 regs; u2/u4=255 regs)
    **{f"s4st2_tn16_bm128_bn256_bk16_u{u}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s4st2_tn16.matmul_s4st2_tn16_tm8_tn16_bm128_bn256_bk16_u{u}", None)
       for u in [1, 2, 4, 8, 16]},
    # Stage 5: auto-tuned over BM/BN/BK/UNROLL (64 configs, 16x16 thread layout, float2 B)
    "s5_autotuned": ("mymatmul.gpu.cuda_core.matmul_cuda_s5.matmul_s5", None),
    "s5_bm256_bn128_bk32_u16": ("mymatmul.gpu.cuda_core.matmul_cuda_s5.matmul_s5_bm256_bn128_bk32_u16", None),
    "s5_bm256_bn128_bk16_u16": ("mymatmul.gpu.cuda_core.matmul_cuda_s5.matmul_s5_bm256_bn128_bk16_u16", None),
    # Stage 5 W4: warp-tiled (4x2 inter-warp, 4x8 intra-warp) with float4 B smem loads
    "s5_w4_autotuned": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4.matmul_s5_w4", None),
    "s5_w4_bm256_bn128_bk16_u16": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4.matmul_s5_w4_bm256_bn128_bk16_u16", None),
    "s5_w4_bm256_bn128_bk8_u16": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4.matmul_s5_w4_bm256_bn128_bk8_u16", None),
    # Stage 5 W4B: s5_w4 + A-tile row padding (+4 floats) to eliminate 2-way bank conflicts
    "s5_w4b_autotuned": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4b.matmul_s5_w4b", None),
    "s5_w4b_bm256_bn128_bk16_u16": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4b.matmul_s5_w4b_bm256_bn128_bk16_u16", None),
    # Stage 5 W4R: s5_w4 + register double-buffering of inner kk loop (hides smem load latency)
    "s5_w4r_autotuned": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4r.matmul_s5_w4r", None),
    "s5_w4r_bm256_bn128_bk16_u16": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4r.matmul_s5_w4r_bm256_bn128_bk16_u16", None),
    "s5_w4r_bm256_bn128_bk16_u8": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4r.matmul_s5_w4r_bm256_bn128_bk16_u8", None),
    # Stage 5 W4R2: s5_w4r with 128 threads (2×2 inter-warp); BM=BN=128 → 2 blocks/SM
    "s5_w4r2_autotuned": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4r2.matmul_s5_w4r2", None),
    "s5_w4r2_bm128_bn128_bk16_u8": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4r2.matmul_s5_w4r2_bm128_bn128_bk16_u8", None),
    "s5_w4r2_bm128_bn128_bk16_u16": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4r2.matmul_s5_w4r2_bm128_bn128_bk16_u16", None),
    # Stage 6: unified NUM_WARPS template (4 or 8); extends search to BM/BN=32
    "s6_autotuned": ("mymatmul.gpu.cuda_core.matmul_cuda_s6.matmul_s6",       None),
    "s6_lb":        ("mymatmul.gpu.cuda_core.matmul_cuda_s6_lb.matmul_s6_lb", None),
    # Stage 7: s6 with M/N/K baked as compile-time constants (JIT per shape)
    "s7_autotuned": ("mymatmul.gpu.cuda_core.matmul_cuda_s7.matmul_s7", None),
    # Stage 7 swz: s7 + CTA swizzle-by-2 (1D grid, A-tile L2 reuse)
    "s7_swz_autotuned": ("mymatmul.gpu.cuda_core.matmul_cuda_s7_swz.matmul_s7_swz", None),
    # Stage 7 swz4: s7 + CTA swizzle-by-4
    "s7_swz4_autotuned": ("mymatmul.gpu.cuda_core.matmul_cuda_s7_swz4.matmul_s7_swz4", None),
    # Stage 7 lw2: s7 with 2×16 intra-warp layout (float2 B loads, fewer B bank conflicts)
    "s7_lw2_autotuned": ("mymatmul.gpu.cuda_core.matmul_cuda_s7_lw2.matmul_s7_lw2", None),
    # Stage 5 W4R2S: s5_w4r2 + GROUP_M=2 block swizzle (1D grid, B-tile L2 reuse)
    "s5_w4r2s_autotuned": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4r2s.matmul_s5_w4r2s", None),
    "s5_w4r2s_bm128_bn128_bk16_u8": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4r2s.matmul_s5_w4r2s_bm128_bn128_bk16_u8", None),
    "s5_w4r2s_bm128_bn128_bk16_u16": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4r2s.matmul_s5_w4r2s_bm128_bn128_bk16_u16", None),
    # Stage 5 W4P: s5_w4 with 4-buffer paired loading (halved barrier stalls)
    "s5_w4p_autotuned": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4p.matmul_s5_w4p", None),
    "s5_w4p_bm256_bn128_u16": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_w4p.matmul_s5_w4p_bm256_bn128_u16", None),
    # Stage 5 SWZ: s5 BK=32 with A-tile swizzle (eliminates 2-way bank conflict)
    "s5_swz_autotuned": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_swz.matmul_s5_swz", None),
    "s5_swz_bm256_bn128_u16": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_swz.matmul_s5_swz_bm256_bn128_u16", None),
    # Stage 5 PTX: s5 with raw PTX cp.async.cg.shared.global.L2::128B
    "s5_ptx_autotuned": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_ptx.matmul_s5_ptx", None),
    "s5_ptx_bm256_bn128_bk32_u16": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_ptx.matmul_s5_ptx_bm256_bn128_bk32_u16", None),
    # Triton BF16 tensor-core (autotuned, same config space as tc2b)
    "triton_bf16_autotuned": ("mymatmul.gpu.matmul_triton.triton_bf16_autotuned", None, torch.bfloat16),
    # cuBLAS BF16 reference (tensor cores, bfloat16 inputs/output)
    "cublas_bf16": ("mymatmul.gpu.matmul_torch.matmul_torch_bf16", None, torch.bfloat16),
    # TC3: TC2b + tunable NUM_STAGES smem pipeline (NS=2 == TC2b)
    "tc3": ("mymatmul.gpu.tensor_core.matmul_cuda_tc3.matmul_tc3", None, torch.bfloat16),
    # TC4: TC2b with ldmatrix.x4.trans for B (pairs two N-tiles, halves B ldmatrix count)
    "tc4": ("mymatmul.gpu.tensor_core.matmul_cuda_tc4.matmul_tc4", None, torch.bfloat16),
    # TC5: TC2b with vectorized __nv_bfloat162 write-back (halves epilogue store count)
    "tc5": ("mymatmul.gpu.tensor_core.matmul_cuda_tc5.matmul_tc5", None, torch.bfloat16),
    # TC5rp: TC5 + register prefetch (double-buffered inner kk loop)
    "tc5rp": ("mymatmul.gpu.tensor_core.matmul_cuda_tc5rp.matmul_tc5rp", None, torch.bfloat16),
    # TC5jit: TC5 with M/K/N baked in as JIT compile-time constants
    "tc5jit":    ("mymatmul.gpu.tensor_core.matmul_cuda_tc5jit.matmul_tc5jit",       None, torch.bfloat16),
    "tc5jit_lb": ("mymatmul.gpu.tensor_core.matmul_cuda_tc5jit_lb.matmul_tc5jit_lb", None, torch.bfloat16),
    # TC5swz: TC5 + GROUP_M CTA swizzle (SW=1..8) for L2 B-tile reuse
    "tc5swz":    ("mymatmul.gpu.tensor_core.matmul_cuda_tc5swz.matmul_tc5swz",       None, torch.bfloat16),
    "tc5swz_lb": ("mymatmul.gpu.tensor_core.matmul_cuda_tc5swz_lb.matmul_tc5swz_lb", None, torch.bfloat16),
    # TC5l2: TC5 with cp.async.cg.L2::128B prefetch hint on tile loads
    "tc5l2":     ("mymatmul.gpu.tensor_core.matmul_cuda_tc5l2.matmul_tc5l2",         None, torch.bfloat16),
    # TC5_reg: TC5 autotuned over tile shape AND maxrregcount (128/168/255)
    "tc5_reg":     ("mymatmul.gpu.tensor_core.matmul_cuda_tc5_reg.matmul_tc5_reg",         None, torch.bfloat16),
    "tc5_regpruned": ("mymatmul.gpu.tensor_core.matmul_cuda_tc5_regpruned.matmul_tc5_regpruned", None, torch.bfloat16),
    # Hopper H2 Stage 1: TMA + mbarrier loads; mma.sync compute; no SMEM swizzle
    "h2_s1": ("mymatmul.gpu.hopper.matmul_h2_s1.matmul_h2_s1", None, torch.bfloat16),
    # Hopper H2 Stage 2: TMA (B 128B swizzle) + wgmma m64nBNk16; BM=64 fixed
    "h2_s2": ("mymatmul.gpu.hopper.matmul_h2_s2.matmul_h2_s2", None, torch.bfloat16),
    # Hopper H2 Stage 3: Stage 2 + multi-warpgroup (BM = NUM_WG * 64, NUM_WG ∈ {1,2})
    "h2_s3": ("mymatmul.gpu.hopper.matmul_h2_s3.matmul_h2_s3", None, torch.bfloat16),
    # Hopper H1 multi-stage: tc5_lb + NUM_STAGES ∈ {2,3,4,5} pipeline depth
    "h1_ms": ("mymatmul.gpu.hopper.matmul_h1_ms.matmul_h1_ms", None, torch.bfloat16),
    # Hopper H3: h1_ms pipeline + wgmma (B128 swizzle, A in regs, B from SMEM)
    "h3": ("mymatmul.gpu.hopper.matmul_h3.matmul_h3", None, torch.bfloat16),
    # Hopper H2-S4: h2_s3 + M-loop per warpgroup (BM up to 256, M_ITERS up to 4)
    "h2_s4": ("mymatmul.gpu.hopper.matmul_h2_s4.matmul_h2_s4", None, torch.bfloat16),
    "h2s4":  ("mymatmul.gpu.hopper.matmul_h2_s4.matmul_h2_s4", None, torch.bfloat16),
    # Hopper H2-S5: h2_s4 + tunable NUM_STAGES ∈ {2,3,4}
    "h2_s5": ("mymatmul.gpu.hopper.matmul_h2_s5.matmul_h2_s5", None, torch.bfloat16),
    # Hopper H2-S6: cp.async + wgmma SS mode (both A and B from SMEM descriptors)
    "h2_s6": ("mymatmul.gpu.hopper.matmul_h2_s6.matmul_h2_s6", None, torch.bfloat16),
    # Hopper H2-S7: h2_s6 + wgmma.wait_group 1 (overlap wgmma with next tile load)
    "h2_s7": ("mymatmul.gpu.hopper.matmul_h2_s7.matmul_h2_s7", None, torch.bfloat16),
    # Triton PTX: pre-compiled BM=128,BN=256,BK=32,NS=4 cp.async+wgmma-SS kernel
    "triton_ptx": ("mymatmul.gpu.hopper.matmul_triton_ptx.matmul_triton_ptx", None, torch.bfloat16),
    "tc6_x4b":       ("mymatmul.gpu.tensor_core.matmul_cuda_tc6_x4b.matmul_tc6_x4b",             None, torch.bfloat16),
    "tc8_4096":     ("mymatmul.gpu.tensor_core.matmul_cuda_tc8_4096.matmul_tc8_4096",             None, torch.bfloat16),
    "tc8_4096_ptx": ("mymatmul.gpu.tensor_core.matmul_cuda_tc8_4096_ptx.matmul_tc8_4096_ptx",   None, torch.bfloat16),
    "tc8_gemini":   ("mymatmul.gpu.tensor_core.matmul_cuda_tc8_gemini.matmul_tc8_gemini",           None, torch.bfloat16),
    "tc8g":         ("mymatmul.gpu.tensor_core.matmul_cuda_tc8g.matmul_tc8g",                       None, torch.bfloat16),
    # TC6: TC5 with kk loop split into 3 passes (ldmatrix-A, ldmatrix-B, MMA)
    "tc6":    ("mymatmul.gpu.tensor_core.matmul_cuda_tc6.matmul_tc6",       None, torch.bfloat16),
    "tc6_lb": ("mymatmul.gpu.tensor_core.matmul_cuda_tc6_lb.matmul_tc6_lb", None, torch.bfloat16),
    # TC7: TC6 with A-tile and B-tile async copies committed separately
    "tc7":    ("mymatmul.gpu.tensor_core.matmul_cuda_tc7.matmul_tc7",       None, torch.bfloat16),
    "tc7_lb": ("mymatmul.gpu.tensor_core.matmul_cuda_tc7_lb.matmul_tc7_lb", None, torch.bfloat16),
    # Blog series article 1: 128-thread, 64×256 tile, BK=16, no pipelining
    "blog1": ("mymatmul.gpu.cuda_core.matmul_cuda_blog1.matmul_blog1", None),
    # Blog series article 3: 128-thread, 128×128 tile, cp.async + register pipelining (N=4096 only)
    "blog3": ("mymatmul.gpu.cuda_core.matmul_cuda_blog3.matmul_blog3", None),
    # Blog3 variant: 256-thread, 256×128 tile, TI=16 (N=4096 only)
    "blog3b": ("mymatmul.gpu.cuda_core.matmul_cuda_blog3b.matmul_blog3b", None),
    # Stage 5 + L2 grouped block ordering: auto-tuned over BM/BN/BK/UNROLL/GROUP_M
    "s5_l2_autotuned": ("mymatmul.gpu.cuda_core.matmul_cuda_s5_l2.matmul_s5_l2", None),
    # Stage 4 Strided+Padded: s4st + A_shared[BM][BK+1] padding → zero A conflicts (educational)
    **{f"s4stp_tm8_tn8_bm64_bn64_bk16_u{u}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s4stp.matmul_s4stp_tm8_tn8_bm64_bn64_bk16_u{u}", None)
       for u in [1, 4, 8, 16]},
    # Stage 3 + warp tiling
    **{f"s3w_{k}": (f"mymatmul.gpu.cuda_core.matmul_cuda_s3_warp.matmul_s3_warp_{k}", None)
       for k in [
           "tm8_tn8_bm128_bn128_bk32_wm64_wn32_u8",
           "tm8_tn8_bm128_bn128_bk32_wm32_wn64_u8",
           "tm8_tn8_bm128_bn64_bk32_wm64_wn32_u8",
           "tm8_tn8_bm128_bn64_bk32_wm32_wn64_u8",
           "tm8_tn4_bm64_bn64_bk32_wm32_wn32_u8",
           "tm4_tn4_bm64_bn64_bk32_wm32_wn16_u8",
           "tm4_tn4_bm32_bn64_bk32_wm16_wn32_u8",
       ]},
}

SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]

# ---------------------------------------------------------------------------
# Auto-discovery: any mymatmul/gpu/{cuda_core,tensor_core}/matmul_cuda_{X}.py
# that exposes matmul_{X}() is auto-registered as impl "X".
# Optional module constants: DTYPE (default float32), MAX_SIZE (default None).
# ---------------------------------------------------------------------------

_GPU_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "mymatmul", "gpu")
_DISCOVERED: dict | None = None


def _auto_discover_impls() -> dict:
    import glob, importlib
    result = {}
    for subdir, pkg in [("cuda_core",   "mymatmul.gpu.cuda_core"),
                        ("tensor_core", "mymatmul.gpu.tensor_core")]:
        for path in sorted(glob.glob(os.path.join(_GPU_DIR, subdir, "matmul_cuda_*.py"))):
            base = os.path.basename(path)[len("matmul_cuda_"):-3]
            dotpath = f"{pkg}.matmul_cuda_{base}.matmul_{base}"
            try:
                mod = importlib.import_module(f"{pkg}.matmul_cuda_{base}")
                if not hasattr(mod, f"matmul_{base}"):
                    continue
                dtype    = getattr(mod, "DTYPE",    torch.float32)
                max_size = getattr(mod, "MAX_SIZE", None)
            except Exception:
                continue
            result[base] = (dotpath, max_size, dtype)
    return result


def _all_impls() -> dict:
    """Merge auto-discovered + explicit IMPLEMENTATIONS (explicit takes precedence)."""
    global _DISCOVERED
    if _DISCOVERED is None:
        _DISCOVERED = _auto_discover_impls()
    merged = dict(_DISCOVERED)
    for k, v in IMPLEMENTATIONS.items():
        # normalise to 3-tuple (dotpath, max_size, dtype)
        merged[k] = (*v, torch.float32) if len(v) == 2 else v
    return merged


def get_impl_dtype(name: str) -> torch.dtype:
    entry = _all_impls().get(name) or _find_prefix_entry(name)
    return entry[2] if entry and len(entry) > 2 else torch.float32


def _find_prefix_entry(name: str):
    """Return the _all_impls() entry for the module prefix (first word before '_')."""
    return _all_impls().get(name.split("_")[0])
WARMUP_RUNS = 3
TIMED_RUNS = 10

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results_gpu.csv")
FIELDNAMES = ["timestamp", "impl", "M", "N", "K", "gflops", "ms_mean", "ms_min"]


def gflops(M, N, K, seconds):
    return 2 * M * N * K / seconds / 1e9


def flush_l2_cache():
    """Flush GPU L2 cache by allocating and accessing a large buffer.

    RTX 4090 has 72MB L2 cache. We allocate 256MB (> L2 size) and read/write
    to force eviction of L2 contents before timed measurements.
    """
    cache_flush_size = 256 * 1024 * 1024  # 256MB in bytes
    # Each float32 is 4 bytes, so we need this many elements
    flush_buffer = torch.zeros(cache_flush_size // 4, dtype=torch.float32, device='cuda')

    # Read and write to the buffer to force L2 eviction
    flush_buffer[:] = flush_buffer[:] + 1.0

    # Synchronize to ensure flush is complete
    torch.cuda.synchronize()

    # Free the buffer
    del flush_buffer
    torch.cuda.synchronize()


def benchmark_fn(fn, A_gpu, B_gpu):
    # Warmup with GPU synchronization
    for _ in range(WARMUP_RUNS):
        fn(A_gpu, B_gpu)
        torch.cuda.synchronize()

    # Timed runs with synchronization
    times = []
    for _ in range(TIMED_RUNS):
        # Flush L2 cache before each timed run for clean measurements
        flush_l2_cache()

        t0 = time.perf_counter()
        fn(A_gpu, B_gpu)
        torch.cuda.synchronize()  # Wait for GPU to finish before stopping timer
        times.append(time.perf_counter() - t0)
    return times


def load_fn(name_or_dotpath: str):
    """Load a matmul function by impl name or dotpath.

    Also accepts specific kernel config names like 'tc4_bm128_bn64_bk64_nw4':
    strips the known impl prefix to find the module, then builds a direct
    launch_matmul call for that kernel without autotuning.
    """
    import importlib, inspect
    if '.' in name_or_dotpath:
        module_path, fn_name = name_or_dotpath.rsplit(".", 1)
        return getattr(importlib.import_module(module_path), fn_name)

    entry = _all_impls().get(name_or_dotpath)
    if entry is not None:
        module_path, fn_name = entry[0].rsplit(".", 1)
        return getattr(importlib.import_module(module_path), fn_name)

    # Specific kernel config: find impl prefix, load module, build direct launcher.
    prefix_entry = _find_prefix_entry(name_or_dotpath)
    if prefix_entry is None:
        raise KeyError(f"Unknown impl: '{name_or_dotpath}'")

    module_path = prefix_entry[0].rsplit(".", 1)[0]
    mod = importlib.import_module(module_path)
    kname = f"matmul_cuda_{name_or_dotpath}"
    cfg = next((c for c in mod._CONFIGS if mod._kname(*c) == kname), None)
    if cfg is None:
        raise KeyError(f"Kernel '{kname}' not found in {module_path}._CONFIGS")

    block = mod._block(cfg[3]) if len(inspect.signature(mod._block).parameters) > 0 else mod._block()
    smem_nparams = len(inspect.signature(mod._smem).parameters)
    smem = mod._smem(*(cfg[:3] + cfg[4:])[:smem_nparams])
    bm, bn = cfg[0], cfg[1]

    def _direct(A, B):
        from mymatmul.gpu._pycuda_loader import launch_matmul, get_module
        get_module(mod._EXT)
        return launch_matmul(mod._EXT, kname, A, B, block,
                             mod._grid(A.shape[0], B.shape[1], bm, bn),
                             smem_bytes=smem)
    _direct.__name__ = kname
    return _direct


def validate_fn(fn, A_gpu, B_gpu, rtol=None, atol=None):
    if A_gpu.dtype == torch.bfloat16:
        # Use float32 reference — cuBLAS BF16 switches to BF16 accumulators at large K,
        # making it an unreliable reference for kernels that keep float32 accumulators.
        # BF16 input quantization errors accumulate as ~sqrt(K)*u (u=2^-8=1/256, unit roundoff)
        # with expected max over N^2 outputs ~= sqrt(K)/32. Scale atol with sqrt(K).
        expected = torch.mm(A_gpu.float(), B_gpu.float())
        if rtol is None: rtol = 1e-2
        if atol is None: atol = max(1.0, A_gpu.shape[1] ** 0.5 / 32)
    else:
        expected = torch.matmul(A_gpu.float(), B_gpu.float())
        if rtol is None: rtol = 1e-2
        if atol is None: atol = 1e-1
    result = fn(A_gpu, B_gpu).float()

    diff = (result - expected).abs()
    rel = diff / expected.abs().clamp_min(1e-3)

    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    sample = diff.flatten()
    if sample.numel() > 10_000_000:
        step = sample.numel() // 10_000_000
        sample = sample[::step][:10_000_000]
    p99_abs = sample.quantile(0.99).item()
    max_rel = rel.max().item()
    mean_rel = rel.mean().item()

    if not torch.allclose(result, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"Mismatch: max_abs={max_abs:.3e}, p99_abs={p99_abs:.3e}, "
            f"mean_abs={mean_abs:.3e}, max_rel={max_rel:.3e}, mean_rel={mean_rel:.3e}"
        )
    return True


def run(impl_names, sizes):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    all_i = _all_impls()

    for name in impl_names:
        entry    = all_i.get(name, (None, None, torch.float32))
        max_size = entry[1]
        dtype    = entry[2] if len(entry) > 2 else torch.float32
        fn       = load_fn(name)
        print(f"\n[{name}]")

        for sz in sizes:
            M = N = K = sz

            if max_size is not None and sz > max_size:
                print(f"  {M}x{N}x{K}: skipped (max_size={max_size})")
                continue

            A_gpu = torch.randn(M, K, dtype=dtype, device='cuda')
            B_gpu = torch.randn(K, N, dtype=dtype, device='cuda')

            # Validate result before benchmarking
            try:
                validate_fn(fn, A_gpu, B_gpu)
            except AssertionError as e:
                print(f"  {M}x{N}x{K}: ✗ validation FAILED: {e}")
                continue

            # Time only the GPU computation
            times = benchmark_fn(fn, A_gpu, B_gpu)
            ms_mean = np.mean(times) * 1e3
            ms_min = np.min(times) * 1e3
            gf = gflops(M, N, K, np.min(times))

            print(f"  {M}x{N}x{K}: ✓ {gf:.2f} GFLOPS  (mean {ms_mean:.1f} ms, best {ms_min:.1f} ms)")

            rows.append({
                "timestamp": timestamp,
                "impl": name,
                "M": M, "N": N, "K": K,
                "gflops": f"{gf:.4f}",
                "ms_mean": f"{ms_mean:.3f}",
                "ms_min": f"{ms_min:.3f}",
            })

    # Merge with existing results: new rows overwrite matching (impl, M, N, K) keys
    existing = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, newline="") as f:
            for row in csv.DictReader(f):
                existing[(row["impl"], row["M"], row["N"], row["K"])] = row
    for row in rows:
        existing[(row["impl"], row["M"], row["N"], row["K"])] = row

    merged = sorted(existing.values(), key=lambda r: (r["impl"], int(r["M"])))
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(merged)

    print(f"\nResults written to {RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impls", nargs="+", default=list(_all_impls().keys()),
                        help="Which implementations to benchmark (default: all)")
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES,
                        help="Matrix sizes to benchmark (square MxMxM)")
    args = parser.parse_args()

    run(args.impls, args.sizes)


if __name__ == "__main__":
    main()
