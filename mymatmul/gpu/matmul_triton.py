"""Triton matmul reference implementations for performance comparison."""

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(a_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _make_triton_matmul(block_m, block_n, block_k, group_m=8, allow_tf32=False):
    """Return a launcher function for the given Triton block sizes (FP32 SIMT, no TF32)."""
    def fn(A, B):
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        _matmul_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP_M=group_m,
            ALLOW_TF32=allow_tf32,
        )
        return C
    fn.__name__ = f"triton_matmul_bm{block_m}_bn{block_n}_bk{block_k}"
    return fn


def _make_triton_fp32_simt(block_m, block_n, block_k, group_m=8):
    """Triton FP32 SIMT matmul — no tensor cores, no TF32. Comparable to our s4 CUDA kernels."""
    fn = _make_triton_matmul(block_m, block_n, block_k, group_m)
    fn.__name__ = f"triton_fp32simt_bm{block_m}_bn{block_n}_bk{block_k}"
    return fn


# ---------------------------------------------------------------------------
# Autotuned FP32 SIMT — sweeps block sizes and pipeline stages.
# Pruned from empirical results: num_warps=8 wins every size; num_stages∈{2} never wins.
# 32 configs (down from 96), ~3× faster autotuning with same quality.
# ---------------------------------------------------------------------------

_autotune_configs = [
    triton.Config(
        {'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk, 'GROUP_M': 8},
        num_stages=ns, num_warps=8,
    )
    for bm in [64, 128, 256]
    for bn in [64, 128, 256]
    for bk in [16, 32]
    for ns in [3, 4]
    if not (bm == 256 and bn == 256)   # register spill: acc[256][256]/nthreads > 255 regs
]


@triton.autotune(configs=_autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _matmul_kernel_autotuned(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(a_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_fp32simt_autotuned(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    _matmul_kernel_autotuned[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        ALLOW_TF32=False,
    )
    return C


# Best autotuned config: BM=128, BN=128, BK=32, num_warps=8, num_stages=4
def triton_fp32simt_bm128_bn128_bk32_w8_s4(A, B):
    M, K = A.shape; _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
    _matmul_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, ALLOW_TF32=False,
        num_warps=8, num_stages=4,
    )
    return C


# FP32 SIMT configs (comparable to our s4 CUDA kernels)
triton_fp32simt_bm128_bn128_bk16 = _make_triton_fp32_simt(128, 128, 16)


# ---------------------------------------------------------------------------
# BF16 tensor-core kernel
# tl.dot with bf16 inputs maps to mma.sync on sm_89 (Ada Lovelace).
# Accumulator stays f32; output converted back to bf16.
# ---------------------------------------------------------------------------

_BF16_BMS      = [64, 128, 256]
_BF16_BNS      = [64, 128, 256]
_BF16_BKS      = [16, 32, 64]
_BF16_NWS      = [4, 8]
_BF16_STAGES   = [3, 4, 5]
_BF16_MAX_SMEM = 100352


def _bf16_smem(bm, bn, bk, ns):
    # Triton uses ns smem stages, each stage holds one A-tile + one B-tile (bf16 = 2 bytes)
    return ns * (bm * bk + bk * bn) * 2


_bf16_configs = [
    triton.Config(
        {'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk, 'GROUP_M': 8},
        num_stages=ns, num_warps=nw,
    )
    for bm in _BF16_BMS
    for bn in _BF16_BNS
    for bk in _BF16_BKS
    for nw in _BF16_NWS
    for ns in _BF16_STAGES
    if _bf16_smem(bm, bn, bk, ns) <= _BF16_MAX_SMEM
    and bm * bn <= 4096 * nw
]


@triton.autotune(configs=_bf16_configs, key=['M', 'N', 'K'])
@triton.jit
def _matmul_bf16_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    a_m_mask = offs_m < M
    b_n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=a_m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & b_n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_bf16_autotuned(A, B):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    _matmul_bf16_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C
triton_fp32simt_bm128_bn64_bk16  = _make_triton_fp32_simt(128,  64, 16)
triton_fp32simt_bm64_bn64_bk16   = _make_triton_fp32_simt(64,   64, 16)
triton_fp32simt_bm128_bn128_bk32 = _make_triton_fp32_simt(128, 128, 32)
triton_fp32simt_bm128_bn64_bk32  = _make_triton_fp32_simt(128,  64, 32)
triton_fp32simt_bm64_bn64_bk32   = _make_triton_fp32_simt(64,   64, 32)
# BN=256 configs; BK=16 fits in 48KB smem with double buffering (2×(128×16+16×256)×4=49152B)
triton_fp32simt_bm128_bn256_bk16 = _make_triton_fp32_simt(128, 256, 16)
