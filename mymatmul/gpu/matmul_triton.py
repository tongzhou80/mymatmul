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
_BF16_BKS      = [32, 64]  # BK=16 dropped — never autotunes as best
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


# ============================================================================
# Blackwell-standard Triton matmul
#
# Mirrors Triton's tutorials/09-persistent-matmul.py `matmul_kernel_tma_persistent`:
#   - TMA-backed loads via `tl.make_tensor_descriptor` (auto-128B-swizzled)
#   - persistent grid: each CTA processes multiple output tiles in a flat loop
#   - warp_specialize=True enables Blackwell's async warp scheduler
#   - tl.dot(a, b.T, acc) — same B-as-(N,K) layout convention as gau-nernst v2
#
# Reference autotune space (from tutorials/09 lines 369-382, March 2026):
#   BM=128, BN∈{128,256}, BK∈{64,128}, num_stages∈{2,3,4}, num_warps∈{4,8}
# ============================================================================

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    _TENSOR_DESCRIPTOR_AVAILABLE = True
except ImportError:
    _TENSOR_DESCRIPTOR_AVAILABLE = False


_bw_configs = [
    triton.Config(
        {'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, 'BLOCK_SIZE_K': BK,
         'GROUP_SIZE_M': 8, 'WARP_SPECIALIZE': ws, 'EPILOGUE_SUBTILE': epi,
         'FLATTEN': fl},
        num_stages=s, num_warps=w,
    )
    for BM in [128]
    for BN in [128, 256]
    for BK in [64, 128]
    for s in [2, 3, 4]
    for w in [4, 8]
    for ws in [True, False]
    for epi in [True, False]
    for fl in [True, False]
    # EPI=True with FLATTEN=False is invalid per tutorial's prune_invalid_configs.
    if not (epi and not fl)
]


@triton.jit
def _bw_compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(configs=_bw_configs, key=["M", "N", "K"])
@triton.jit
def _matmul_bf16_bw_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Device-side descriptor creation (Triton-tutorial matmul_kernel_descriptor_persistent).
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N // 2 if EPILOGUE_SUBTILE else BLOCK_SIZE_N])

    # Decouple epilogue tile_id from prologue tile_id so they can pipeline.
    tile_id_c = start_pid - NUM_SMS

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS,
                            flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _bw_compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _bw_compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c_desc.store([offs_cm, offs_cn], acc0.to(dtype))
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], acc1.to(dtype))
        else:
            c_desc.store([offs_cm, offs_cn], accumulator.to(dtype))


_BW_ALLOCATOR_SET = False


def _bw_register_allocator():
    """Triton needs a global memory allocator for TMA descriptors; register once."""
    global _BW_ALLOCATOR_SET
    if _BW_ALLOCATOR_SET:
        return
    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    _BW_ALLOCATOR_SET = True


def triton_bf16_blackwell(A, B):
    """Blackwell-standard Triton matmul (device-side TMA descriptors).

    Requires B as (N, K) row-major in memory.  WARP_SPECIALIZE + EPILOGUE_SUBTILE
    are both autotuned per shape.
    """
    _bw_register_allocator()

    M, K = A.shape
    N_, K_ = B.shape
    assert K == K_, f"K mismatch: A is (M,K)={A.shape}, B is (N,K)={B.shape}"
    N = N_
    C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(META):
        BM = META["BLOCK_SIZE_M"]
        BN = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)),)

    _matmul_bf16_bw_kernel[grid](A, B, C, M, N, K, NUM_SMS=NUM_SMS)
    return C


def triton_bf16_blackwell_pytorch(A, B):
    """PyTorch-convention wrapper: takes A (M,K), B (K,N) and pre-transposes B."""
    return triton_bf16_blackwell(A, B.t().contiguous())
