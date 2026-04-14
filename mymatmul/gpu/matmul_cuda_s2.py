"""
Stage 2: Shared Memory Tiling — Python wrappers
================================================
Exposes two kernels for studying the occupancy vs arithmetic-intensity tradeoff.

Both kernels use bfloat16 input/output with float32 accumulation, BK=32, TM=2, TN=1.

Case 1  BM=8,  BN=32, 32x4 threads:  global AI = 6.4,  smem AI = 0.67
Case 2  BM=16, BN=32, 32x8 threads:  global AI = 10.67, smem AI = 0.67

Register and shared-memory usage is printed to stderr at build time by -Xptxas -v.
"""

import torch


def matmul_s2_bm8_bn32(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Case 1: BM=8, BN=32, BK=32, thread block 32x4 (128 threads).

    TM=2, TN=1: each thread computes 2 output rows and 1 output column.

    Arithmetic intensity:
      Global memory: BM*BN / (BM+BN) = 8*32 / (8+32) = 6.4
      Shared memory: TM*TN / (TM+TN) = 2*1  / (2+1)  = 0.67

    Shared memory footprint per block:
      A_shared[8][32]  + B_shared[32][32] = (256 + 1024) * 4B = 5120 B

    Args:
        A: bfloat16 CUDA tensor, shape (M, K)
        B: bfloat16 CUDA tensor, shape (K, N)

    Returns:
        C: bfloat16 CUDA tensor, shape (M, N)
    """
    from . import _matmul_cuda_ext_s2
    return _matmul_cuda_ext_s2.matmul_s2_bm8_bn32_bk32_threads32x4(A, B)


def matmul_s2_bm16_bn32(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Case 2: BM=16, BN=32, BK=32, thread block 32x8 (256 threads).

    TM=2, TN=1: each thread computes 2 output rows and 1 output column.

    Arithmetic intensity:
      Global memory: BM*BN / (BM+BN) = 16*32 / (16+32) = 10.67  (vs 6.4 in case 1)
      Shared memory: TM*TN / (TM+TN) = 2*1   / (2+1)   = 0.67   (same as case 1)

    Shared memory footprint per block:
      A_shared[16][32] + B_shared[32][32] = (512 + 1024) * 4B = 6144 B

    Doubling BM increases the global memory arithmetic intensity by ~67% with only
    a 20% increase in shared memory — but the block now has 256 threads (8 warps),
    which may reduce the number of concurrent blocks per SM.

    Args:
        A: bfloat16 CUDA tensor, shape (M, K)
        B: bfloat16 CUDA tensor, shape (K, N)

    Returns:
        C: bfloat16 CUDA tensor, shape (M, N)
    """
    from . import _matmul_cuda_ext_s2
    return _matmul_cuda_ext_s2.matmul_s2_bm16_bn32_bk32_threads32x8(A, B)
