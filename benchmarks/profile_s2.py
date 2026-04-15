"""Profile Stage 2 kernels with ncu for runtime occupancy."""
import torch
from mymatmul.gpu.matmul_cuda_s2 import matmul_s2_bm8_bn32, matmul_s2_bm16_bn32

M, K, N = 4096, 4096, 4096
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

# Warmup
for _ in range(3):
    matmul_s2_bm8_bn32(A, B)
    matmul_s2_bm16_bn32(A, B)
torch.cuda.synchronize()

# Single timed calls for ncu to capture
matmul_s2_bm8_bn32(A, B)
torch.cuda.synchronize()

matmul_s2_bm16_bn32(A, B)
torch.cuda.synchronize()
