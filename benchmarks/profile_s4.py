"""Minimal script for ncu to profile the best Stage 4 kernel."""
import torch
from mymatmul.gpu.cuda_core.matmul_cuda_s4 import matmul_s4_tm8_tn8_bm128_bn128_bk16_u16 as fn

M = N = K = 4096
A = torch.randn(M, K, dtype=torch.float32, device='cuda')
B = torch.randn(K, N, dtype=torch.float32, device='cuda')

# Warmup
for _ in range(3):
    fn(A, B)
torch.cuda.synchronize()

# Profiled iteration
fn(A, B)
torch.cuda.synchronize()
