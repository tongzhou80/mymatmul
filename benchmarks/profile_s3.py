"""Profile best Stage 3 kernel (tm8_tn8_bm128_bn128_bk32_u8) with ncu."""
import torch
from mymatmul.gpu.matmul_cuda_s3 import matmul_s3_tm8_tn8_bm128_bn128_bk32_u8 as matmul_s3_best

M, K, N = 4096, 4096, 4096
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

# Warmup
for _ in range(3):
    matmul_s3_best(A, B)
torch.cuda.synchronize()

# Kernel calls for ncu to capture
matmul_s3_best(A, B)
torch.cuda.synchronize()
