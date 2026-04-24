"""Profile two kernels back-to-back for ncu comparison."""
import sys
import torch

M = N = K = 4096
A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')

name = sys.argv[1] if len(sys.argv) > 1 else "bm128"

if name == "bm128":
    from mymatmul.gpu.matmul_cuda_s4 import matmul_s4_tm8_tn8_bm128_bn128_bk16_u16 as fn
else:
    from mymatmul.gpu.matmul_cuda_s4 import matmul_s4_tm8_tn8_bm64_bn64_bk16_u16 as fn

for _ in range(3):
    fn(A, B)
torch.cuda.synchronize()

fn(A, B)
torch.cuda.synchronize()
