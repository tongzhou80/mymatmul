"""Profile cuBLAS fp32 (TF32 disabled) for NCU."""
import sys
import torch

torch.backends.cuda.matmul.allow_tf32 = False

sz = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
A = torch.randn(sz, sz, dtype=torch.float32, device='cuda')
B = torch.randn(sz, sz, dtype=torch.float32, device='cuda')

for _ in range(3):
    torch.matmul(A, B)
torch.cuda.synchronize()

torch.matmul(A, B)
torch.cuda.synchronize()
