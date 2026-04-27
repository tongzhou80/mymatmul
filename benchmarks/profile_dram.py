"""Profile DRAM bandwidth for a single impl. Usage: python profile_dram.py <impl_name> [size]"""
import sys
import torch
from bench_gpu import IMPLEMENTATIONS, load_fn

impl_name = sys.argv[1]
sz = int(sys.argv[2]) if len(sys.argv) > 2 else 4096

dotpath, _ = IMPLEMENTATIONS[impl_name]
fn = load_fn(dotpath)

M = N = K = sz
A = torch.randn(M, K, dtype=torch.float32, device='cuda')
B = torch.randn(K, N, dtype=torch.float32, device='cuda')

for _ in range(3):
    fn(A, B)
torch.cuda.synchronize()

fn(A, B)
torch.cuda.synchronize()
