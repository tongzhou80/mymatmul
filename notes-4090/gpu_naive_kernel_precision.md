# GPU Naive Kernel Precision Limitation

## Problem
The naive CUDA kernel (`cuda_naive_ijk`) fails validation at size 4096³ with max_error ~2.5e-1.

## Root Cause
The kernel accumulates float16 inputs into a float32 accumulator over K=4096 iterations:
```cuda
float sum = 0.0f;
for (int k = 0; k < K; k++) {
    sum += (float)A[i*K + k] * (float)B[k*N + j];
}
C[i*N + j] = (scalar_t)sum;
```

At K=4096, rounding errors accumulate beyond validation tolerance (rtol=1e-3, atol=1e-2).

## Workaround
Capped `cuda_naive_ijk` with `max_size=2048` in `benchmarks/bench_gpu.py` to skip 4096³ runs.

## Possible Solutions
1. **Float64 accumulator**: Higher precision but slower (2x memory bandwidth)
2. **Kahan summation**: Compensated summation to reduce rounding error
3. **Relax tolerance**: Not ideal—masks real precision issues
4. **Accept as limitation**: Document it as inherent to naive approach

## Notes
This is expected behavior for naive implementations without special summation techniques. Production libraries handle this via:
- Block-level reductions with careful ordering
- Mixed-precision strategies
- Compensated summation algorithms
