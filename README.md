# matmul

A progressive study of matrix multiplication optimizations, from naive CPU to cuBLAS-level GPU performance.

## Goal

Each version introduces one or more optimizations on top of the previous, making it easy to isolate the performance impact of each technique.

## Implementations

| Import | Description |
|--------|-------------|
| `matmul_v0` | Naive triple-loop (baseline) |

More versions to come: cache blocking, SIMD, multithreading, CUDA naive, shared memory tiling, register blocking, tensor cores, cuBLAS.

## Install

```bash
pip install -e .
```

## Usage

```python
from mymatmul import matmul_v0

import numpy as np
A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)
C = matmul_v0(A, B)
```
