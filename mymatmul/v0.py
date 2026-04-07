import numpy as np
import numba


@numba.njit
def matmul_v0(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Naive triple-loop matrix multiplication (baseline).

    C[i, j] = sum_k A[i, k] * B[k, j]

    JIT-compiled via numba, but no tiling, no vectorization, no parallelism.
    """
    M, K = A.shape
    N = B.shape[1]

    C = np.zeros((M, N), dtype=A.dtype)
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
    return C
