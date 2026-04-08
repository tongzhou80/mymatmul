import numpy as np
import numba


@numba.njit
def matmul_numba_ijk(A, B):
    """Naive triple-loop matmul JIT-compiled via Numba, loop order i-j-k.

    B is accessed column-wise (strided) — cache-unfriendly for large N.
    """
    M, K = A.shape
    N = B.shape[1]
    C = np.zeros((M, N), dtype=A.dtype)
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
    return C


@numba.njit
def matmul_numba_ikj(A, B):
    """Naive triple-loop matmul JIT-compiled via Numba, loop order i-k-j.

    Both A[i,k] and B[k,j] are accessed row-wise — cache-friendly.
    """
    M, K = A.shape
    N = B.shape[1]
    C = np.zeros((M, N), dtype=A.dtype)
    for i in range(M):
        for k in range(K):
            for j in range(N):
                C[i, j] += A[i, k] * B[k, j]
    return C
