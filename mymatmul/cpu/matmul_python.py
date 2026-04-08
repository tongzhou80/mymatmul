import numpy as np


def matmul_openblas(A, B):
    """NumPy's matmul using OpenBLAS backend."""
    return np.matmul(A, B)


def matmul_python_ijk(A, B):
    """Naive triple-loop matmul in pure Python, loop order i-j-k.

    B is accessed column-wise (strided) — cache-unfriendly for large N.
    Only suitable for small matrices (e.g. up to ~128x128).
    """
    M, K = A.shape
    N = B.shape[1]
    C = np.zeros((M, N), dtype=A.dtype)
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
    return C


def matmul_python_ikj(A, B):
    """Naive triple-loop matmul in pure Python, loop order i-k-j.

    Both A[i,k] and B[k,j] are accessed row-wise — cache-friendly.
    Only suitable for small matrices (e.g. up to ~128x128).
    """
    M, K = A.shape
    N = B.shape[1]
    C = np.zeros((M, N), dtype=A.dtype)
    for i in range(M):
        for k in range(K):
            for j in range(N):
                C[i, j] += A[i, k] * B[k, j]
    return C
