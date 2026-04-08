"""GPU implementations (PyTorch/CUDA)."""

from .matmul_torch import matmul_torch_naive_ijk

__all__ = ["matmul_torch_naive_ijk"]
