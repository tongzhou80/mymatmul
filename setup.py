# Install with: pip install -e . --no-build-isolation
#
# The --no-build-isolation flag is required because this package builds CUDA
# extensions via torch.utils.cpp_extension, which must match your system's
# CUDA version. pip's default isolated build environment installs a generic
# torch that may be compiled against a different CUDA version, causing a
# mismatch error at build time.
#
# Before installing, make sure you have the correct torch for your CUDA version:
#   https://pytorch.org/get-started/locally/
# e.g. for CUDA 12.1:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

from setuptools import setup, find_packages, Extension
import numpy as np
import os

# Target only the local GPU architecture to speed up compilation.
# Override by setting TORCH_CUDA_ARCH_LIST in the environment.
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')  # RTX 4090

# Check if CUDA is available for GPU extension
cuda_available = os.environ.get('CUDA_HOME') or os.path.exists('/usr/local/cuda')
ext_modules = []

cpp_ext = Extension(
    "mymatmul.cpu._matmul_cpp_ext",
    sources=["mymatmul/cpu/_matmul_cpp_ext.cpp"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-std=c++11", "-mavx2", "-mfma", "-fopenmp"],
    extra_link_args=["-fopenmp"],
    language="c++",
)
ext_modules.append(cpp_ext)

# Add CUDA extension if available
if cuda_available:
    try:
        from torch.utils.cpp_extension import CUDAExtension

        cuda_ext = CUDAExtension(
            "mymatmul.gpu.cuda_core._matmul_cuda_ext",
            sources=["mymatmul/gpu/cuda_core/_matmul_cuda_ext.cu"],
            extra_compile_args={
                'cxx': ["-O3"],
                'nvcc': ["-O3", "-std=c++17"],
            },
        )
        ext_modules.append(cuda_ext)

        cuda_ext_s2 = CUDAExtension(
            "mymatmul.gpu.cuda_core._matmul_cuda_ext_s2",
            sources=["mymatmul/gpu/cuda_core/_matmul_cuda_ext_s2.cu"],
            extra_compile_args={
                'cxx': ["-O3"],
                'nvcc': ["-O3", "-std=c++17", "-Xptxas", "-v"],
            },
        )
        ext_modules.append(cuda_ext_s2)

        cuda_ext_s3 = CUDAExtension(
            "mymatmul.gpu.cuda_core._matmul_cuda_ext_s3",
            sources=["mymatmul/gpu/cuda_core/_matmul_cuda_ext_s3.cu"],
            extra_compile_args={
                'cxx': ["-O3"],
                'nvcc': ["-O3", "-std=c++17", "-Xptxas", "-v"],
            },
        )
        ext_modules.append(cuda_ext_s3)

        cuda_ext_s3_warp = CUDAExtension(
            "mymatmul.gpu.cuda_core._matmul_cuda_ext_s3_warp",
            sources=["mymatmul/gpu/cuda_core/_matmul_cuda_ext_s3_warp.cu"],
            extra_compile_args={
                'cxx': ["-O3"],
                'nvcc': ["-O3", "-std=c++17", "-Xptxas", "-v"],
            },
        )
        ext_modules.append(cuda_ext_s3_warp)

        cuda_ext_s4 = CUDAExtension(
            "mymatmul.gpu.cuda_core._matmul_cuda_ext_s4",
            sources=["mymatmul/gpu/cuda_core/_matmul_cuda_ext_s4.cu"],
            extra_compile_args={
                'cxx': ["-O3"],
                'nvcc': ["-O3", "-std=c++17", "-Xptxas", "-v"],
            },
        )
        ext_modules.append(cuda_ext_s4)

        cuda_ext_s4b = CUDAExtension(
            "mymatmul.gpu.cuda_core._matmul_cuda_ext_s4b",
            sources=["mymatmul/gpu/cuda_core/_matmul_cuda_ext_s4b.cu"],
            extra_compile_args={
                'cxx': ["-O3"],
                'nvcc': ["-O3", "-std=c++17", "-Xptxas", "-v"],
            },
        )
        ext_modules.append(cuda_ext_s4b)

        cuda_ext_s4sw = CUDAExtension(
            "mymatmul.gpu.cuda_core._matmul_cuda_ext_s4sw",
            sources=["mymatmul/gpu/cuda_core/_matmul_cuda_ext_s4sw.cu"],
            extra_compile_args={
                'cxx': ["-O3"],
                'nvcc': ["-O3", "-std=c++17", "-Xptxas", "-v"],
            },
        )
        ext_modules.append(cuda_ext_s4sw)

        cuda_ext_s5 = CUDAExtension(
            "mymatmul.gpu.tensor_core._matmul_cuda_ext_s5",
            sources=["mymatmul/gpu/tensor_core/_matmul_cuda_ext_s5.cu"],
            extra_compile_args={
                'cxx': ["-O3"],
                'nvcc': ["-O3", "-std=c++17", "-Xptxas", "-v"],
            },
        )
        ext_modules.append(cuda_ext_s5)

    except ImportError:
        print("Warning: PyTorch CUDA extension not available, skipping GPU extension")

setup(
    name="mymatmul",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=[
        "numpy",
        "numba",
        "torch",
    ],
    cmdclass={
        'build_ext': __import__('torch.utils.cpp_extension', fromlist=['BuildExtension']).BuildExtension if cuda_available else None
    } if cuda_available else {},
)
