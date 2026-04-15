from setuptools import setup, find_packages, Extension
import numpy as np
import os

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
        # cuda_ext = CUDAExtension(
        #     "mymatmul.gpu._matmul_cuda_ext",
        #     sources=["mymatmul/gpu/_matmul_cuda_ext.cu"],
        #     extra_compile_args={
        #         'cxx': ["-O3"],
        #         'nvcc': ["-O3", "-std=c++17"],
        #     },
        # )
        # ext_modules.append(cuda_ext)

        cuda_ext_s3 = CUDAExtension(
            "mymatmul.gpu._matmul_cuda_ext_s3",
            sources=["mymatmul/gpu/_matmul_cuda_ext_s3.cu"],
            extra_compile_args={
                'cxx': ["-O3"],
                'nvcc': ["-O3", "-std=c++17", "-Xptxas", "-v"],
            },
        )
        ext_modules.append(cuda_ext_s3)

        cuda_ext_s4 = CUDAExtension(
            "mymatmul.gpu._matmul_cuda_ext_s4",
            sources=["mymatmul/gpu/_matmul_cuda_ext_s4.cu"],
            extra_compile_args={
                'cxx': ["-O3"],
                'nvcc': ["-O3", "-std=c++17", "-Xptxas", "-v"],
            },
        )
        ext_modules.append(cuda_ext_s4)

        # cuda_ext2 = CUDAExtension(
        #     "mymatmul.gpu._matmul_cuda_ext2",
        #     sources=["mymatmul/gpu/_matmul_cuda_ext2.cu"],
        #     extra_compile_args={
        #         'cxx': ["-O3"],
        #         'nvcc': ["-O3", "-std=c++17", "-Xptxas", "-v"],
        #     },
        # )
        # ext_modules.append(cuda_ext2)

        cuda_ext_s5 = CUDAExtension(
            "mymatmul.gpu._matmul_cuda_ext_s5",
            sources=["mymatmul/gpu/_matmul_cuda_ext_s5.cu"],
            extra_compile_args={
                'cxx': ["-O3"],
                'nvcc': ["-O3", "-std=c++17", "-Xptxas", "-v"],
            },
        )
        ext_modules.append(cuda_ext_s5)

        cuda_ext_s2 = CUDAExtension(
            "mymatmul.gpu._matmul_cuda_ext_s2",
            sources=["mymatmul/gpu/_matmul_cuda_ext_s2.cu"],
            extra_compile_args={
                'cxx': ["-O3"],
                'nvcc': ["-O3", "-std=c++17", "-Xptxas", "-v"],
            },
        )
        ext_modules.append(cuda_ext_s2)
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
