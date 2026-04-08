from setuptools import setup, find_packages, Extension
import numpy as np

cpp_ext = Extension(
    "mymatmul.cpu._matmul_cpp_ext",
    sources=["mymatmul/cpu/_matmul_cpp_ext.cpp"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-std=c++11", "-mavx2", "-mfma", "-fopenmp"],
    extra_link_args=["-fopenmp"],
    language="c++",
)

setup(
    name="mymatmul",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[cpp_ext],
    install_requires=[
        "numpy",
        "numba",
    ],
)
