from setuptools import setup, find_packages

setup(
    name="mymatmul",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "numba",
        "torch",
    ],
)
