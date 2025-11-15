from setuptools import setup, find_packages, Extension
import numpy as np
import os

# Base directory for C++ wrapper sources
CPP_BASE = os.path.join("multireg", "cpp_wrappers")

ext_modules = [
    Extension(
        name="multireg.cpp_wrappers.radius_neighbors",
        sources=[
            os.path.join(CPP_BASE, "cpp_utils", "cloud", "cloud.cpp"),
            os.path.join(CPP_BASE, "radius_neighbors", "neighbors.cpp"),
            os.path.join(CPP_BASE, "radius_neighbors", "wrapper.cpp"),
        ],
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    ),
    Extension(
        name="multireg.cpp_wrappers.grid_subsampling",
        sources=[
            os.path.join(CPP_BASE, "cpp_utils", "cloud", "cloud.cpp"),
            os.path.join(CPP_BASE, "grid_subsampling", "grid_subsampling.cpp"),
            os.path.join(CPP_BASE, "grid_subsampling", "wrapper.cpp"),
        ],
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    ),
]

setup(
    name="multireg",
    version="1.0.0",
    packages=find_packages(),
    ext_modules=ext_modules,
)
