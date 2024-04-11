import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

parent_dir = os.path.dirname(os.path.abspath(__file__))

tnn_utils = CUDAExtension(
    name="elrec_kernel",
    sources=[
        parent_dir + "/efficient_kernel_wrap.cpp", 
        parent_dir + "/efficient_tt_cuda.cu", 
    ],
    extra_compile_args={
        "cxx": [
            "-O3",
            "-g",
            "-DUSE_MKL",
            "-m64",
            "-mfma",
            "-masm=intel",
        ],
        "nvcc": [
            "-O3",
            "-g",
            "--expt-relaxed-constexpr",
            "-D__CUDA_NO_HALF_OPERATORS__",
        ],
    },
)

setup(
    name="elrec_kernel",
    description="elrec tt embedding cuda kernel",
    packages=find_packages(),
    ext_modules=[tnn_utils],
    cmdclass={"build_ext": BuildExtension},
)
