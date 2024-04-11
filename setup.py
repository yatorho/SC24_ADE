import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def cuda_extension():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(root_dir, "ecorec", "tt_emb")

    extension = CUDAExtension(
        name="ecorec._C",
        sources=[
            os.path.join(parent_dir, "tt_emb_kernel_wrap.cpp"),
            os.path.join(parent_dir, "tt_emb_cuda.cu"),
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

    return extension


setup(
    name="ecorec",
    description="EcoRec is a TT-based library for efficient recommendation system.",
    version="0.0",
    packages=find_packages(),
    ext_modules=[cuda_extension()],
    install_requires=[
        "torch",
        "torchrec",
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False
)
