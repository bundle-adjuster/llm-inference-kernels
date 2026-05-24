"""PyTorch C++/CUDA extension build.

    python setup.py build_ext --inplace

Architecture: pure-CUDA launchers live in kernels/ and operate on raw device
pointers (so they also build standalone via CMake). bindings/extension.cpp is
the thin Torch glue that unpacks torch::Tensor and calls those launchers.
"""
import glob
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))

# microbench.cu files carry their own main() — exclude them from the extension.
cuda_sources = [f for f in glob.glob("kernels/**/*.cu", recursive=True)
                if "microbench" not in f]
sources = sorted(glob.glob("bindings/*.cpp") + cuda_sources)

setup(
    name="llmik",
    version="0.0.1",
    description="Custom CUDA kernels for LLM inference",
    ext_modules=[
        CUDAExtension(
            name="llmik_cuda",
            sources=sources,
            include_dirs=[os.path.join(HERE, "kernels")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
