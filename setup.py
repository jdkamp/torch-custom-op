from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME, CUDAExtension

sources = ["csrc/rmsnorm.cpp"]
cxx_flags = ["-O3", "-Werror=return-type"]

if CUDA_HOME is not None:
    ext = CUDAExtension(
        name="rmsnorm._C",
        sources = sources + ["csrc/rmsnorm_cuda.cu"],
        extra_compile_args={"cxx": cxx_flags, "nvcc": ["-O3"]},
    )
else:
    ext = CppExtension(
        name="rmsnorm._C",
        sources=sources,
        extra_compile_args=cxx_flags,
    )


setup(
    name="rmsnorm",
    version="0.1.0",
    packages=["rmsnorm"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)