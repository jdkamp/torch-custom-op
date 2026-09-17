from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="rmsnorm",
    version="0.1.0",
    packages=["rmsnorm"],
    ext_modules=[
        CppExtension(
            name="rmsnorm._C",
            sources=["csrc/rmsnorm.cpp"],
            extra_compile_args=["-O3", "-Werror=return-type"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)