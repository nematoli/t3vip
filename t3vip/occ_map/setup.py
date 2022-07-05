from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="OccMap_cuda",
    ext_modules=[
        CUDAExtension(
            "OccMap_cuda",
            [
                "OccMap_cuda.cpp",
                "OccMap_cuda_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
