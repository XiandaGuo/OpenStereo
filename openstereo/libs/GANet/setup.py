from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension


setup(
    name='GANet',
    ext_modules=[
        CUDAExtension('GANet', [
            'src/GANet_cuda.cpp',
            'src/GANet_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
