from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(
      name='sync_bn_cpu',
      ext_modules=[
      CppExtension('sync_bn_cpu', [
            'src/cpu/operator.cpp',
            'src/cpu/sync_bn.cpp',
	])
      ],
      cmdclass={
	'build_ext': BuildExtension
    })


setup(
    name='sync_bn_gpu',
    ext_modules=[
        CUDAExtension('sync_bn_gpu', [
            'src/gpu/operator.cpp',
            'src/gpu/sync_bn_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
