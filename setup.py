from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

# Get the torch library path
torch_lib_path = os.path.dirname(torch.__file__)

setup(
    name='cuda_kernels',
    ext_modules=[
        CUDAExtension('cuda_kernels', [
            'model_code/cuda_kernels.cu'
        ],
        extra_compile_args={
            'cxx': ['-O2'],
            'nvcc': ['-O2']
        },
        library_dirs=[os.path.join(torch_lib_path, 'lib')],
        libraries=['c10', 'torch', 'torch_cpu', 'torch_python'],
        include_dirs=[os.path.join(torch_lib_path, 'include')]),
        
        CUDAExtension('dct_blur_kernels', [
            'model_code/dct_blur_kernels.cu'
        ],
        extra_compile_args={
            'cxx': ['-O2'],
            'nvcc': ['-O2']
        },
        library_dirs=[os.path.join(torch_lib_path, 'lib')],
        libraries=['c10', 'torch', 'torch_cpu', 'torch_python'],
        include_dirs=[os.path.join(torch_lib_path, 'include')])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }) 