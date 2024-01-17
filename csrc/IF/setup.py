from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import glob
import os.path as osp

sources = glob.glob('*.cpp') + glob.glob('*.cu')
ROOT_DIR = osp.dirname(osp.abspath(__file__))

setup(
        name='IFNeuron',
        version="1.0",
        ext_modules=[
            CUDAExtension(
                name='IF_cppcuda', 
                sources=sources,
                include_dirs=[ROOT_DIR],
                extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
                )
            ],
        packages = ["IFNeuron"],
        cmdclass={
            'build_ext': BuildExtension
            }
        )

