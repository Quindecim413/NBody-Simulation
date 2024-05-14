from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include as np_get_include

ext = Extension(
    name='simulator',
    sources=['simulator.pyx'],
    language="c++",  
    library_dirs=['libSimulation'],
    libraries=['NBodySimulation'],
    include_dirs=['.', np_get_include(), 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include']
)

setup(
    ext_modules = cythonize(ext)
)