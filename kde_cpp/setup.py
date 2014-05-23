__author__ = 'gabriel'
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "Mvn",
    ext_modules = cythonize(
           "pymvn.pyx",                 # our Cython source
           sources=["vbkde.cpp"],
           language="c++",             # generate C++ code
      ))