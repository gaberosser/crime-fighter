__author__ = 'gabriel'
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name = "Mvn",
    ext_modules = [Extension(
        "pymvn",
        sources=["pymvn.pyx", "vbkde.cpp", "libalg/src/alglibinternal.cpp", "libalg/src/alglibmisc.cpp", "libalg/src/ap.cpp"],
        language="c++",
        extra_compile_args=["-O3"],
        )],
    cmdclass = {'build_ext': build_ext},
)
