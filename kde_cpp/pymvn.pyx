# distutils: language = c++
# distutils: sources = vbkde.cpp
# distutils: libraries = alglib
# distutils: library_dirs = lib
# distutils: include_dirs = libalg/src

from libcpp cimport bool
from libcpp.vector cimport vector

# cdef extern from "<vector>" namespace "std":
#     cdef cppclass vector[T]:
#         vector()
#         void push_back(T&)

cdef extern from "mvn.h":
    cdef cppclass Mvn:
        Mvn(vector[double], vector[double])
        double pdf(vector[double])

cdef extern from "vbkde.h":
    cdef cppclass FixedBandwidthKde:
        FixedBandwidthKde(vector[vector[double]], vector[double], bool)
        double pdf(vector[double])
        vector[double] pdf(vector[vector[double]])

cdef vector[double] x1
x1.push_back(0.0)
x1.push_back(0.0)
x1.push_back(0.0)
cdef vector[double] x2
x2.push_back(1.0)
x2.push_back(1.0)
x2.push_back(1.0)
cdef vector[vector[double]] data
data.push_back(x1)
data.push_back(x2)

cdef vector[double] bd
bd.push_back(1.0)
bd.push_back(2.0)
bd.push_back(3.0)

cdef FixedBandwidthKde *f = new FixedBandwidthKde(data, bd, True)
print f.pdf(x1)

# cdef class PyMvn:
#     cdef Mvn *thisptr
#     def __cinit__(self, vector[double] mean, vector[double] stdev):
#         self.thisptr = new Mvn(mean, stdev)
#     def pdf(self, vector[double] x):
#         return self.thisptr.pdf(x)

cdef class PyFixedBandwidthKde:
    cdef FixedBandwidthKde *thisptr
    def __cinit__(self, vector[vector[double]] data, vector[double] bd, bool normed=True):
        self.thisptr = new FixedBandwidthKde(data, bd, normed)
    def pdf(self, vector[double] x):
        return self.thisptr.pdf(x)
    def pdfm(self, vector[vector[double]] x):
        return self.thisptr.pdf(x)