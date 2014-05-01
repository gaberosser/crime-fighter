cdef extern from "math.h":
    float exp(float t)
    float sqrt(float t)
    float pow(float t, float n)

import numpy as np
cimport numpy as np
ctypedef np.float_t DTYPE_t

PI = np.pi

cdef class MultivariateNormal:
    cdef public int ndim
    cdef public np.ndarray mean
    cdef public np.ndarray var

    def __cinit__(self, np.ndarray[DTYPE_t, ndim=1] mean, np.ndarray[DTYPE_t, ndim=1] var):
        self.mean = mean
        self.var = var
        self.ndim = mean.size
        if var.size != self.ndim:
            raise AttributeError("Dims of mean and var do not match")

    cpdef np.ndarray pdf(self, np.ndarray[DTYPE_t, ndim=2] x):
        if x.shape[1] != self.ndim:
            raise AttributeError("Second dimension must match ndim")
        cdef np.ndarray[DTYPE_t, ndim=1] res = self.pdf_c(x)
        return res

    cdef np.ndarray pdf_c(self, np.ndarray[DTYPE_t, ndim=2] x):
        cdef np.ndarray[DTYPE_t, ndim=1] res = np.zeros(x.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            res[i] = self.normnd(x[i])
        return res

    cdef float normnd(self, np.ndarray[DTYPE_t, ndim=1] x):
        cdef float a = 1.0
        cdef float b = 0.0
        for i in range(self.ndim):
            a *= sqrt(self.var[i])
            b -= pow(x[i] - self.mean[i], 2) / (2. * self.var[i])
        return pow(2*PI, -self.ndim/2) * exp(b) / a
