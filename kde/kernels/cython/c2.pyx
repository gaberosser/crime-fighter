import numpy as np
cimport numpy as np
DTYPE = np.float64

# cdef extern from "math.h":
#     float exp(float t)
#     float sqrt(float t)
#     float pow(float t, float n)

PI = np.pi

def normnd(np.ndarray x, np.ndarray mu, np.ndarray var):
    # each input is a (1 x self.ndim) array
    cdef int ndim
    if isinstance(x, np.ndarray):
        ndim = x.size
    else:
        ndim = 1
    cdef float a = np.power(2 * PI, ndim/2.)
    cdef float b = np.prod(np.sqrt(var))
    cdef float c = -np.sum((x - mu)**2 / (2 * var))
    return np.exp(c) / (a * b)

def norm3d(np.ndarray mu, np.ndarray var, float x1, float x2, float x3):
    cdef float a = np.power(2 * PI, 3/2.)
    cdef float b = np.prod(np.sqrt(var))
    cdef float c = -np.sum(([x1, x2, x3] - mu)**2 / (2 * var))
    return np.exp(c) / (a * b)