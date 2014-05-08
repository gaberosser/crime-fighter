import numpy as np
PI = np.pi

def normnd(x, mu, var):
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