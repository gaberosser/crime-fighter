__author__ = 'gabriel'
import numpy as np


uint_dtypes = [(t, np.iinfo(t)) for t in (
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)]

int_dtypes = [(t, np.iinfo(t)) for t in (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
)]


def numpy_most_compact_int_dtype(arr):
    """
    Compress supplied array of integers as much as possible by changing the dtype
    :param arr:
    :return:
    """
    if np.any(arr < 0):
        dtypes = int_dtypes
    else:
        dtypes = uint_dtypes

    arr_max = arr.max()  ## FIXME: max ABS value
    for t, ii in dtypes:
        if arr_max <= ii.max:
            return arr.astype(t)

    raise ValueError("Unable to find a suitable datatype")