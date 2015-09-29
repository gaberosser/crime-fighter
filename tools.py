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


def get_ellipse_coords(a=0.0, b=0.0, x=0.0, y=0.0, angle=0.0, k=2):
    """ Draws an ellipse using (360*k + 1) discrete points; based on pseudo code
    given at http://en.wikipedia.org/wiki/Ellipse
    k = 1 means 361 points (degree by degree)
    a = major axis distance,
    b = minor axis distance,
    x = offset along the x-axis
    y = offset along the y-axis
    angle = clockwise rotation [in degrees] of the ellipse;
        * angle=0  : the ellipse is aligned with the positive x-axis
        * angle=30 : rotated 30 degrees clockwise from positive x-axis
    """
    pts = np.zeros((360*k+1, 2))

    beta = -angle * np.pi/180.0
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    alpha = np.radians(np.r_[0.:360.:1j*(360*k+1)])

    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)

    pts[:, 0] = x + (a * cos_alpha * cos_beta - b * sin_alpha * sin_beta)
    pts[:, 1] = y + (a * cos_alpha * sin_beta + b * sin_alpha * cos_beta)

    return pts