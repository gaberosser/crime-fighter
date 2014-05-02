__author__ = 'gabriel'
import numpy as np
from scipy.special import erf
PI = np.pi

# a few helper functions
def normpdf(x, mu, var):
    return 1/np.sqrt(2*PI*var) * np.exp(-(x - mu)**2 / (2*var))


def normcdf(x, mu, var):
    return 0.5 * (1 + erf((x - mu) / (np.sqrt(2 * var))))