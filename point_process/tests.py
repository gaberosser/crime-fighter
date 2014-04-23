__author__ = 'gabriel'
import unittest
import simulate
import numpy as np

class TestNonStationaryPoisson(unittest.TestCase):

    # test first with stationary process
    def test_stationary(self):
        t = simulate.nonstationary_poisson(lambda x:x, 50000, 50000)
        dt = np.diff(t)
        self.assertLess(np.abs(np.mean(dt) - 1.0), 0.05)
        self.assertLess(np.abs(np.var(dt) - 1.0), 0.05)

    def test_non_stationary(self):
        t = simulate.nonstationary_poisson(lambda x:np.sqrt(2 * x), 1000000, 500)
        dt = np.diff(t)
        n = len(t)
        self.assertLess(np.abs(n/float(500**2/2) - 1.0), 0.05)


class TestMultivariateNormal(unittest.TestCase):
    # test in 1D by comparing to numpy
    # test in 2/3D using symmetry and known values
    pass


class TestVariableBandwidthKde(unittest.TestCase):
    # test values at datapoints in single datum case
    # test nn distances are computed correctly on a regular grid
    pass