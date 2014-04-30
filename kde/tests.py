__author__ = 'gabriel'
import unittest
import kernels
from methods.pure_python import VariableBandwidthKde, FixedBandwidthKde
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.integrate import quad, tplquad


class TestMultivariateNormal(unittest.TestCase):

    def test_mvn1d(self):
        mvn = kernels.MultivariateNormal([0], [1])
        self.assertEqual(mvn.ndim, 1)
        q = quad(mvn.pdf, -5., 5.)
        self.assertAlmostEqual(q[0], 1.0, places=5)
        x = np.linspace(-1, 1, 10)
        y = mvn.pdf(x)
        y_expct = norm.pdf(x)
        for y1, y2 in zip(y, y_expct):
            self.assertAlmostEqual(y1, y2)

    def test_mvn3d(self):
        mvn = kernels.MultivariateNormal([0, 0, 0], [1, 1, 1])
        self.assertEqual(mvn.ndim, 3)
        q = tplquad(mvn.pdf, -5, 5, lambda x:-5, lambda x:5, lambda x,y: -5, lambda x,y: 5)
        self.assertAlmostEqual(q[0], 1.0, places=5)
        x = np.meshgrid(*([np.linspace(-1, 1, 10)]*3))
        xa = np.vstack((x[0].flatten(), x[1].flatten(), x[2].flatten())).transpose()
        y = mvn.pdf(*x)
        y_expct = multivariate_normal.pdf(xa, mean=[0, 0, 0], cov=np.eye(3)).reshape((10, 10, 10))
        self.assertEqual(np.sum(np.abs((y - y_expct)).flatten()>1e-12), 0) # no single difference > 1e-12


class TestFixedBandwidthKde(unittest.TestCase):

    def test_kde_1d(self):
        data_1d = np.array([0, 3])
        bd_1d = [1.] # FIXME: should probably accept a float here
        kde = FixedBandwidthKde(data_1d, bandwidths=bd_1d)
        q = quad(kde.pdf, -5., 8.)
        self.assertAlmostEqual(q[0], 1.0, places=5)
        self.assertIsInstance(kde.pdf(0.), float)
        x = np.linspace(-1, 4, 50)
        y = kde.pdf(x)
        self.assertIsInstance(y, np.ndarray)
        y_expct = 0.5 * (norm.pdf(x, loc=0.) + norm.pdf(x, loc=3.))
        for y1, y2 in zip(y, y_expct):
            self.assertAlmostEqual(y1, y2)

    def test_kde_3d(self):
        d = np.meshgrid(*([[0, 3]]*3))
        data_3d = np.vstack(tuple([x.flatten() for x in d])).transpose()
        bd_3d = np.ones(3)
        kde = FixedBandwidthKde(data_3d, bandwidths=bd_3d)
        self.assertEqual(kde.ndata, 8)
        self.assertIsInstance(kde.pdf(0., 0., 0.), float)
        x = np.meshgrid(*([np.linspace(-1, 1, 10)]*3))
        xa = np.vstack((x[0].flatten(), x[1].flatten(), x[2].flatten())).transpose()
        y = kde.pdf(*x)
        y_expct = np.zeros((10, 10, 10))
        for i in range(kde.ndata):
            y_expct += multivariate_normal.pdf(xa, mean=data_3d[i, :], cov=np.eye(3)).reshape((10, 10, 10))
        y_expct /= kde.ndata
        self.assertEqual(np.sum(np.abs((y - y_expct)).flatten()>1e-12), 0) # no single difference > 1e-12


class TestVariableBandwidthKde(unittest.TestCase):

    def test_kde_1d(self):
        data_1d = np.array([0, 3])
        bd_1d = [1., 2.]
        kde = VariableBandwidthKde(data_1d, bandwidths=bd_1d)
        self.assertIsInstance(kde.pdf(0.), float)
        q = quad(kde.pdf, -10., 20.)
        self.assertAlmostEqual(q[0], 1.0, places=5)
        x = np.linspace(-1, 4, 50)
        y = kde.pdf(x)
        self.assertIsInstance(y, np.ndarray)
        y_expct = 0.5 * (norm.pdf(x, loc=0.) + norm.pdf(x, loc=3., scale=2))
        for y1, y2 in zip(y, y_expct):
            self.assertAlmostEqual(y1, y2)

    def test_kde_3d(self):
        d = np.meshgrid(*([[0, 3]]*3))
        data_3d = np.vstack(tuple([x.flatten() for x in d])).transpose()
        bd_3d = np.vstack(tuple([np.ones(3)+np.random.random(3) for i in range(1,9)]))
        kde = VariableBandwidthKde(data_3d, bandwidths=bd_3d)
        self.assertIsInstance(kde.pdf(0., 0., 0.), float)
        self.assertIsInstance(kde.pdf([0., 1.], [0., 1.], [0., 1.]), np.ndarray)
        x = np.meshgrid(*([np.linspace(-1, 1, 10)]*3))
        xa = np.vstack((x[0].flatten(), x[1].flatten(), x[2].flatten())).transpose()
        y = kde.pdf(*x)
        y_expct = np.zeros((10, 10, 10))
        for i in range(kde.ndata):
            y_expct += multivariate_normal.pdf(xa, mean=data_3d[i, :], cov=np.eye(3) * bd_3d[i, :]**2).reshape((10, 10, 10))
        y_expct /= kde.ndata
        self.assertEqual(np.sum(np.abs((y - y_expct)).flatten()>1e-12), 0) # no single difference > 1e-12