__author__ = 'gabriel'
import unittest
import kernels
from methods.pure_python import VariableBandwidthKde, FixedBandwidthKde
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.integrate import quad, tplquad


class TestHelperFunctions(unittest.TestCase):

    def test_normpdf(self):
        x = np.linspace(-4, 4, 100)
        y1 = norm.pdf(x, loc=0.1, scale=2)
        y2 = kernels.normpdf(x, 0.1, 4)
        for a, b in zip(y1, y2):
            self.assertAlmostEqual(a, b)

    def test_normcdf(self):
        self.assertAlmostEqual(kernels.normcdf(0., 0., 1.), 0.5)
        self.assertAlmostEqual(kernels.normcdf(10., 0., 1.), 1.0)
        self.assertAlmostEqual(kernels.normcdf(0., 0., 10.), 0.5)


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
        m = mvn.marginal_pdf(x)
        self.assertListEqual(list(y), list(m))
        c = mvn.marginal_cdf(0.)
        self.assertAlmostEqual(c, 0.5)

    def test_mvn3d(self):
        mvn = kernels.MultivariateNormal([0, 0, 0], [1, 1, 1])
        self.assertEqual(mvn.ndim, 3)
        int_fun = lambda z, y, x: mvn.pdf(np.array([x, y, z]))
        q = tplquad(int_fun, -5, 5, lambda x:-5, lambda x:5, lambda x,y: -5, lambda x,y: 5)
        self.assertAlmostEqual(q[0], 1.0, places=5)
        x = np.meshgrid(*([np.linspace(-1, 1, 10)]*3))
        x = np.vstack((x[0].flatten(), x[1].flatten(), x[2].flatten())).transpose()
        y = mvn.pdf(x)
        y_expct = multivariate_normal.pdf(x, mean=[0, 0, 0], cov=np.eye(3))
        self.assertEqual(np.sum(np.abs((y - y_expct)).flatten()>1e-12), 0) # no single difference > 1e-12


class TestFixedBandwidthKde(unittest.TestCase):

    def test_kde_1d(self):
        data_1d = np.array([0, 3])
        bd_1d = [1.] # FIXME: should probably accept a float here
        kde = FixedBandwidthKde(data_1d, bandwidths=bd_1d)
        q = quad(kde.pdf, -5., 8.)
        self.assertAlmostEqual(q[0], 1.0, places=5)
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
        # NORMED
        data_1d = np.array([0, 3])
        bd_1d = [1., 2.]
        kde = VariableBandwidthKde(data_1d, bandwidths=bd_1d)
        q = quad(kde.pdf, -10., 20.)
        self.assertAlmostEqual(q[0], 1.0, places=5)
        x = np.linspace(-1, 4, 50)
        y = kde.pdf(x)
        self.assertIsInstance(y, np.ndarray)
        y_expct = 0.5 * (norm.pdf(x, loc=0.) + norm.pdf(x, loc=3., scale=2))
        for y1, y2 in zip(y, y_expct):
            self.assertAlmostEqual(y1, y2)

        # UNNORMED
        kde = VariableBandwidthKde(data_1d, bandwidths=bd_1d, normed=False)
        q = quad(kde.pdf, -10., 20.)
        self.assertAlmostEqual(q[0], 2.0, places=5)
        y = kde.pdf(x)
        y_expct *= kde.ndata
        for y1, y2 in zip(y, y_expct):
            self.assertAlmostEqual(y1, y2)

    def test_kde_3d(self):
        # NORMED
        d = np.meshgrid(*([[0, 3]]*3))
        data_3d = np.vstack(tuple([x.flatten() for x in d])).transpose()
        bd_3d = np.vstack(tuple([np.ones(3)+np.random.random(3) for i in range(1,9)]))
        kde = VariableBandwidthKde(data_3d, bandwidths=bd_3d)
        self.assertIsInstance(kde.pdf([0., 1.], [0., 1.], [0., 1.]), np.ndarray)
        x = np.meshgrid(*([np.linspace(-1, 1, 10)]*3))
        xa = np.vstack((x[0].flatten(), x[1].flatten(), x[2].flatten())).transpose()
        y = kde.pdf(*x)
        y_expct = np.zeros((10, 10, 10))
        for i in range(kde.ndata):
            y_expct += multivariate_normal.pdf(xa, mean=data_3d[i, :], cov=np.eye(3) * bd_3d[i, :]**2).reshape((10, 10, 10))
        y_expct /= kde.ndata
        self.assertEqual(np.sum(np.abs((y - y_expct)).flatten()>1e-12), 0) # no single difference > 1e-12

        # UNNORMED
        kde = VariableBandwidthKde(data_3d, bandwidths=bd_3d, normed=False)
        y = kde.pdf(*x)
        y_expct *= kde.ndata
        self.assertEqual(np.sum(np.abs((y - y_expct)).flatten()>1e-12), 0) # no single difference > 1e-12

    def test_marginals(self):
        # NORMED
        d = np.meshgrid(*([[0, 3]]*3))
        data_3d = np.vstack(tuple([x.flatten() for x in d])).transpose()
        bd_3d = np.vstack(tuple([np.ones(3)+np.random.random(3) for i in range(1,9)]))
        kde_normed = VariableBandwidthKde(data_3d, bandwidths=bd_3d)
        kde_unnormed = VariableBandwidthKde(data_3d, bandwidths=bd_3d, normed=False)
        x = np.linspace(-1, 1, 10)

        # check marginals in each dim
        for dim in range(3):

            p = kde_normed.marginal_pdf(x, dim=dim)
            pu = kde_unnormed.marginal_pdf(x, dim=dim)
            p_expct = np.zeros(x.shape)
            for i in range(kde_normed.ndata):
                p_expct += norm.pdf(x, loc=data_3d[i, dim], scale=bd_3d[i, dim])
            p_expct /= kde_normed.ndata
            self.assertEqual(np.sum(np.abs(p - p_expct) > 1e-12), 0) # no single difference > 1e-12
            self.assertEqual(np.sum(np.abs(p * kde_normed.ndata - pu) > 1e-12), 0)

            c = kde_normed.marginal_cdf(x, dim=dim)
            cu = kde_unnormed.marginal_cdf(x, dim=dim)
            c_expct = np.zeros(x.shape)
            for i in range(kde_normed.ndata):
                c_expct += norm.cdf(x, loc=data_3d[i, dim], scale=bd_3d[i, dim])
            c_expct /= kde_normed.ndata
            self.assertEqual(np.sum(np.abs(c - c_expct) > 1e-12), 0) # no single difference > 1e-12
            self.assertListEqual(list(c), list(cu)) # always norm CDF

    def test_inverse_cdf(self):
        d = np.meshgrid(*([[0, 3]]*3))
        data_3d = np.vstack(tuple([x.flatten() for x in d])).transpose()
        bd_3d = np.vstack(tuple([np.ones(3)+np.random.random(3) for i in range(1,9)]))
        kde = VariableBandwidthKde(data_3d, bandwidths=bd_3d)
        kde_unnormed = VariableBandwidthKde(data_3d, bandwidths=bd_3d, normed=False)

        for y in [0.75, 0.9, 0.99, 0.9999]:
            for dim in range(2):
                x = kde.marginal_icdf(y, dim=dim)
                self.assertAlmostEqual(kde.marginal_cdf(x, dim=dim), y)
                xu = kde_unnormed.marginal_icdf(y, dim=dim)
                self.assertEqual(x, xu)