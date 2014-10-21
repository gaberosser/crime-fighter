__author__ = 'gabriel'
import unittest
import kernels
from models import VariableBandwidthKde, VariableBandwidthNnKde, FixedBandwidthKde, \
    WeightedVariableBandwidthNnKde
from data.models import CartesianSpaceTimeData, CartesianData
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.integrate import quad, tplquad
from functools import partial


def quad_pdf_fun(*args, **kwargs):
    func = kwargs.pop('func')
    t = np.array(args[::-1])
    t.shape = (1, len(args))
    return func(t)


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


class TestKernelMultivariateNormal(unittest.TestCase):

    def test_mvn1d(self):
        mvn = kernels.MultivariateNormal([0], [1])
        self.assertEqual(mvn.ndim, 1)

        # def quad_pdf_fun(x):
        #     # required because quad calls functions with single datapoint
        #     t = np.array(x)
        #     t.shape = (1, 1)
        #     return mvn.pdf(t)

        q = quad(partial(quad_pdf_fun, func=mvn.pdf), -5., 5.)
        self.assertAlmostEqual(q[0], 1.0, places=5)

        # test with absolute values
        x = np.linspace(-1, 1, 10).reshape(10, 1)
        y = mvn.pdf(x)
        y_expct = norm.pdf(x)
        for y1, y2 in zip(y, y_expct):
            self.assertAlmostEqual(y1, y2)

        # repeat with Data type
        x = CartesianData(np.linspace(-1, 1, 10))
        y = mvn.pdf(x)
        for y1, y2 in zip(y, y_expct):
            self.assertAlmostEqual(y1, y2)

        m = mvn.marginal_pdf(x)
        self.assertListEqual(list(y), list(m))
        c = mvn.marginal_cdf(0.)
        self.assertAlmostEqual(c, 0.5)

    def test_mvn3d(self):
        mvn = kernels.MultivariateNormal([0, 0, 0], [1, 1, 1])
        self.assertEqual(mvn.ndim, 3)

        q = tplquad(partial(quad_pdf_fun, func=mvn.pdf), -5, 5, lambda x: -5, lambda x: 5, lambda x, y: -5, lambda x, y: 5)
        self.assertAlmostEqual(q[0], 1.0, places=5)

        # test with absolute values
        x = np.meshgrid(*([np.linspace(-1, 1, 10)]*3))
        x = np.vstack((x[0].flatten(), x[1].flatten(), x[2].flatten())).transpose().reshape(1000, 3)
        y = mvn.pdf(x)
        y_expct = multivariate_normal.pdf(x, mean=[0, 0, 0], cov=np.eye(3))
        self.assertEqual(np.sum(np.abs((y - y_expct)).flatten()>1e-12), 0) # no single difference > 1e-12

        # repeat with Data type
        x = CartesianData(x)
        y = mvn.pdf(x)
        self.assertEqual(np.sum(np.abs((y - y_expct)).flatten()>1e-12), 0) # no single difference > 1e-12


class TestKernelLinear(unittest.TestCase):
    ## TODO: some trivial tests
    pass

class TestFixedBandwidthKde(unittest.TestCase):

    def test_kde_1d(self):
        data_1d = np.array([0, 3])
        bd_1d = [1.]
        kde = FixedBandwidthKde(data_1d, bandwidths=bd_1d)

        q = quad(partial(quad_pdf_fun, func=kde.pdf), -5., 8.)
        self.assertAlmostEqual(q[0], 1.0, places=5)
        x = np.linspace(-1, 4, 50).reshape(50, 1)
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
        y = kde.pdf(xa)
        y_expct = np.zeros_like(y)
        for i in range(kde.ndata):
            y_expct += multivariate_normal.pdf(xa, mean=data_3d[i, :], cov=np.eye(3))
        y_expct /= kde.ndata
        self.assertEqual(np.sum(np.abs((y - y_expct)).flatten()>1e-12), 0) # no single difference > 1e-12


class TestVariableBandwidthKde(unittest.TestCase):

    def setUp(self):
        d = np.meshgrid(*([[0, 3]]*3))
        self.data_3d = np.vstack(tuple([x.flatten() for x in d])).transpose()
        self.bd_3d = np.vstack(tuple([np.ones(3)+np.random.RandomState(42).rand(3) for i in range(1, 9)]))
        self.kde = VariableBandwidthKde(self.data_3d, bandwidths=self.bd_3d)

    def test_kde_1d(self):
        data_1d = np.array([0, 3])
        bd_1d = [1., 2.]
        kde = VariableBandwidthKde(data_1d, bandwidths=bd_1d)
        q = quad(partial(quad_pdf_fun, func=kde.pdf), -10., 20.)
        self.assertAlmostEqual(q[0], 1.0, places=5)
        x = np.linspace(-1, 4, 50).reshape(50, 1)
        y = kde.pdf(x)
        self.assertIsInstance(y, np.ndarray)
        y_expct = 0.5 * (norm.pdf(x, loc=0.) + norm.pdf(x, loc=3., scale=2))
        for y1, y2 in zip(y, y_expct):
            self.assertAlmostEqual(y1, y2)
        # repeat unnormed
        y = kde.pdf(x, normed=False)
        y_expct *= kde.ndata
        for y1, y2 in zip(y, y_expct):
            self.assertAlmostEqual(y1, y2)

    def test_kde_3d(self):
        self.assertEqual(self.kde.ndata, self.data_3d.shape[0])
        x = np.meshgrid(*([np.linspace(-1, 1, 10)]*3))
        xa = np.vstack((x[0].flatten(), x[1].flatten(), x[2].flatten())).transpose()
        y = self.kde.pdf(xa)
        self.assertIsInstance(y, np.ndarray)
        y_expct = np.zeros_like(y)
        for i in range(self.kde.ndata):
            y_expct += multivariate_normal.pdf(xa,
                                               mean=self.data_3d[i, :],
                                               cov=np.eye(3) * self.bd_3d[i, :]**2)
        y_expct /= self.kde.ndata
        self.assertEqual(np.sum(np.abs((y - y_expct)).flatten()>1e-12), 0) # no single difference > 1e-12

        # UNNORMED
        y = self.kde.pdf(xa, normed=False)
        y_expct *= self.kde.ndata
        self.assertEqual(np.sum(np.abs((y - y_expct)).flatten()>1e-12), 0) # no single difference > 1e-12

    def test_marginals(self):
        x = np.linspace(-1, 1, 10)

        # check marginals in each dim
        for dim in range(3):

            p = self.kde.marginal_pdf(x, dim=dim)
            pu = self.kde.marginal_pdf(x, dim=dim, normed=False)
            p_expct = np.zeros(x.shape)
            for i in range(self.kde.ndata):
                p_expct += norm.pdf(x, loc=self.data_3d[i, dim], scale=self.bd_3d[i, dim])
            p_expct /= self.kde.ndata
            self.assertEqual(np.sum(np.abs(p - p_expct) > 1e-12), 0) # no single difference > 1e-12
            self.assertEqual(np.sum(np.abs(p * self.kde.ndata - pu) > 1e-12), 0)

            c = self.kde.marginal_cdf(x, dim=dim)
            c_expct = np.zeros(x.shape)
            for i in range(self.kde.ndata):
                c_expct += norm.cdf(x, loc=self.data_3d[i, dim], scale=self.bd_3d[i, dim])
            c_expct /= self.kde.ndata
            self.assertEqual(np.sum(np.abs(c - c_expct) > 1e-12), 0) # no single difference > 1e-12

    def test_inverse_cdf(self):
        for y in [0.01, 0.1, 0.5, 0.9, 0.99, 0.9999]:
            for dim in range(2):
                x = self.kde.marginal_icdf(y, dim=dim)
                self.assertAlmostEqual(self.kde.marginal_cdf(x, dim=dim), y)

    def test_moments(self):
        eps = 1e-5  # results accurate to 3sf
        lower = [0] * 3
        upper = [0] * 3
        lower[0] = self.kde.marginal_icdf(eps, dim=0)
        upper[0] = self.kde.marginal_icdf(1 - eps, dim=0)
        lower[1] = self.kde.marginal_icdf(eps, dim=1)
        upper[1] = self.kde.marginal_icdf(1 - eps, dim=1)
        lower[2] = self.kde.marginal_icdf(eps, dim=2)
        upper[2] = self.kde.marginal_icdf(1 - eps, dim=2)

        m1 = self.kde.marginal_mean
        m2 = self.kde.marginal_second_moment
        var = self.kde.marginal_variance

        m1_computed = []
        m2_computed = []
        for i in range(3):

            opt = lambda t: t * self.kde.marginal_pdf(t, dim=i)
            m1_computed.append(quad(opt, lower[i], upper[i])[0])
            self.assertAlmostEqual(m1_computed[-1], m1[i], places=2)  # NB places is DP

            opt = lambda t: t ** 2 * self.kde.marginal_pdf(t, dim=i)
            m2_computed.append(quad(opt, lower[i], upper[i])[0])
            self.assertAlmostEqual(m2_computed[-1], m2[i], places=2)  # NB places is DP

            self.assertAlmostEqual(m2_computed[-1] - m1_computed[-1] ** 2, var[i], places=2)


class TestVariableBandwidthKdeNn(unittest.TestCase):

    def test_kde_1d(self):
        data = np.linspace(0, 1, 11)
        kde = VariableBandwidthNnKde(data, nn=1)
        nndiste = np.diff(data)[0] / np.std(data, ddof=1)
        for n in kde.nn_distances:
            self.assertAlmostEqual(n, nndiste)


class TestWeightedVariableBandwidthKdeNn(unittest.TestCase):

    # NB weighted stdev tested in class TestHelperFunctions

    def test_kde_1d(self):
        data = np.linspace(0, 1, 11)
        kde = WeightedVariableBandwidthNnKde(data, weights=np.ones_like(data), nn=2)

        self.assertTrue(np.all(kde.weights == 1.))

        # nn distances and bandwidths calculated as before
        kdeu = VariableBandwidthNnKde(data, nn=2)
        self.assertListEqual(list(kde.nn_distances), list(kdeu.nn_distances))
        self.assertListEqual(list(kde.bandwidths), list(kdeu.bandwidths))

        # pdf unchanged when weights are all 1
        x = np.array([[0.134]])
        self.assertAlmostEqual(kde.pdf(x), kdeu.pdf(x))


