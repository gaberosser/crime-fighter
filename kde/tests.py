__author__ = 'gabriel'
import unittest
import kernels
from methods.pure_python import VariableBandwidthKde, VariableBandwidthNnKde, FixedBandwidthKde, \
    WeightedVariableBandwidthNnKde, weighted_stdev
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

    def test_weighted_stdev(self):
        # 1D data

        # weights all unity
        data = np.linspace(0, 1, 11)
        weights = np.ones_like(data)
        # test the weighted stdev is equal to the usual UNBIASED estimator
        sw = weighted_stdev(data, weights)
        self.assertIsInstance(sw, float)
        self.assertEqual(sw, np.std(data, ddof=1))

        # weights all equal
        weights *= np.pi
        # test the weighted stdev is equal to the usual UNBIASED estimator
        self.assertEqual(weighted_stdev(data, weights), np.std(data, ddof=1))

        # two source distributions (weights differ)
        n = 1000000
        weighted_data = np.hstack((np.random.RandomState(42).randn(n) - 1, np.random.RandomState(42).randn(n) + 1))
        weights = np.hstack((np.ones(n) * 2 / 3., np.ones(n) / 3.))
        sw = weighted_stdev(weighted_data, weights)

        # analytic variance is 1 + 8/9
        self.assertAlmostEqual(sw, np.sqrt(1 + 8/9.), places=2)  # 2DP

        # 2D data
        data = np.tile(np.linspace(0, 1, 11).reshape(11, 1), (1, 2))
        weights = np.ones(11)
        sw = weighted_stdev(data, weights)
        self.assertEqual(sw.size, 2)
        self.assertEqual(sw[0], sw[1])


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

    def setUp(self):
        d = np.meshgrid(*([[0, 3]]*3))
        self.data_3d = np.vstack(tuple([x.flatten() for x in d])).transpose()
        self.bd_3d = np.vstack(tuple([np.ones(3)+np.random.RandomState(42).rand(3) for i in range(1, 9)]))
        self.kde = VariableBandwidthKde(self.data_3d, bandwidths=self.bd_3d)

    def test_kde_1d(self):
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
        # repeat unnormed
        y = kde.pdf(x, normed=False)
        y_expct *= kde.ndata
        for y1, y2 in zip(y, y_expct):
            self.assertAlmostEqual(y1, y2)

    def test_kde_3d(self):
        self.assertEqual(self.kde.ndata, self.data_3d.shape[0])
        self.assertIsInstance(self.kde.pdf([0., 1.], [0., 1.], [0., 1.]), np.ndarray)
        x = np.meshgrid(*([np.linspace(-1, 1, 10)]*3))
        xa = np.vstack((x[0].flatten(), x[1].flatten(), x[2].flatten())).transpose()
        y = self.kde.pdf(*x)
        y_expct = np.zeros((10, 10, 10))
        for i in range(self.kde.ndata):
            y_expct += multivariate_normal.pdf(xa,
                                               mean=self.data_3d[i, :],
                                               cov=np.eye(3) * self.bd_3d[i, :]**2).reshape((10, 10, 10))
        y_expct /= self.kde.ndata
        self.assertEqual(np.sum(np.abs((y - y_expct)).flatten()>1e-12), 0) # no single difference > 1e-12

        # UNNORMED
        y = self.kde.pdf(*x, normed=False)
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
        kde = VariableBandwidthNnKde(data, nn=2)
        nndiste = data[1] / np.std(data, ddof=1)
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
        self.assertAlmostEqual(kde.pdf(0.134), kdeu.pdf(0.134))

    pass

