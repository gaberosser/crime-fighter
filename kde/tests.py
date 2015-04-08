__author__ = 'gabriel'
import unittest
import kernels
from models import VariableBandwidthKde, VariableBandwidthNnKde, FixedBandwidthKde, \
    WeightedVariableBandwidthNnKde, FixedBandwidthKdeSeparable
from data.models import DataArray, SpaceTimeDataArray
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.integrate import quad, dblquad, tplquad
from scipy.special import erf
from functools import partial
import ipdb


def quad_pdf_fun(*args, **kwargs):
    func = kwargs.pop('func')
    t = np.array(args[::-1])
    t.shape = (1, len(args))
    return func(t, **kwargs)


class TestHelperFunctions(unittest.TestCase):

    def test_normcdf(self):
        self.assertAlmostEqual(kernels.normcdf(0., 0., 1.), 0.5)
        self.assertAlmostEqual(kernels.normcdf(10., 0., 1.), 1.0)
        self.assertAlmostEqual(kernels.normcdf(0., 0., 10.), 0.5)

        x = np.linspace(-1, 1, 50)
        kc = kernels.normcdf(x, 0.1, 0.5)
        kc_expct = norm.cdf(x, loc=0.1, scale=np.sqrt(0.5))
        self.assertTrue(np.all(np.abs(kc - kc_expct) < 1e-14))


class TestKernelTemporalRadial(unittest.TestCase):

    def test_radial_norming(self):
        locations = [
            (0., 0.),
            (0., 1.),
            (0., 1.),
            (0., 2.),
        ]
        scales = [
            (1., 1.),
            (1., 0.1),
            (1., 1.),
            (1., 0.5)
        ]

        for l, s in zip(locations, scales):
            k = kernels.RadialTemporal(l, s)

            # radial norming
            q = dblquad(lambda x, y: k.pdf(np.sqrt(x ** 2 + y ** 2), dims=[1]), -10, 10,
                        lambda *args: -10, lambda *args: 10)
            self.assertAlmostEqual(q[0], 1.0, places=5)

    def test_cdf(self):
        locations = [
            (0., 0.),
            (0., 1.),
            (0., 1.),
            (0., 2.),
        ]
        scales = [
            (1., 1.),
            (1., 0.1),
            (1., 1.),
            (1., 0.5)
        ]

        for l, s in zip(locations, scales):
            k = kernels.RadialTemporal(l, s)
            self.assertAlmostEqual(k.marginal_cdf(-1., dim=1), 0., places=12)
            self.assertAlmostEqual(k.marginal_cdf(0., dim=1), 0., places=12)
            self.assertAlmostEqual(k.marginal_cdf(1000., dim=0), 1., places=12)
            self.assertAlmostEqual(k.marginal_cdf(1000., dim=1), 1., places=12)


class TestKernelMultivariateNormal(unittest.TestCase):

    kernel_class = kernels.MultivariateNormal
    location = [0.5, 1.0, 1.5]
    scale = [1.0, 2.0, 3.0]
    min_nd = 1
    tol_places = 5
    tol = 1e-12

    def limits(self, n):
        if n == 1:
            return -5. * self.scale[0], 5. * self.scale[0]
        if n == 2:
            return self.limits(1) + (lambda x: -5. * self.scale[1], lambda x: 5. * self.scale[1])
        if n == 3:
            return self.limits(2) + (lambda x, y: -5. * self.scale[2], lambda x, y: 5. * self.scale[2])

    def setUp(self):
        self.kernels = {}
        for i in range(self.min_nd, 4):
            self.kernels[i] = self.kernel_class(self.location[:i], self.scale[:i])

    def test_norming(self):
        # 1D
        if self.min_nd < 2:
            q = quad(partial(quad_pdf_fun, func=self.kernels[1].pdf),
                     *self.limits(1))
            self.assertAlmostEqual(q[0], 1.0, places=self.tol_places)


        # 2D
        if self.min_nd < 3:
            q = dblquad(partial(quad_pdf_fun, func=self.kernels[2].pdf),
                        *self.limits(2)
                        )
            self.assertAlmostEqual(q[0], 1.0, places=self.tol_places)

        # 3D
        q = tplquad(partial(quad_pdf_fun, func=self.kernels[3].pdf),
                    *self.limits(3)
                    )
        self.assertAlmostEqual(q[0], 1.0, places=self.tol_places)

    def expected_pdf(self, x):
        loc = self.location[:x.nd]
        vars = self.scale[:x.nd]
        res = multivariate_normal.pdf(x.data, mean=loc, cov=vars)
        if x.original_shape:
            res = res.reshape(x.original_shape)
        return res

    def expected_marginal_pdf(self, x, dim):
        assert x.nd == 1
        loc = self.location[dim]
        var = self.scale[dim]
        return norm.pdf(x.toarray(0), loc=loc, scale=np.sqrt(var))

    def expected_marginal_cdf(self, x, dim):
        assert x.nd == 1
        loc = self.location[dim]
        var = self.scale[dim]
        return norm.cdf(x.toarray(0), loc=loc, scale=np.sqrt(var))

    def test_pdf_values(self):
        for i in range(self.min_nd, 4):
            x = DataArray.from_meshgrid(
                *np.meshgrid(
                    *[np.linspace(-5, 5, 50)] * i
                )
            )
            y = self.kernels[i].pdf(x)
            ye = self.expected_pdf(x)
            self.assertTrue(np.all(np.abs(y - ye) < self.tol))

    def test_marginal_pdf_values(self):
        k = self.kernels[3]
        x = DataArray(np.linspace(-5, 5, 100))
        for i in range(3):
            y = k.marginal_pdf(x, dim=i)
            ye1 = k.pdf(x, dims=[i])
            ye2 = self.expected_marginal_pdf(x, i)
            self.assertTrue(np.all(y == ye1))
            self.assertTrue(np.all(np.abs(y - ye2) < self.tol))

    def test_marginal_cdf_values(self):
        k = self.kernels[3]
        x = DataArray(np.linspace(-5, 5, 100))
        for i in range(3):
            y = k.marginal_cdf(x, dim=i)
            ye = self.expected_marginal_cdf(x, i)
            self.assertTrue(np.all(np.abs(y - ye) < self.tol))


class TestKernelOneSided(TestKernelMultivariateNormal):

    kernel_class = kernels.SpaceTimeNormalOneSided
    tol_places = 4

    def limits(self, n):
        if n == 1:
            return 0., 5. * self.scale[0]
        else:
            return super(TestKernelOneSided, self).limits(n)

    def test_norming(self):
        # TODO: this is extremely slow.  Why?  Discontinuity at cutoff?
        pass

    def test_cutoff(self):
        x = DataArray(np.linspace(-5, 5, 100))
        self.assertTrue(np.all(self.kernels[1].pdf(x)[x < self.location[0]] == 0))
        x = DataArray.from_meshgrid(
            *np.meshgrid(
                np.linspace(-5, 5, 50),
                np.linspace(-5, 5, 50)
            )
        )
        res = self.kernels[2].pdf(x)
        self.assertTrue(np.all(res[x.toarray(0) < self.location[0]] == 0))
        self.assertTrue(np.all(res[x.toarray(0) >= self.location[0]] > 0.))

    def expected_pdf(self, x):
        res = super(TestKernelOneSided, self).expected_pdf(x)
        res[x.toarray(0) < self.location[0]] = 0.
        res[x.toarray(0) >= self.location[0]] *= 2.
        return res

    def expected_marginal_pdf(self, x, dim):
        res = super(TestKernelOneSided, self).expected_marginal_pdf(x, dim)
        if dim == 0:
            res[x.toarray(0) < self.location[0]] = 0.
            res[x.toarray(0) >= self.location[0]] *= 2.
        return res

    def expected_marginal_cdf(self, x, dim):
        if dim == 0:
            # analytic CDF
            res = erf((x.toarray(0) - self.location[0]) / (np.sqrt(2 * self.scale[0])))
            res[x.toarray(0) < self.location[0]] = 0.
        else:
            res = super(TestKernelOneSided, self).expected_marginal_cdf(x, dim)
        return res


class TestKernelReflective(TestKernelOneSided):

    kernel_class = kernels.SpaceTimeNormalReflective

    def test_cutoff(self):
        x = DataArray(np.linspace(-5, 5, 100))
        self.assertTrue(np.all(self.kernels[1].pdf(x)[x < 0] == 0))
        x = DataArray.from_meshgrid(
            *np.meshgrid(
                np.linspace(-5, 5, 50),
                np.linspace(-5, 5, 50)
            )
        )
        res = self.kernels[2].pdf(x)
        self.assertTrue(np.all(res[x.toarray(0) < 0] == 0))
        self.assertTrue(np.all(res[x.toarray(0) >= 0] > 0.))

    def expected_pdf(self, x):
        res = TestKernelMultivariateNormal.expected_pdf(self, x)
        x2 = x.copy()
        x2[:, 0] *= -1.0
        res2 = TestKernelMultivariateNormal.expected_pdf(self, x2)
        res += res2
        res[x.toarray(0) < 0] = 0.
        return res

    def expected_marginal_pdf(self, x, dim):
        res = TestKernelMultivariateNormal.expected_marginal_pdf(self, x, dim=dim)
        if dim == 0:
            x2 = -x.copy()
            res2 = TestKernelMultivariateNormal.expected_pdf(self, x2)
            res += res2
            res[x.toarray(0) < 0] = 0.
        return res

    def expected_marginal_cdf(self, x, dim):
        assert x.nd == 1
        loc = self.location[dim]
        var = self.scale[dim]
        if dim == 0:
            x = x.toarray(0)
            res = erf((x - loc) / (np.sqrt(2 * var)))
            res -= erf((-x - loc) / (np.sqrt(2 * var)))
            res[x < 0] = 0.
            return 0.5 * res
        else:
            res = TestKernelMultivariateNormal.expected_marginal_cdf(self, x, dim=dim)
        return res


class TestKernelMultivariateNormal2(unittest.TestCase):

    def test_mvn1d(self):
        mvn = kernels.MultivariateNormal([0], [1])
        self.assertEqual(mvn.ndim, 1)

        q = quad(partial(quad_pdf_fun, func=mvn.pdf), -5., 5.)
        self.assertAlmostEqual(q[0], 1.0, places=5)

        # test with absolute values
        x = np.linspace(-1, 1, 10).reshape(10, 1)
        y = mvn.pdf(x)
        y_expct = norm.pdf(x)
        for y1, y2 in zip(y, y_expct):
            self.assertAlmostEqual(y1, y2)

        # repeat with Data type
        x = DataArray(np.linspace(-1, 1, 10))
        y = mvn.pdf(x)
        for y1, y2 in zip(y, y_expct):
            self.assertAlmostEqual(y1, y2)

        m = mvn.marginal_pdf(x)
        self.assertListEqual(list(y), list(m))
        c = mvn.marginal_cdf(np.array(0.))
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
        x = DataArray(x)
        y = mvn.pdf(x)
        self.assertEqual(np.sum(np.abs((y - y_expct)).flatten()>1e-12), 0) # no single difference > 1e-12

    def test_marginals(self):
        mvn = kernels.MultivariateNormal([0, 0, 0], [1, 2, 3])
        x = np.linspace(-15, 15, 500)
        p = mvn.marginal_pdf(x, dim=0)
        p_expct = norm.pdf(x, loc=0., scale=1.)
        self.assertTrue(np.all(np.abs(p - p_expct) < 1e-14))
        p = mvn.marginal_pdf(x, dim=1)
        p_expct = norm.pdf(x, loc=0., scale=np.sqrt(2.))
        self.assertTrue(np.all(np.abs(p - p_expct) < 1e-14))
        p = mvn.marginal_pdf(x, dim=2)
        p_expct = norm.pdf(x, loc=0., scale=np.sqrt(3.))
        self.assertTrue(np.all(np.abs(p - p_expct) < 1e-14))

    def test_partials(self):
        mvn = kernels.MultivariateNormal([0, 0, 0], [1, 2, 3])
        arr = np.meshgrid(np.linspace(-15, 15, 20), np.linspace(-15, 15, 20))
        xy = DataArray(np.concatenate([t[..., np.newaxis] for t in arr], axis=2))

        # attempt to call with data of too few dims
        with self.assertRaises(AttributeError):
            p = mvn.partial_marginal_pdf(xy.getdim(0), dim=0)

        p = mvn.partial_marginal_pdf(xy, dim=0)  # should be marginal in 2nd and 3rd dims
        p_expct = mvn.pdf(xy, dims=[1, 2])
        self.assertTrue(np.all(np.abs(p - p_expct) < 1e-14))
        p_expct = mvn.marginal_pdf(xy.getdim(0), dim=1) * mvn.marginal_pdf(xy.getdim(1), dim=2)
        self.assertTrue(np.all(np.abs(p - p_expct) < 1e-14))

        p = mvn.partial_marginal_pdf(xy, dim=1)  # should be marginal in 1st and 3rd dims
        p_expct = mvn.pdf(xy, dims=[0, 2])
        self.assertTrue(np.all(np.abs(p - p_expct) < 1e-14))
        # test directly
        p_expct = mvn.marginal_pdf(xy.getdim(0), dim=0) * mvn.marginal_pdf(xy.getdim(1), dim=2)
        self.assertTrue(np.all(np.abs(p - p_expct) < 1e-14))


class TestSpaceTimeNormalOneSided(unittest.TestCase):


    pass

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
        kde = VariableBandwidthNnKde(data, number_nn=2)
        nndiste = np.diff(data)[0] / np.std(data, ddof=1)
        for n in kde.nn_distances:
            self.assertAlmostEqual(n, nndiste)


class TestWeightedVariableBandwidthKdeNn(unittest.TestCase):

    # NB weighted stdev tested in class TestHelperFunctions

    def test_kde_1d(self):
        data = np.linspace(0, 1, 11)
        kde = WeightedVariableBandwidthNnKde(data, weights=np.ones_like(data), number_nn=2)

        self.assertTrue(np.all(kde.weights == 1.))

        # nn distances and bandwidths calculated as before
        kdeu = VariableBandwidthNnKde(data, number_nn=2)
        self.assertListEqual(list(kde.nn_distances), list(kdeu.nn_distances))
        self.assertListEqual(list(kde.bandwidths), list(kdeu.bandwidths))

        # pdf unchanged when weights are all 1
        x = np.array([[0.134]])
        self.assertAlmostEqual(kde.pdf(x), kdeu.pdf(x))



def quad_pdf_fun_st(*args, **kwargs):
    func = kwargs.pop('func')
    t = SpaceTimeDataArray([args[::-1]])
    return func(t, **kwargs)


class TestFixedBandwidthSeparableKde(unittest.TestCase):

    def test_equivalence(self):
        # check that KDE is equivalent to the non-separable version when only one target is present
        x = np.random.rand(1, 3)
        data = SpaceTimeDataArray(x)
        ks = FixedBandwidthKdeSeparable(data, bandwidths=[1., 2., 3.])
        k = FixedBandwidthKde(data, bandwidths=[1., 2., 3.])

        arr = np.meshgrid(np.ones(10), np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        txy = SpaceTimeDataArray(np.concatenate([t[..., np.newaxis] for t in arr], axis=3))

        ps = ks.pdf(txy)
        p = k.pdf(txy)

        self.assertTrue(np.all(np.abs(ps - p) < 1e-16))

    def test_equivalence2(self):
        # separable kde should be equivalent to product of two separate KDEs
        data = SpaceTimeDataArray(np.random.rand(10, 3))
        ks = FixedBandwidthKdeSeparable(data, bandwidths=[1., 2., 3.])

        kt = FixedBandwidthKde(data.time, bandwidths=[1.])
        kxy = FixedBandwidthKde(data.space, bandwidths=[2., 3.])

        arr = np.meshgrid(np.ones(10), np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        txy = SpaceTimeDataArray(np.concatenate([t[..., np.newaxis] for t in arr], axis=3))

        # norming True
        p = ks.pdf(txy, normed=True)
        pt = kt.pdf(txy.time, normed=True)
        pxy = kxy.pdf(txy.space, normed=True)
        p_comb = pt * pxy

        self.assertFalse(np.any(np.abs(p - p_comb) > 1e-14))

        # norming False
        # this is important - should be equivalent to norming one component and not the other
        p = ks.pdf(txy, normed=False)
        pt = kt.pdf(txy.time, normed=False)
        pxy = kxy.pdf(txy.space, normed=True)
        p_comb = pt * pxy

        self.assertFalse(np.any(np.abs(p - p_comb) > 1e-14))

    def test_kde_3d(self):
        x = np.random.rand(10, 3)
        data = SpaceTimeDataArray(x)  # first col is now time
        ks = FixedBandwidthKdeSeparable(data, bandwidths=[1., 2., 3.])

        arr = np.meshgrid(np.ones(10), np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        txy = SpaceTimeDataArray(np.concatenate([t[..., np.newaxis] for t in arr], axis=3))

        mp_time = ks.marginal_pdf(txy.time, dim=0)
        pp_space = ks.partial_marginal_pdf(txy.space, dim=0)
        p = ks.pdf(txy)

        self.assertTupleEqual(p.shape, (10, 10, 10,))
        self.assertTupleEqual(mp_time.shape, (10, 10, 10,))
        self.assertTupleEqual(pp_space.shape, (10, 10, 10,))

        # build expected output
        mpdf_time_expct = np.zeros_like(txy.time.toarray(0))
        ppdf_space_expct = np.zeros_like(txy.time.toarray(0))
        for row in x:
            mpdf_time_expct += norm.pdf(txy.time.toarray(0), loc=row[0], scale=1.) / x.shape[0]
            ppdf_space_expct += norm.pdf(txy.getdim(1).toarray(0), loc=row[1], scale=2.) * \
                                norm.pdf(txy.getdim(2).toarray(0), loc=row[2], scale=3.) / x.shape[0]

        pdf_expct = mpdf_time_expct * ppdf_space_expct
        tol = 1e-14
        self.assertFalse(np.any(np.abs(mpdf_time_expct - mp_time) > tol))
        self.assertFalse(np.any(np.abs(ppdf_space_expct - pp_space) > tol))
        self.assertFalse(np.any(np.abs(pdf_expct - p) > tol))

    def test_norming(self):
        data = SpaceTimeDataArray(np.random.rand(10, 3))
        ks = FixedBandwidthKdeSeparable(data, bandwidths=[1., 2., 3.])

        # normed

        q = quad(partial(quad_pdf_fun, func=ks.marginal_pdf, dim=0), -5, 6)
        self.assertAlmostEqual(q[0], 1.0, places=4)
        q = quad(partial(quad_pdf_fun, func=ks.marginal_pdf, dim=1), -10, 10)
        self.assertAlmostEqual(q[0], 1.0, places=4)
        q = quad(partial(quad_pdf_fun, func=ks.marginal_pdf, dim=2), -15, 15)
        self.assertAlmostEqual(q[0], 1.0, places=4)

        # unnormed
        q = quad(partial(quad_pdf_fun, func=ks.marginal_pdf, dim=0, normed=False), -5, 6)
        self.assertAlmostEqual(q[0], 10.0, places=4)
        q = quad(partial(quad_pdf_fun, func=ks.marginal_pdf, dim=1, normed=False), -10, 10)
        self.assertAlmostEqual(q[0], 10.0, places=4)
        q = quad(partial(quad_pdf_fun, func=ks.marginal_pdf, dim=2, normed=False), -15, 15)
        self.assertAlmostEqual(q[0], 10.0, places=4)
