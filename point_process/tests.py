__author__ = 'gabriel'
import unittest
import simulate
import estimation
import utils
import models
import validate
import numpy as np
from mock import patch
from scipy.spatial import KDTree
import os
import pickle

cd = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(cd, 'test_data')

class TestSimulation(unittest.TestCase):

    def test_mohler_simulation(self):
        c = simulate.MohlerSimulation()

        # test background
        c.initialise_background()
        num_bg = c.number_bg
        self.assertEqual(c.data.shape[0], num_bg)
        # check that all events are background
        self.assertTrue(np.all(np.isnan(c.data[:, -1])))

        # generate aftershocks
        shocks = c.generate_aftershocks()
        self.assertEqual(c.number_aftershocks, len(shocks))
        self.assertEqual(c.number_aftershocks, sum(~np.isnan(c.data[:, -1])))
        self.assertTrue(np.all(c.data[~np.isnan(c.data[:, -1]), -1] >= 0))
        self.assertTrue(np.all(c.data[~np.isnan(c.data[:, -1]), -1] < max(c.data[:, 0])))


class TestUtils(unittest.TestCase):

    def test_self_linkage(self):
        data1 = np.random.randn(5000, 3)
        max_t = max_d = 0.5
        i, j = utils.linkages_euclidean(data1, max_d=max_d, max_t=max_t)
        # manually test restrictions
        # all time differences positive
        self.assertTrue(np.all(data1[j, 0] > data1[i, 0]))
        # all time diffs less than max_t
        self.assertTrue(np.all(data1[j, 0] - data1[i, 0] <= max_t))
        # all distances <= max_d
        d = np.sqrt((data1[j, 1] - data1[i, 1])**2 + (data1[j, 2] - data1[i, 2])**2)
        self.assertTrue(np.all(d <= max_d))

    def test_cross_linkage(self):
        data_source = np.random.randn(5000, 3)
        data_target = np.random.randn(1000, 3)
        max_t = max_d = 0.5
        i, j = utils.linkages_euclidean(data_source=data_source, max_d=max_d, max_t=max_t, data_target=data_target)
        self.assertTrue(np.all(i < 5000))
        self.assertTrue(np.all(j < 1000))
        # manually test restrictions
        # all time differences positive
        self.assertTrue(np.all(data_target[j, 0] > data_source[i, 0]))
        # all time diffs less than max_t
        self.assertTrue(np.all(data_target[j, 0] - data_source[i, 0] <= max_t))
        # all distances <= max_d
        d = np.sqrt((data_target[j, 1] - data_source[i, 1])**2 + (data_target[j, 2] - data_source[i, 2])**2)
        self.assertTrue(np.all(d <= max_d))


class TestPointProcessStochasticNn(unittest.TestCase):

    def setUp(self):
        self.c = simulate.MohlerSimulation()
        self.c.number_to_prune = 3500
        self.c.seed(42)
        self.c.run()
        self.data = self.c.data[:, :3]

    def test_linkage(self):
        r = models.PointProcessStochasticNn(max_trigger_d=0.75, max_trigger_t=80)
        r.set_data(self.data)
        link_mesh = r._set_linkages_meshed()
        link_iter = r._set_linkages_iterated()
        self.assertListEqual(list(link_iter[0]), list(link_mesh[0]))
        self.assertListEqual(list(link_iter[1]), list(link_mesh[1]))
        ## TODO: (1) use a known linkage structure to test output, (2) test linkage_cols

    def test_point_process(self):
        """
        Tests the output of the PP stochastic method based on a given random seed.
        The tests are all based on KNOWN results, NOT on the ideal results.  Failing some of these tests may still
        indicate an improvement.
        """
        r = models.PointProcessStochasticNn(max_trigger_d=0.75, max_trigger_t=80)
        models.estimation.set_seed(42)
        ps = r.train(self.data, niter=15, verbose=False, tol_p=1e-8)
        self.assertEqual(r.ndata, self.data.shape[0])
        self.assertEqual(len(r.num_bg), 15)
        self.assertEqual(r.niter, 15)
        self.assertAlmostEqual(r.l2_differences[0], 0.0011281, places=3)
        self.assertAlmostEqual(r.l2_differences[-1], 0.0001080, places=3)
        num_bge = [
            1949,
            1880,
            1758,
            1682,
            1655,
            1627,
            1624,
            1606,
            1616,
            1612,
            1597,
            1595,
            1598,
            1600,
            1597
        ]
        self.assertListEqual(r.num_bg, num_bge)
        self.assertListEqual(r.num_trig, [r.ndata - x for x in r.num_bg])
        self.assertEqual(len(r.linkage[0]), 6927)
        t = np.linspace(0, max(self.data[:, 0]), 10000)
        zt = r.bg_t_kde.pdf(t, normed=False)
        # mean BG_t
        mt = np.mean(zt)
        self.assertAlmostEqual(mt, 5.379642997277057, places=3)  # should be 5.71
        # stdev BG_t
        # should be as low as possible (no time variation in simulation):
        stdevt = np.std(zt)
        self.assertAlmostEqual(stdevt, 0.87447835663897755, places=3)
        # mean BG_x, BG_y
        # should be (0, 0)
        x, y = np.meshgrid(np.linspace(-15, 15, 200), np.linspace(-15, 15, 200))
        zxy = r.bg_xy_kde.pdf(x,y,normed=False)
        mx = np.sum(x * zxy) / x.size
        my = np.sum(y * zxy) / y.size
        self.assertAlmostEqual(mx, -0.16809429994382197, places=3)  # should be 0
        self.assertAlmostEqual(my, -0.13588032032996472, places=3)  # should be 0
        # stdev BG_x, BG_y
        stdevx = np.sqrt((np.sum(x**2 * zxy)/(x.size - 1)) - mx**2)
        stdevy = np.sqrt((np.sum(y**2 * zxy)/(y.size - 1)) - my**2)
        self.assertAlmostEqual(stdevx, 6.1230874070314485)  # should be 4.5
        self.assertAlmostEqual(stdevy, 6.1335341582567739)  # should be 4.5
        self.assertTrue(2 * np.abs(stdevx - stdevy)/(stdevx + stdevy) < 0.003) # should be 0
        # TODO: trigger


class TestSampling(unittest.TestCase):

    def test_roulette_selection(self):
        num_iter = 100
        weights = [0.8, 0.2]
        # expected output
        prng = np.random.RandomState(42)
        rvs = prng.rand(num_iter)
        num1 = sum(rvs > 0.8)
        # force random seed of 42
        estimation.set_seed(42)
        # with patch('numpy.random.RandomState', return_value=np.random.RandomState(42)) as mock:
        idx = np.array([estimation.weighted_choice_np(weights) for i in range(num_iter)])
        self.assertEqual(sum(idx == 0), num_iter - num1)
        self.assertEqual(sum(idx == 1), num1)

    # FIXME: test this in the PointProcess class instead
    # def test_weighted_sampling(self):
    #     # all BG
    #     P = np.array(
    #         [
    #             [1., 0.],
    #             [0., 1.]
    #         ]
    #     )
    #     res = estimation.sample_events(P)
    #     self.assertListEqual(list(res[0]), [0, 0])
    #     self.assertListEqual(list(res[1]), [1, 1])
    #     # all trigger
    #     P = np.array(
    #         [
    #             [0., 1.],
    #             [1., 0.]
    #         ]
    #     )
    #     res = estimation.sample_events(P)
    #     self.assertListEqual(list(res[0]), [0, 1])
    #     self.assertListEqual(list(res[1]), [1, 0])
    #     # mix
    #     P = np.array(
    #         [
    #             [0.6, 0.2],
    #             [0.4, 0.8]
    #         ]
    #     )
    #     estimation.set_seed(42)
    #     res = estimation.sample_events(P)
    #     idx = np.array([estimation.weighted_choice_np(w) for w in P.transpose()])
    #     prng = np.random.RandomState(42)
    #     rvs = prng.rand(2)
    #     self.assertListEqual(list(res[0]), [0, 0 if rvs[0] <= 0.6 else 1])
    #     self.assertListEqual(list(res[1]), [1, 0 if rvs[1] > 0.8 else 1])

    ## FIXME: this is obsolete, replace with test for estimator_bowers
    def test_initial_guess(self):
        data = np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 0., 1.],
            [1., 2., 0.],
            [1., 100., 0.],
            [2., 0., 0.],
            [3., 0., 0.],
            [100., 0., 0.],
        ])
        P = estimation.initial_guess(data)

        # check normalisation
        colsums  = np.sum(P, axis=0)
        for x in colsums:
            self.assertAlmostEqual(x, 1.)

        # check no negatives
        self.assertEqual(np.sum(P < 0), 0)

        # check time asymmetry
        for i in range(P.shape[0]):
            for j in range(i+1, P.shape[0]):
                self.assertEqual(P[j, i], 0.)

        # sanity check: events further apart are less likely to be related
        self.assertTrue(P[0, 1] > P[0, 2])
        self.assertTrue(P[0, 2] > P[0, 4])
        self.assertTrue(P[2, 3] > P[2, 4])

    ## FIXME: replace with test on PointProcess object
    # def test_sample_bg_and_interpoint(self):
    #     data = np.array([
    #         [0., 0., 0.],
    #         [1., 0., 0.],
    #         [1., 0., 1.],
    #         [1., 2., 0.],
    #         [1., 100., 0.],
    #         [2., 0., 0.],
    #         [3., 0., 0.],
    #         [100., 0., 0.],
    #     ])
    #     P = estimation.initial_guess(data)
    #     # with patch('numpy.random.RandomState', return_value=np.random.RandomState(42)) as mock:
    #     estimation.set_seed(42)
    #     res = estimation.sample_events(P)
    #     for x0, x1 in res:
    #         self.assertTrue(x0 >= x1)
    #
    #     # with patch('numpy.random.RandomState', return_value=np.random.RandomState(42)) as mock:
    #     estimation.set_seed(42)
    #     bg, interpoint, cause_effect = estimation.sample_bg_and_interpoint(data, P)
    #
    #     self.assertEqual(interpoint.shape[0], cause_effect.shape[0])
    #     self.assertEqual(bg.shape[0] + interpoint.shape[0], data.shape[0])
    #     self.assertListEqual(list(bg[0, :]), list(data[0, :]))
    #
    #     # no negative times
    #     self.assertTrue(np.sum(interpoint[:, 0] < 0) == 0)
    #
    #     # no cause and effect pairs have the same index
    #     self.assertFalse(np.any(cause_effect[:, 0] == cause_effect[:, 1]))
    #
    #     # check division was created as expected
    #     bg_n = 0
    #     interp_n = 0
    #     for eff, cau in res:
    #         if eff == cau:
    #             self.assertListEqual(list(bg[bg_n, :]), list(data[eff, :]))
    #             bg_n += 1
    #         else:
    #             self.assertListEqual(list(interpoint[interp_n, :]), list(data[eff, :] - data[cau, :]))
    #             interp_n += 1



class TestNonStationaryPoisson(unittest.TestCase):

    # test first with stationary process
    def test_stationary(self):
        t = simulate.nonstationary_poisson(lambda x:x, 50000)
        dt = np.diff(t)
        self.assertLess(np.abs(np.mean(dt) - 1.0), 0.05)
        self.assertLess(np.abs(np.var(dt) - 1.0), 0.05)

    def test_non_stationary(self):
        t = simulate.nonstationary_poisson(lambda x:np.sqrt(2 * x), 500)
        dt = np.diff(t)
        n = len(t)
        self.assertLess(np.abs(n/float(500**2/2) - 1.0), 0.05)


class TestKDNearestNeighbours(unittest.TestCase):

    def test_2d_array(self):
        # 2D regular array
        a, b = np.meshgrid(*[np.arange(1,11) for i in range(2)])
        kd = KDTree(np.vstack((a.flatten(), b.flatten())).transpose())
        d, idx = kd.query([1, 1], k=4)
        self.assertEqual(d[0], 0.)
        self.assertEqual(d[1], 1.)
        self.assertAlmostEqual(d[3], np.sqrt(2))
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[3], 11)


class TestValidate(unittest.TestCase):

    def test_confusion_matrix(self):
        pass
    # upon running, size of each successive dataset grows


