__author__ = 'gabriel'
import unittest
import simulate
import estimation
import validate
import numpy as np
from mock import patch
from scipy.spatial import KDTree


class TestSampling(unittest.TestCase):

    def test_roulette_selection(self):
        num_iter = 100
        weights = [0.8, 0.2]
        # expected output
        prng = np.random.RandomState(42)
        rvs = prng.rand(num_iter)
        num1 = sum(rvs > 0.8)
        # force random seed of 42
        with patch('numpy.random.RandomState', return_value=np.random.RandomState(42)) as mock:
            idx = np.array([estimation.weighted_choice_np(weights) for i in range(num_iter)])
            self.assertEqual(sum(idx == 0), num_iter - num1)
            self.assertEqual(sum(idx == 1), num1)

    def test_weighted_sampling(self):
        # all BG
        P = np.array(
            [
                [1., 0.],
                [0., 1.]
            ]
        )
        res = estimation.sample_events(P)
        self.assertListEqual(list(res[0]), [0, 0])
        self.assertListEqual(list(res[1]), [1, 1])
        # all trigger
        P = np.array(
            [
                [0., 1.],
                [1., 0.]
            ]
        )
        res = estimation.sample_events(P)
        self.assertListEqual(list(res[0]), [0, 1])
        self.assertListEqual(list(res[1]), [1, 0])
        # mix
        P = np.array(
            [
                [0.6, 0.2],
                [0.4, 0.8]
            ]
        )
        with patch('numpy.random.RandomState', return_value=np.random.RandomState(42)) as mock:
            res = estimation.sample_events(P)
            idx = np.array([estimation.weighted_choice_np(w) for w in P.transpose()])
        prng = np.random.RandomState(42)
        rvs = prng.rand(2)
        self.assertListEqual(list(res[0]), [0, 0 if rvs[0] <= 0.6 else 1])
        self.assertListEqual(list(res[1]), [1, 0 if rvs[1] > 0.8 else 1])

    def test_pairwise_differences(self):
        data = np.array([
            [0., 0., 1.],
            [0., 1., 0.],
            [1., 0., 0.],
            [1., 1., 1.],
        ])
        pdiff = estimation.pairwise_differences(data)
        self.assertTupleEqual(pdiff.shape, (4, 4, 3))
        for i in range(4):
            self.assertListEqual(list(pdiff[i, i, :]), [0., 0., 0.])
        self.assertListEqual(list(pdiff[0, 1, :]), [0., 1., -1.])
        self.assertListEqual(list(pdiff[0, 2, :]), [1., 0., -1.])
        self.assertListEqual(list(pdiff[0, 3, :]), [1., 1., 0.])

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

    def test_sample_bg_and_interpoint(self):
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
        with patch('numpy.random.RandomState', return_value=np.random.RandomState(42)) as mock:
            res = estimation.sample_events(P)
        for x0, x1 in res:
            self.assertTrue(x0 >= x1)

        with patch('numpy.random.RandomState', return_value=np.random.RandomState(42)) as mock:
            bg, interpoint, cause_effect = estimation.sample_bg_and_interpoint(data, P)

        self.assertEqual(interpoint.shape[0], cause_effect.shape[0])
        self.assertEqual(bg.shape[0] + interpoint.shape[0], data.shape[0])
        self.assertListEqual(list(bg[0, :]), list(data[0, :]))

        # no negative times
        self.assertTrue(np.sum(interpoint[:, 0] < 0) == 0)

        # no cause and effect pairs have the same index
        self.assertFalse(np.any(cause_effect[:, 0] == cause_effect[:, 1]))

        # check division was created as expected
        bg_n = 0
        interp_n = 0
        for eff, cau in res:
            if eff == cau:
                self.assertListEqual(list(bg[bg_n, :]), list(data[eff, :]))
                bg_n += 1
            else:
                self.assertListEqual(list(interpoint[interp_n, :]), list(data[eff, :] - data[cau, :]))
                interp_n += 1



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

    def test_spatial_grid(self):
        from django.contrib.gis.geos import Polygon
        # create a circle polygon centred at origin
        x = 5. * np.cos(np.linspace(0, 2 * np.pi, 101))
        y = 5. * np.sin(np.linspace(0, 2 * np.pi, 101))
        x[-1] = x[0]
        y[-1] = y[0]
        circ = Polygon(zip(x, y))

        a = validate.ValidationBase([], [], circ)

        edges, centres = a.create_spatial_grid(grid_length=1.0)
        self.assertEqual(len(edges), 121)
        self.assertEqual(len(centres), 100)
        expctd_edges = np.meshgrid(range(-5, 6), range(-5, 6))
        expctd_edges = np.vstack((expctd_edges[0].flatten(), expctd_edges[1].flatten())).transpose().astype(float)
        for x, y in zip(np.array(edges)[:, 0], expctd_edges[:, 0]):
            self.assertAlmostEqual(x, y)
        for x, y in zip(np.array(edges)[:, 1], expctd_edges[:, 1]):
            self.assertAlmostEqual(x, y)
        expctd_centres = np.meshgrid(np.arange(-4.5, 5.5), np.arange(-4.5, 5.5))
        expctd_centres = np.vstack((expctd_centres[0].flatten(), expctd_centres[1].flatten())).transpose().astype(float)
        for x, y in zip(np.array(centres)[:, 0], expctd_centres[:, 0]):
            self.assertAlmostEqual(x, y)
        for x, y in zip(np.array(centres)[:, 1], expctd_centres[:, 1]):
            self.assertAlmostEqual(x, y)

        edges, centres = a.create_spatial_grid(grid_length=1.5)
        self.assertEqual(len(edges), 81)
        self.assertEqual(len(centres), 64)
        self.assertAlmostEqual(np.min(np.array(edges)[:, 0]), -6.0)
        self.assertAlmostEqual(np.max(np.array(edges)[:, 0]), 6.0)

        edges, centres = a.create_spatial_grid(grid_length=1., offset_coords=(0.1, 0.2))
        self.assertEqual(len(edges), 144)
        self.assertEqual(len(centres), 121)
        self.assertAlmostEqual(np.min(np.array(edges)[:, 0]), -5.9)
        self.assertAlmostEqual(np.max(np.array(edges)[:, 0]), 5.1)
        self.assertAlmostEqual(np.min(np.array(edges)[:, 1]), -5.8)
        self.assertAlmostEqual(np.max(np.array(edges)[:, 1]), 5.2)

    def test_split_data(self):
        data = np.arange(300).reshape((100, 3))
        a = validate.ValidationBase(data, [], [])
        prng = np.random.RandomState(42)
        expctd_p = prng.rand(100)
        with patch('numpy.random.random', side_effect=lambda x:np.random.RandomState(42).rand(x)) as mock:
            a.split_data(test_frac=0.2)
        self.assertEqual(len(a.test_idx[0]), sum(expctd_p < 0.2))
        self.assertEqual(len(a.train_idx[0]), sum(expctd_p >= 0.2))
        for i, idx in enumerate(np.where(expctd_p < 0.2)[0]):
            self.assertListEqual(list(a.testing[i]), list(data[idx]))
        for i, idx in enumerate(np.where(expctd_p >= 0.2)[0]):
            self.assertListEqual(list(a.training[i]), list(data[idx]))
