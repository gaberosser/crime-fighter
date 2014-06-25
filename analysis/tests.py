__author__ = 'gabriel'
import unittest
import numpy as np
import mock
from django.contrib.gis import geos
import spatial
import hotspot
import validation


class TestSpatial(unittest.TestCase):

    def test_grid_offset(self):
        """
        Simple square domain, test with offset coords
        """
        domain = geos.Polygon([
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 0),
            (0, 0),
        ])
        grid = spatial.create_spatial_grid(domain, grid_length=0.1) # no offset
        self.assertEqual(len(grid), 100)
        self.assertAlmostEqual(sum([g.area for g in grid]), 1.0)
        for g in grid:
            self.assertAlmostEqual(g.area, 0.01)
            self.assertTrue(g.intersects(domain))

        grid = spatial.create_spatial_grid(domain, grid_length=0.1, offset_coords=(0.05, 0.)) # offset
        self.assertEqual(len(grid), 110)
        self.assertAlmostEqual(sum([g.area for g in grid]), 1.0)
        self.assertAlmostEqual(grid[0].area, 0.005)
        for g in grid:
            self.assertTrue(g.intersects(domain))

        grid = spatial.create_spatial_grid(domain, grid_length=0.1, offset_coords=(0., 1.1)) # offset outside of domain
        self.assertEqual(len(grid), 100)
        self.assertAlmostEqual(sum([g.area for g in grid]), 1.0)

    def test_grid_circle(self):
        """
        Circular domain
        """
        domain = geos.Point([0., 0.]).buffer(1.0, 100)
        grid = spatial.create_spatial_grid(domain, grid_length=0.25)
        self.assertEqual(len(grid), 60)
        self.assertAlmostEqual(sum([g.area for g in grid]), np.pi, places=3) # should approximate PI

    ## TODO: test case where a grid square is split into a multipolygon?  seems to work


class TestHotspot(unittest.TestCase):

    def test_stkernel_bowers(self):
        a = 2
        b = 10
        stk = hotspot.STKernelBowers(a, b)
        data = np.tile(np.arange(10), (3, 1)).transpose()
        stk.train(data)
        self.assertTupleEqual(stk.data.shape, (10, 3))
        z = stk.predict(10, 10, 10)
        zt_expct = np.sum(1 / (1 + a * np.arange(1, 11)))
        zd_expct = np.sum(1 / (1 + b * np.sqrt(2) * np.arange(1, 11)))
        self.assertAlmostEqual(z, zt_expct * zd_expct)

    def test_hotspot(self):
        stk = mock.create_autospec(hotspot.STKernelBowers)

        data = np.tile(np.arange(10), (3, 1)).transpose()
        h = hotspot.Hotspot(data, stk)
        self.assertEqual(stk.train.call_count, 1)
        self.assertListEqual(list(stk.train.call_args[0][0].flat), list(data.flat))

        a = 2
        b = 10
        stk = hotspot.STKernelBowers(a, b)
        h = hotspot.Hotspot(data, stk)
        self.assertEqual(h.predict(1.3, 4.6, 7.8), stk.predict(1.3, 4.6, 7.8))


class TestValidation(unittest.TestCase):

    def test_mcsampler(self):
        poly = geos.Polygon([
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 0),
            (0, 0),
        ])
        mcs = validation.mc_sampler(poly)
        with mock.patch('numpy.random.random', side_effect=np.random.RandomState(42).rand) as m:
            rvs = np.array([mcs.next() for i in range(10)])

        for r in rvs:
            self.assertTrue(geos.Point(list(r)).intersects(poly))

        # square domain so every rand() call should have generated a valid RV
        self.assertEqual(m.call_count, 20)

        # triangular poly
        poly = geos.Polygon([
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 0),
        ])

        mcs = validation.mc_sampler(poly)
        with mock.patch('numpy.random.random', side_effect=np.random.RandomState(42).rand) as m:
            rvs = np.array([mcs.next() for i in range(10)])

        for r in rvs:
            self.assertTrue(geos.Point(list(r)).intersects(poly))

        ncalls = m.call_count
        x_min, y_min, x_max, y_max = poly.extent
        expct_draws = np.random.RandomState(42).rand(ncalls).reshape((ncalls / 2, 2)) * np.array([x_max-x_min, y_max-y_min]) + \
            np.array([x_min, y_min])
        in_idx = np.where([geos.Point(list(x)).intersects(poly) for x in expct_draws])[0]
        self.assertEqual(len(in_idx), 10)
        for x, y in zip(expct_draws[in_idx, :], rvs):
            self.assertListEqual(list(x), list(y))

    ## test me: (in setUp) instantiation: sorting data by time, picking correct t_cutoff, calling model 'train' method once
    ## test me: extracting testing data correctly using dt_plus, dt_minus
    ## test me: predict_on_poly with lambda x,y: x to check it works
    ## test me: pai with a few known incidents in different squares
    ## test me: run correctly steps through time sequence


