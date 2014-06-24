__author__ = 'gabriel'
import unittest
import numpy as np
from mock import patch
from django.contrib.gis import geos
import spatial


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
