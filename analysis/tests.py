from validation import hotspot

__author__ = 'gabriel'
import unittest
import csv
import pickle
import numpy as np
import mock
from django.contrib.gis import geos
import spatial
import cad
from database.models import Division, DivisionType
from database.populate import setup_cad250_grid


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
        igrid, egrid, fgs = spatial.create_spatial_grid(domain, grid_length=0.1) # no offset
        self.assertEqual(len(igrid), 100)
        self.assertEqual(len(egrid), 100)
        self.assertAlmostEqual(sum([g.area for g in igrid]), 1.0)
        for g in igrid:
            self.assertAlmostEqual(g.area, 0.01)
            self.assertTrue(g.intersects(domain))
        for g in egrid:
            self.assertAlmostEqual(g[2], g[0] + 0.1)
            self.assertAlmostEqual(g[3], g[1] + 0.1)
        self.assertTrue(np.all(fgs))

        igrid, egrid, fgs = spatial.create_spatial_grid(domain, grid_length=0.1, offset_coords=(0.05, 0.)) # offset
        self.assertEqual(len(igrid), 110)
        self.assertEqual(len(igrid), 110)
        self.assertAlmostEqual(sum([g.area for g in igrid]), 1.0)
        self.assertAlmostEqual(igrid[0].area, 0.005)
        for g in igrid:
            self.assertTrue(g.intersects(domain))
        for g in egrid:
            self.assertAlmostEqual(g[2], g[0] + 0.1)
            self.assertAlmostEqual(g[3], g[1] + 0.1)
        # 20 grid squares are not fully within the domain
        self.assertEqual(sum(fgs), 90)

        igrid, egrid, fgs = spatial.create_spatial_grid(domain, grid_length=0.1, offset_coords=(0., 1.1)) # offset outside of domain
        self.assertEqual(len(igrid), 100)
        self.assertEqual(len(egrid), 100)
        self.assertAlmostEqual(sum([g.area for g in igrid]), 1.0)
        for g in igrid:
            self.assertAlmostEqual(g.area, 0.01)
            self.assertTrue(g.intersects(domain))
        for g in egrid:
            self.assertAlmostEqual(g[2], g[0] + 0.1)
            self.assertAlmostEqual(g[3], g[1] + 0.1)
        self.assertTrue(np.all(fgs))

    def test_grid_circle(self):
        """
        Circular domain
        """
        domain = geos.Point([0., 0.]).buffer(1.0, 100)
        igrid, egrid, fgs = spatial.create_spatial_grid(domain, grid_length=0.25)
        self.assertEqual(len(igrid), 60)
        self.assertEqual(len(igrid), len(egrid))
        self.assertAlmostEqual(sum([g.area for g in igrid]), np.pi, places=3) # should approximate PI
        for g in egrid:
            self.assertAlmostEqual(g[2], g[0] + 0.25)
            self.assertAlmostEqual(g[3], g[1] + 0.25)

    ## TODO: test case where a grid square is split into a multipolygon?  seems to work


class TestCadAnalysis(unittest.TestCase):

    def setUp(self):
        # load spatial points data
        with open('analysis/test_data/spatial_points.csv', 'r') as f:
            c = csv.reader(f)
            self.xy = np.array([[float(a) for a in b] for b in c])
        # setup grid
        with open('analysis/test_data/camden.pickle', 'r') as f:
            d = pickle.load(f)
            dt = DivisionType(name='borough', description='foo')
            dt.save()
            # repair FK link
            d['type'] = dt
        d = Division(**d)
        d.save()
        DivisionType(name='cad_250m_grid', description='foo').save()
        setup_cad250_grid(verbose=False, test=True)


    def test_jiggle_all_on_grid(self):
        data = np.array([
            [ 526875.0, 185625.0],
            [ 526875.1, 185625.1],
            [ 526874.9, 185625.0],
            [ 0., 0.],
            ])
        extent1 = np.array([ 526750.0, 185500.0, 527000.0, 185750.0])
        extent2 = extent1 - [250, 0, 250, 0]
        new_xy = cad.jiggle_all_points_on_grid(data[:, 0], data[:, 1])
        self.assertTupleEqual(data.shape, new_xy.shape)
        ingrid1 = lambda t: (t[0] >= extent1[0]) & (t[0] < extent1[2]) & (t[1] >= extent1[1]) & (t[1] < extent1[3])
        ingrid2 = lambda t: (t[0] >= extent2[0]) & (t[0] < extent2[2]) & (t[1] >= extent2[1]) & (t[1] < extent2[3])
        # import ipdb; ipdb.set_trace()
        self.assertTrue(ingrid1(new_xy[0]))
        self.assertTrue(ingrid1(new_xy[1]))
        self.assertFalse(ingrid2(new_xy[2]))
        self.assertListEqual(list(data[-1, :]), list(new_xy[-1, :]))

    def test_jiggle_on_and_off(self):
        new_xy = cad.jiggle_on_and_off_grid_points(self.xy[:, 0], self.xy[:, 1], scale=5)
        self.assertEqual(new_xy.shape[0], self.xy.shape[0])
        # all now unique:
        num_unique = sum([np.sum(np.sum(new_xy == t, axis=1) == 2)==1 for t in new_xy])
        self.assertEqual(num_unique, new_xy.shape[0])
        # no on-grid results remaining:
        divs = Division.objects.filter(type='cad_250m_grid')
        centroids = np.array([t.centroid.coords for t in divs.centroid()])
        num_on_grid = sum([np.sum(np.sum(new_xy == t, axis=1) == 2)==1 for t in centroids])
        self.assertEqual(num_on_grid, 0)

    def tearDown(self):
        Division.objects.all().delete()
        DivisionType.objects.all().delete()


class TestHotspot(unittest.TestCase):

    def test_stkernel_bowers(self):
        a = 2
        b = 10
        stk = hotspot.STKernelBowers(a, b)
        data = np.tile(np.arange(10), (3, 1)).transpose()
        stk.train(data)
        self.assertEqual(stk.data.ndata, 10)
        self.assertEqual(stk.data.nd, 3)
        z = stk.predict([[10, 10, 10]])
        zt_expct = 1. / (1. + a * np.arange(1, 11))
        zd_expct = 1. / (1. + b * np.sqrt(2) * np.arange(1, 11))
        z_expct = sum(zt_expct * zd_expct)
        self.assertAlmostEqual(z, z_expct)

    def test_hotspot(self):
        stk = mock.create_autospec(hotspot.STKernelBowers)

        data = np.tile(np.arange(10), (3, 1)).transpose()
        h = hotspot.Hotspot(stk, data=data)
        self.assertEqual(stk.train.call_count, 1)
        self.assertListEqual(list(stk.train.call_args[0][0].flat), list(data.flat))

        a = 2
        b = 10
        stk = hotspot.STKernelBowers(a, b)
        h = hotspot.Hotspot(stk, data=data)
        self.assertEqual(h.predict([[1.3, 4.6, 7.8]]), stk.predict([[1.3, 4.6, 7.8]]))



