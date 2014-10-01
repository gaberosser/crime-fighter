__author__ = 'gabriel'
import logic
import numpy as np
import csv
import os
import pdb
from statsmodels.api import tsa
from django.test import TestCase
from database import models
from django.db.models.signals import pre_save
from django.contrib.gis.utils import LayerMapping
from django.contrib.gis.geos import Polygon, MultiPolygon
from pandas import Series, DataFrame


UKTEMP = os.path.join(os.path.dirname(__file__), 'test_data', 'uk_avg_temp.csv')
METREGIONS = os.path.join(os.path.dirname(__file__), 'test_data/met_office_regions', 'met_office_regions.shp')


class BasicStats(TestCase):

    def test_weighted_stdev(self):
        # 1D data

        # weights all unity
        data = np.linspace(0, 1, 11)
        weights = np.ones_like(data)
        # test the weighted stdev is equal to the usual UNBIASED estimator
        sw = logic.weighted_stdev(data, weights)
        self.assertIsInstance(sw, float)
        self.assertEqual(sw, np.std(data, ddof=1))

        # weights all equal
        weights *= np.pi
        # test the weighted stdev is equal to the usual UNBIASED estimator
        self.assertEqual(logic.weighted_stdev(data, weights), np.std(data, ddof=1))

        # two source distributions (weights differ)
        n = 1000000
        weighted_data = np.hstack((np.random.RandomState(42).randn(n) - 1, np.random.RandomState(42).randn(n) + 1))
        weights = np.hstack((np.ones(n) * 2 / 3., np.ones(n) / 3.))
        sw = logic.weighted_stdev(weighted_data, weights)

        # analytic variance is 1 + 8/9
        self.assertAlmostEqual(sw, np.sqrt(1 + 8/9.), places=2)  # 2DP

        # 2D data
        data = np.tile(np.linspace(0, 1, 11).reshape(11, 1), (1, 2))
        weights = np.ones(11)
        sw = logic.weighted_stdev(data, weights)
        self.assertEqual(sw.size, 2)
        self.assertEqual(sw[0], sw[1])


class TemporalStats(TestCase):

    def setUp(self):
        with open(UKTEMP, 'r') as f:
            c = csv.reader(f, delimiter=',')
            self.fields = c.next()
            self.data = np.array(list(c), dtype=float)

    def test_acf(self):
        x = self.data[:, 0]
        t_acf = logic.truncated_acf_1d(x, max_lag=36)
        c_acf = logic.conv_acf_1d(x, max_lag=36)
        sm_acf = tsa.stattools.acf(x, nlags=35, unbiased=True)
        for t in zip(t_acf, c_acf, sm_acf):
            self.assertAlmostEqual(t[0], t[1])
            self.assertAlmostEqual(t[1], t[2])

    def test_p_acf(self):
        x = self.data[:, 0]
        c_acf = logic.conv_acf_1d(x, max_lag=36)
        pacf = logic.yw_pacf_1d(c_acf)
        sm_pacf = tsa.pacf(x, nlags=35)
        for t in zip(pacf, sm_pacf):
            self.assertAlmostEqual(t[0], t[1])


class SpatialConnectivity(TestCase):

    def setUp(self):
        # met regions
        dt = models.DivisionType(name='met_region')
        dt.save()

        def pre_save_callback(sender, instance, *args, **kwargs):
            instance.type = dt

        mapping = {
            'name': 'NAME',
            'mpoly': 'MULTIPOLYGON',
        }

        lm = LayerMapping(models.Division, METREGIONS, mapping, transform=False)
        pre_save.connect(pre_save_callback, sender=models.Division)
        try:
            lm.save(strict=True, verbose=False)
        except Exception as exc:
            print repr(exc)
            raise
        finally:
            pre_save.disconnect(pre_save_callback, sender=models.Division)

        # toy regions
        dt = models.DivisionType(name='toy')
        dt.save()

        coords = [
            [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)],
            [(2, 1), (3, 1), (3, 2), (2, 2), (2, 1)],
            [(3, 1), (4, 1), (4, 2), (3, 2), (3, 1)],
            [(1, 2), (2, 2), (2, 3), (1, 3), (1, 2)],
            [(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)],
            [(3, 2), (4, 2), (4, 3), (3, 3), (3, 2)],
            [(1, 3), (2, 3), (2, 4), (1, 4), (1, 3)],
            [(2, 3), (3, 3), (3, 4), (2, 4), (2, 3)],
            [(3, 3), (4, 3), (4, 4), (3, 4), (3, 3)],
            [(4, 3), (5, 3), (5, 4), (4, 4), (4, 3)],
        ]
        for (i, c) in enumerate(coords):
            mpoly = MultiPolygon(Polygon(c))
            models.Division(name="%02u" % (i+1), code=str(i), mpoly=mpoly, type=dt).save()

    def test_toy_connectivity(self):

        regions_qset = models.Division.objects.filter(type='toy')
        n = regions_qset.count()
        self.assertEqual(n, 10)

        # rook connectivity matrix
        W_rook = logic.rook_boolean_connectivity(regions_qset)
        W_rook.sort_index(inplace=True)
        for row in W_rook.values:
            self.assertAlmostEqual(sum(row), 1.0)
            
        W_expct = np.zeros((n, n))
        W_expct[0, 1] = W_expct[0, 3] = 1
        W_expct[1, 0] = W_expct[1, 2] = W_expct[1, 4] = 1
        W_expct[2, 1] = W_expct[2, 5] = 1
        W_expct[3, 0] = W_expct[3, 4] = W_expct[3, 6] = 1
        W_expct[4, 1] = W_expct[4, 3] = W_expct[4, 5] = W_expct[4, 7] = 1
        W_expct[5, 2] = W_expct[5, 4] = W_expct[5, 8] = 1
        W_expct[6, 3] = W_expct[6, 7] = 1
        W_expct[7, 6] = W_expct[7, 4] = W_expct[7, 8] = 1
        W_expct[8, 5] = W_expct[8, 7] = W_expct[8, 9] = 1
        W_expct[9, 8] = 1

        for i in range(n):
            W_expct[i] = W_expct[i]/np.sum(W_expct[i])
            self.assertListEqual(list(W_expct[i]), list(W_rook.values[i]))

        # queen connectivity matrix
        W_queen = logic.intersection_boolean_connectivity(regions_qset)
        W_queen.sort_index(inplace=True)
        for row in W_queen.values:
            self.assertAlmostEqual(sum(row), 1.0)

        W_expct = np.zeros((n, n))
        W_expct[0, 1] = W_expct[0, 3] = W_expct[0, 4] = 1
        W_expct[1, 0] = W_expct[1, 2] = W_expct[1, 4] = W_expct[1, 3] = W_expct[1, 5] = 1
        W_expct[2, 1] = W_expct[2, 5] = W_expct[2, 4] = 1
        W_expct[3, 0] = W_expct[3, 4] = W_expct[3, 6] = W_expct[3, 1] = W_expct[3, 7] = 1
        W_expct[4, 1] = W_expct[4, 3] = W_expct[4, 5] = W_expct[4, 7] = W_expct[4, 0] = W_expct[4, 2] = W_expct[4, 6] = W_expct[4, 8] = 1
        W_expct[5, 2] = W_expct[5, 4] = W_expct[5, 8] = W_expct[5, 1] = W_expct[5, 7] = W_expct[5, 9] = 1
        W_expct[6, 3] = W_expct[6, 7] = W_expct[6, 4] = 1
        W_expct[7, 6] = W_expct[7, 4] = W_expct[7, 8] = W_expct[7, 3] = W_expct[7, 5] = 1
        W_expct[8, 5] = W_expct[8, 7] = W_expct[8, 9] = W_expct[8, 4] = 1
        W_expct[9, 8] = W_expct[9, 5] = 1

        for i in range(n):
            W_expct[i] = W_expct[i]/np.sum(W_expct[i])
            self.assertListEqual(list(W_expct[i]), list(W_queen.values[i]))

        # regions_qset = models.Division.objects.filter(type='met_region')
        # n = regions_qset.count()
        # self.assertEqual(n, 9)
        # W_met = logic.intersection_boolean_connectivity(regions_qset)
        #
        # W_expct = np.zeros((n, n))
        # # EngNE
        # W_expct[0, 1] = W_expct[0, 3] = W_expct[0, 5] = W_expct[0, 6] = 1
        # W_expct[1, 0] = W_expct[3, 0] = W_expct[5, 0] = W_expct[6, 0] = 1
        # # EA
        # W_expct[1, 2] = W_expct[1, 3] = 1
        # W_expct[2, 1] = W_expct[3, 1] = 1
        # # EngSE
        # W_expct[2, 3] = W_expct[2, 4] = 1
        # W_expct[3, 2] = W_expct[4, 2] = 1
        # # Mid
        # W_expct[3, 4] = W_expct[3, 5] = 1
        # W_expct[4, 3] = W_expct[5, 3] = 1
        # # EngSW
        # W_expct[4, 5] = 1
        # W_expct[5, 4] = 1
        # # EngNW
        # W_expct[5, 7] = W_expct[5, 6] = 1
        # W_expct[6, 5] = W_expct[7, 5] = 1
        # # ScoE
        # W_expct[6, 7] = W_expct[6, 8] = 1
        # W_expct[7, 6] = W_expct[8, 6] = 1
        # # ScoW
        # W_expct[7, 8] = 1
        # W_expct[8, 7] = 1
        #
        # for i in range(n):
        #     W_expct[i] = W_expct[i]/np.sum(W_expct[i])
        #     self.assertListEqual(list(W_expct[i]), list(W_met.values[i]))


class SpatialAnalysis(TestCase):

    def setUp(self):
        # met regions
        dt = models.DivisionType(name='toy')
        dt.save()

        coords = [
            [(2, 1), (3, 1), (3, 2), (2, 2), (2, 1)],
            [(1, 2), (2, 2), (2, 3), (1, 3), (1, 2)],
            [(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)],
            [(3, 2), (4, 2), (4, 3), (3, 3), (3, 2)],
            [(2, 3), (3, 3), (3, 4), (2, 4), (2, 3)],
        ]

        for (i, c) in enumerate(coords):
            mpoly = MultiPolygon(Polygon(c))
            models.Division(name="%02u" % (i+1), code=str(i), mpoly=mpoly, type=dt).save()

    def test_local_global_i(self):

        regions_qset = models.Division.objects.filter(type='toy')
        # rook connectivity matrix
        W_rook = logic.rook_boolean_connectivity(regions_qset)

        # perfect anticorrelation
        data = Series([0, 0, 1., 0, 0], index=["%02u" % (i+1) for i in range(5)])

        global_i = logic.global_morans_i(data, W_rook)
        self.assertAlmostEqual(global_i, -1.0)

        local_i = logic.local_morans_i(data, W_rook)
        for x in local_i.values:
            self.assertAlmostEqual(x, -0.16)

        z = data - data.mean()
        self.assertAlmostEqual(local_i.sum(), global_i*sum(z**2))

        # perfect anticorrelation
        data = Series([1., 1., 0., 1., 1.], index=["%02u" % (i+1) for i in range(5)])

        global_i = logic.global_morans_i(data, W_rook)
        self.assertAlmostEqual(global_i, -1.0)

        local_i = logic.local_morans_i(data, W_rook)
        for x in local_i.values:
            self.assertAlmostEqual(x, -0.16)

        z = data - data.mean()
        self.assertAlmostEqual(local_i.sum(), global_i*sum(z**2))

        # no correlation
        data = Series([1., 0, 0.5, 0, 1], index=["%02u" % (i+1) for i in range(5)])

        global_i = logic.global_morans_i(data, W_rook)
        self.assertAlmostEqual(global_i, 0.0)

        local_i = logic.local_morans_i(data, W_rook)
        for x in local_i.values:
            self.assertAlmostEqual(x, 0.0)

        # some correlation
        data = Series([1., 0, 1, 0, 1], index=["%02u" % (i+1) for i in range(5)])

        global_i = logic.global_morans_i(data, W_rook)
        self.assertAlmostEqual(global_i, -1/6.)

        local_i = logic.local_morans_i(data, W_rook)
        expct_local_i = [4/25., -6/25., -1/25., -6/25., 4/25.]

        for x, y in zip(local_i.values, expct_local_i):
            self.assertAlmostEqual(x, y)

        z = data - data.mean()
        self.assertAlmostEqual(local_i.sum(), global_i*sum(z**2))