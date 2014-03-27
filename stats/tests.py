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


UKTEMP = os.path.join(os.path.dirname(__file__), 'test_data', 'uk_avg_temp.csv')
METREGIONS = os.path.join(os.path.dirname(__file__), 'test_data/met_office_regions', 'met_office_regions.shp')

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


class SpatialStats(TestCase):

    def setUp(self):
        # met regions
        dt = models.DivisionType(name='met_region')
        dt.save()

        def pre_save_callback(sender, instance, *args, **kwargs):
            instance.type = dt

        mapping = {
            'name': 'NAME',
            # 'code': 'ID',
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
        W_rook = logic.rook_connectivity(regions_qset)
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
        W_queen = logic.boolean_connectivity(regions_qset)
        for row in W_queen.values:
            self.assertAlmostEqual(sum(row), 1.0)

        W_expct = np.zeros((n, n))
        W_expct[0, 1] = W_expct[0, 3] = W_expct[0, 4] = 1
        W_expct[1, 0] = W_expct[1, 2] = W_expct[1, 4] = W_expct[1, 3] = W_expct[1, 5] = 1
        W_expct[2, 1] = W_expct[2, 5] = W_expct[2, 4] = 1
        W_expct[3, 0] = W_expct[3, 4] = W_expct[3, 6] = W_expct[3, 1] = W_expct[3, 7] = 1
        W_expct[4, 1] = W_expct[4, 3] = W_expct[4, 5] = W_expct[4, 7] = W_expct[4, 0] = W_expct[4, 2] = W_expct[4, 6] = W_expct[4, 8] = 1
        W_expct[5, 2] = W_expct[5, 4] = W_expct[5, 8] = W_expct[5, 1] = W_expct[5, 7] = 1
        W_expct[6, 3] = W_expct[6, 7] = W_expct[6, 4] = 1
        W_expct[7, 6] = W_expct[7, 4] = W_expct[7, 8] = W_expct[7, 3] = W_expct[7, 5] = 1
        W_expct[8, 5] = W_expct[8, 7] = W_expct[8, 9] = W_expct[8, 4] = 1
        W_expct[9, 8] = W_expct[9, 5] = 1

        for i in range(n):
            W_expct[i] = W_expct[i]/np.sum(W_expct[i])
            print i
            try:
                self.assertListEqual(list(W_expct[i]), list(W_queen.values[i]))
            except Exception:
                pdb.set_trace()

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
        # pdb.set_trace()
        #
        # for i in range(n):
        #     W_expct[i] = W_expct[i]/np.sum(W_expct[i])
        #     self.assertListEqual(list(W_expct[i]), list(W_bool[i]))