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

    def test_connectivity(self):
        regions_qset = models.Division.objects.all()
        n = regions_qset.count()
        self.assertEqual(n, 9)
        W_bool = logic.boolean_connectivity(regions_qset)
        for row in W_bool:
            self.assertAlmostEqual(sum(row), 1.0)
            
        W_expct = np.zeros((n, n))
        # EngNE
        W_expct[0, 1] = W_expct[0, 3] = W_expct[0, 5] = W_expct[0, 6] = 1
        W_expct[1, 0] = W_expct[3, 0] = W_expct[5, 0] = W_expct[6, 0] = 1
        # EA
        W_expct[1, 2] = W_expct[1, 3] = 1
        W_expct[2, 1] = W_expct[3, 1] = 1
        # EngSE
        W_expct[2, 3] = W_expct[2, 4] = 1
        W_expct[3, 2] = W_expct[4, 2] = 1
        # Mid
        W_expct[3, 4] = W_expct[3, 5] = 1
        W_expct[4, 3] = W_expct[5, 3] = 1
        # EngSW
        W_expct[4, 5] = 1
        W_expct[5, 4] = 1
        # EngNW
        W_expct[5, 7] = W_expct[5, 6] = 1
        W_expct[6, 5] = W_expct[7, 5] = 1
        # ScoE
        W_expct[6, 7] = W_expct[6, 8] = 1
        W_expct[7, 6] = W_expct[8, 6] = 1
        # ScoW
        W_expct[7, 8] = 1
        W_expct[8, 7] = 1

        for i in range(n):
            W_expct[i] = W_expct[i]/np.sum(W_expct[i])
            self.assertListEqual(list(W_expct[i]), list(W_bool[i]))