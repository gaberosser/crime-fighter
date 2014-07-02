__author__ = 'gabriel'
import unittest
import csv
import pickle
import numpy as np
import mock
import math
from django.contrib.gis import geos
import spatial
import hotspot
import validation
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
        self.assertTupleEqual(stk.data.shape, (10, 3))
        z = stk.predict(10, 10, 10)
        zt_expct = np.sum(1 / (1 + a * np.arange(1, 11)))
        zd_expct = np.sum(1 / (1 + b * np.sqrt(2) * np.arange(1, 11)))
        self.assertAlmostEqual(z, zt_expct * zd_expct)

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
        self.assertEqual(h.predict(1.3, 4.6, 7.8), stk.predict(1.3, 4.6, 7.8))


class TestValidation(unittest.TestCase):

    def setUp(self):
        n = 100
        t = np.linspace(0, 1, n).reshape((n, 1))
        np.random.shuffle(t)
        self.data = np.hstack((t, np.random.RandomState(42).rand(n, 2)))
        stk = hotspot.SKernelHistoric(1, bdwidth=0.3)
        self.vb = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))
        self.vb.train_model()

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

    def test_instantiation(self):
        # check that model is NOT trained initially
        stk = mock.create_autospec(hotspot.STKernelBowers)
        vb = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))
        self.assertEqual(stk.train.call_count, 0)

        # check data are present and sorted
        self.assertEqual(self.vb.ndata, self.data.shape[0])
        self.assertTrue(np.all(np.diff(self.vb.data[:, 0]) > 0))

    def test_grid(self):

        #  no grid length provided
        self.assertEqual(self.vb.spatial_domain, None)
        with self.assertRaises(Exception):
            self.vb.grid
        self.vb.set_grid(0.1)

        # check that spatial extent is now set correctly
        poly = self.vb.spatial_domain
        self.assertTupleEqual(poly.extent, (
            min(self.data[:, 1]),
            min(self.data[:, 2]),
            max(self.data[:, 1]),
            max(self.data[:, 2]),
        ))
        self.assertEqual(len(self.vb.grid), 100)

    def test_time_cutoff(self):

        # check time cutoff and training/test split
        data = self.data[np.argsort(self.data[:, 0])]
        cutoff_te = data[int(self.data.shape[0] / 2), 0]
        self.assertEqual(self.vb.cutoff_t, cutoff_te)
        # training set
        self.assertEqual(self.vb.training.shape[0], sum(self.data[:, 0] <= cutoff_te))
        for i in range(int(self.data.shape[0] / 2)):
            self.assertListEqual(list(self.vb.training[i]), list(data[i]))
        # testing set
        testinge = data[data[:, 0] > cutoff_te]
        self.assertEqual(self.vb.testing().shape[0], len(testinge))
        for i in range(len(testinge)):
            self.assertListEqual(list(self.vb.testing()[i]), list(testinge[i]))

        # testing dataset with dt_plus specified
        dt_plus = data[data[:, 0] > cutoff_te][2, 0] - cutoff_te
        testinge = data[(data[:, 0] > cutoff_te) & (data[:, 0] <= (cutoff_te + dt_plus))]
        self.assertEqual(self.vb.testing(dt_plus=dt_plus).shape[0], len(testinge))
        for i in range(len(testinge)):
            self.assertListEqual(list(self.vb.testing(dt_plus=dt_plus)[i]), list(testinge[i]))

        # testing dataset with dt_plus and dt_minus
        dt_plus = data[data[:, 0] > cutoff_te][17, 0] - cutoff_te
        dt_minus = data[data[:, 0] > cutoff_te][10, 0] - cutoff_te
        testinge = data[(data[:, 0] > (cutoff_te + dt_minus)) & (data[:, 0] <= (cutoff_te + dt_plus))]
        self.assertEqual(self.vb.testing(dt_plus=dt_plus, dt_minus=dt_minus).shape[0], len(testinge))
        for i in range(len(testinge)):
            self.assertListEqual(list(self.vb.testing(dt_plus=dt_plus, dt_minus=dt_minus)[i]), list(testinge[i]))

    def test_training(self):
        stk = mock.create_autospec(hotspot.STKernelBowers)
        vb = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))
        self.assertEqual(stk.train.call_count, 0)
        vb.set_t_cutoff(vb.cutoff_t)
        self.assertEqual(stk.train.call_count, 1)
        vb.train_model()
        self.assertEqual(stk.train.call_count, 2)

    def test_predict(self):
        self.vb.set_grid(0.1)
        self.vb.train_model()
        pop = self.vb.predict_on_poly(self.vb.cutoff_t, self.vb.spatial_domain)
        centroid = self.vb.spatial_domain.centroid.coords
        pope = self.vb.model.predict(self.vb.cutoff_t, *centroid)
        self.assertEqual(pop, pope)

        stk = mock.create_autospec(hotspot.STKernelBowers)
        vb = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))
        vb.train_model()
        vb.set_grid(0.1)
        pop = vb.predict(self.vb.cutoff_t)
        # FIXME: next assertion may break if we implement a more efficient routine for eval (reducing num of calls)
        self.assertEqual(stk.predict.call_count, len(vb.grid))

        class MockKernel(hotspot.STKernelBase):

            def _evaluate(self, t, x, y):
                return sum(self.data[:, 1] - x)

        stk = MockKernel()
        vb = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))
        self.assertEqual(stk.ndata, 0)
        vb.train_model()
        vb.set_grid(0.1)
        pop = vb.predict(self.vb.cutoff_t)
        self.assertEqual(len(pop), len(vb.grid))
        pope = np.array([np.sum(self.data[self.data[:, 0] <= self.vb.cutoff_t, 1] - x.centroid.coords[0]) for x in vb.grid])
        for (p, pe) in zip(pop, pope):
            self.assertAlmostEqual(p, pe)

    def test_true_values(self):
        self.vb.set_grid(0.1)
        self.vb.train_model()
        polys = self.vb.grid

        # check computing true values
        true = self.vb.true_values(0.2, 0)
        testing_idx = (self.data[:, 0] > self.vb.cutoff_t) & (self.data[:, 0] <= (self.vb.cutoff_t + 0.2))
        testing = [geos.Point(x[1], x[2]) for x in self.data[testing_idx]]
        truee = []
        for p in polys:
            truee.append(sum([p.intersects(t) for t in testing]))
        truee = np.array(truee)
        self.assertEqual(len(true), len(truee))
        self.assertListEqual(list(true), list(truee))

    def test_assess(self):
        self.vb.set_grid(0.1)
        self.vb.train_model()

        polys, pred, carea, cfrac, pai = self.vb.predict_assess(dt_plus=0.2)

        # check pred and polys
        self.assertEqual(len(pred), len(self.vb.grid))

        stk = self.vb.model
        prede = np.array([stk.predict(self.vb.cutoff_t + 0.2, *p.centroid.coords) for p in self.vb.grid])
        sort_idx = np.argsort(prede)[::-1]
        polyse = [self.vb.grid[i] for i in sort_idx]
        self.assertEqual(len(polys), len(self.vb.grid))
        for (p, pe) in zip(polys, polyse):
            self.assertTupleEqual(p.coords, pe.coords)

        prede = prede[sort_idx]
        for (p, pe) in zip(pred, prede):
            self.assertAlmostEqual(p, pe)

        # check carea
        ae = np.array([p.area for p in polyse])
        self.assertAlmostEqual(self.vb.A, sum(ae))
        careae = np.cumsum(ae) / sum(ae)
        for (p, pe) in zip(carea, careae):
            self.assertAlmostEqual(p, pe)

        # check cfrac
        true = self.vb.true_values(dt_plus=0.2, dt_minus=0)
        cfrace = np.cumsum(true[sort_idx]) / np.sum(true)
        for (p, pe) in zip(cfrac, cfrace):
            self.assertAlmostEqual(p, pe)

        # check pai
        paie = cfrace * sum(ae) / np.cumsum(ae)
        for (p, pe) in zip(pai, paie):
            self.assertAlmostEqual(p, pe)

    def test_run(self):
        with mock.patch.object(validation.ValidationBase, 'predict_assess',
                               return_value=tuple([0 for i in range(5)])) as m:
            stk = hotspot.SKernelHistoric(1, bdwidth=0.3)
            vb = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))
            t0 = vb.cutoff_t
            polys, carea, cfrac, pai = vb.run(dt=0.1)
            expct_call_count = math.ceil((1 - vb.cutoff_t) / 0.1)
            self.assertEqual(m.call_count, expct_call_count)
            for c in m.call_args_list:
                self.assertEqual(c, mock.call(dt_plus=0.1, dt_minus=0))
            # check that cutoff time is reset correctly
            self.assertEqual(vb.cutoff_t, t0)

        self.vb.set_grid(0.1)
        polys, carea, cfrac, pai = self.vb.run(dt=0.1)
        self.assertEqual(len(polys[0]), len(self.vb.grid))
        self.assertEqual(len(carea), expct_call_count)
        self.assertEqual(len(cfrac), expct_call_count)
        self.assertEqual(len(pai), expct_call_count)
