__author__ = 'gabriel'
import validation
import hotspot
import roc
from data.models import DataArray
import unittest
import numpy as np
import mock
import math
import collections
from shapely import geometry


class TestRoc(unittest.TestCase):

    def setUp(self):
        n = 100
        t = np.linspace(0, 1, n).reshape((n, 1))
        np.random.shuffle(t)
        self.udata = np.hstack((t, np.random.RandomState(42).rand(n, 2)))
        self.data = np.array([
            [0.25, 0.25],
            [0.49, 0.25],
            [0.51, 0.25],
            [1.01, 0.25],
            [0.25, 0.49],
            [0.75, 0.75],
            [0.75, 0.75],
            [0.75, 0.75],
        ])

    def test_instantiation(self):
        r = roc.RocSpatialGrid()
        with self.assertRaises(AttributeError):
            r.ngrid
        with self.assertRaises(AttributeError):
            r.ndata
        with self.assertRaises(Exception):
            r.true_count
        with self.assertRaises(AttributeError):
            r.set_data(self.udata)
        r.set_data(self.udata[:, 1:])
        self.assertListEqual(list(r.data[:, 0]), list(self.udata[:, 1]))

    def test_grid_no_poly(self):
        # no spatial domain supplied
        r = roc.RocSpatialGrid(data=self.udata[:, 1:])
        r.set_grid(0.1)
        self.assertTupleEqual(r.poly.bounds, (
            min(self.udata[:, 1]),
            min(self.udata[:, 2]),
            max(self.udata[:, 1]),
            max(self.udata[:, 2]),
        ))

        self.assertEqual(r.ngrid, 100)
        self.assertEqual(max(np.array(r.egrid)[:, 2]), 1.0)
        self.assertEqual(max(np.array(r.egrid)[:, 3]), 1.0)
        self.assertEqual(sum(np.array([x.area for x in r.igrid]) > 0.00999), 64) # 8 x 8 centre grid

        # different arrangement
        r = roc.RocSpatialGrid(data=self.data)
        r.set_grid(0.5)
        self.assertEqual(r.ngrid, 6)
        areas = sorted([x.area for x in r.igrid])
        areas_expctd = [0.0025, 0.0025, 0.0625, 0.0625, 0.125, 0.125]
        for a, ae in zip(areas, areas_expctd):
            self.assertAlmostEqual(a, ae)

    def test_sample_points(self):
        # RocSpatialGrid
        r = roc.RocSpatialGrid(data=self.data)
        r.set_grid(0.05)
        self.assertTrue(np.all(r.sample_points[:, 0] == r.centroids[:, 0]))
        self.assertTrue(np.all(r.sample_points[:, 1] == r.centroids[:, 1]))

        # RocSpatialGridMonteCarloIntegration
        r = roc.RocSpatialGridMonteCarloIntegration(data=self.data)
        r.set_grid(0.05, 10)  # 10 sample points per grid square

        for i in range(r.ngrid):
            xmin, ymin, xmax, ymax = r.egrid[i]
            self.assertTrue(np.all(r.sample_points.toarray(0)[:, i] > xmin))
            self.assertTrue(np.all(r.sample_points.toarray(0)[:, i] < xmax))
            self.assertTrue(np.all(r.sample_points.toarray(1)[:, i] > ymin))
            self.assertTrue(np.all(r.sample_points.toarray(1)[:, i] < ymax))

    def test_true_count(self):
        r = roc.RocSpatialGrid(data=self.data)
        r.set_grid(0.5)
        tc = r.true_count
        self.assertEqual(len(tc), r.ngrid)
        self.assertEqual(sum(tc), self.data.shape[0])
        expctd_count = {
            (0.26, 0.26): 3,
            (0.75, 0.26): 1,
            (1.005, 0.26): 1,
            (0.26, 0.74): 0,
            (0.75, 0.74): 3,
            (1.005, 0.74): 0,
        }
        for k, v in expctd_count.items():
            # find correct grid square
            idx = [i for i, x in enumerate(r.igrid) if x.intersects(geometry.Point(k))][0]
            # check count
            self.assertEqual(tc[idx], v)

    def test_prediction(self):
        r = roc.RocSpatialGrid(data=self.data)
        r.set_grid(0.5)
        tc = r.true_count

        pred = np.linspace(0, 1, r.ngrid).reshape((1, r.ngrid))
        # check that an error is raised if the incorrect quantity of data is supplied
        with self.assertRaises(AttributeError):
            r.set_prediction(pred[:, 1:])
        r.set_prediction(pred)
        self.assertTrue(np.all(pred == r.prediction_values))
        self.assertListEqual(list(r.prediction_rank), range(r.ngrid)[::-1])

    def test_evaluate(self):
        r = roc.RocSpatialGrid(data=self.data)
        r.set_grid(0.5)
        with self.assertRaises(AttributeError):
            res = r.evaluate()
        r.set_prediction(np.linspace(0, 1, r.ngrid).reshape((1, r.ngrid)))
        res = r.evaluate()
        self.assertListEqual(list(res['prediction_rank']), range(r.ngrid)[::-1])
        self.assertListEqual(list(res['prediction_values']), list(r.prediction_values[::-1])) # sorted descending

        cumul_area_expctd = np.cumsum([x.area for x in r.igrid][::-1])
        cumul_area_expctd /= cumul_area_expctd[-1]
        self.assertListEqual(list(res['cumulative_area']), list(cumul_area_expctd))

        cumul_crime_expctd = np.cumsum(r.true_count[::-1])
        cumul_crime_expctd = cumul_crime_expctd / float(cumul_crime_expctd[-1])  ## FIXME: why can't we use /= here??
        self.assertListEqual(list(res['cumulative_crime']), list(cumul_crime_expctd))

        cumulative_crime_max_expctd = np.cumsum(np.sort(r.true_count)[::-1]) / float(sum(r.true_count))
        self.assertListEqual(list(res['cumulative_crime_max']), list(cumulative_crime_max_expctd))

        ## TODO: test pai too


class TestValidation(unittest.TestCase):

    def setUp(self):
        n = 100
        t = np.linspace(0, 1, n).reshape((n, 1))
        self.data = np.hstack((t, np.random.RandomState(42).rand(n, 2)))
        stk = hotspot.SKernelHistoric(1, bdwidth=0.3)

    def test_mcsampler(self):
        poly = geometry.Polygon([
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
            self.assertTrue(geometry.Point(list(r)).intersects(poly))

        # square domain so every rand() call should have generated a valid RV
        self.assertEqual(m.call_count, 20)

        # triangular poly
        poly = geometry.Polygon([
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 0),
        ])

        mcs = validation.mc_sampler(poly)
        with mock.patch('numpy.random.random', side_effect=np.random.RandomState(42).rand) as m:
            rvs = np.array([mcs.next() for i in range(10)])

        for r in rvs:
            self.assertTrue(geometry.Point(list(r)).intersects(poly))

        ncalls = m.call_count
        x_min, y_min, x_max, y_max = poly.bounds
        expct_draws = np.random.RandomState(42).rand(ncalls).reshape((ncalls / 2, 2)) * np.array([x_max-x_min, y_max-y_min]) + \
            np.array([x_min, y_min])
        in_idx = np.where([geometry.Point(list(x)).intersects(poly) for x in expct_draws])[0]
        self.assertEqual(len(in_idx), 10)
        for x, y in zip(expct_draws[in_idx, :], rvs):
            self.assertListEqual(list(x), list(y))

    def test_instantiation(self):

        # shuffle data to check that it is resorted
        data = np.array(self.data)
        np.random.shuffle(data)

        stk = hotspot.SKernelHistoric(1, bdwidth=0.3)
        vb = validation.ValidationBase(data, hotspot.Hotspot, model_args=(stk,))

        # check data are present and sorted
        self.assertEqual(vb.ndata, self.data.shape[0])
        self.assertTrue(np.all(np.diff(vb.data[:, 0]) > 0))
        self.assertTrue(np.all(vb.data == self.data))

        # no grid present yet as not supplied
        self.assertTrue(vb.roc.poly is None)

        # instantiate grid
        vb.set_grid(0.1)

        # should now have automatically made a spatial domain
        self.assertListEqual(list(vb.roc.poly.bounds), [
            min(self.data[:, 1]),
            min(self.data[:, 2]),
            max(self.data[:, 1]),
            max(self.data[:, 2]),
        ])

        # repeat but this time copy the grid
        vb2 = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))
        self.assertTrue(vb2.roc.poly is None)
        vb2.set_grid(vb.roc)

        self.assertListEqual(vb.roc.egrid, vb2.roc.egrid)

    def test_time_cutoff(self):

        stk = hotspot.SKernelHistoric(1, bdwidth=0.3)
        vb = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))

        # expected cutoff automatically chosen
        cutoff_te = vb.data[int(self.data.shape[0] / 2), 0]

        # check time cutoff and training/test split
        self.assertEqual(vb.cutoff_t, cutoff_te)

        # training set
        training_expctd = self.data[self.data[:, 0] <= cutoff_te]
        self.assertEqual(vb.training.ndata, len(training_expctd))
        self.assertTrue(np.all(vb.training == training_expctd))

        # testing set
        testing_expctd = self.data[self.data[:, 0] > cutoff_te]
        self.assertEqual(vb.testing().ndata, len(testing_expctd))
        self.assertTrue(np.all(vb.testing() == testing_expctd))

        # test when as_point=True
        tst = vb.testing(as_point=True)
        self.assertEqual(len(tst), len(testing_expctd))
        for i in range(len(testing_expctd)):
            self.assertEqual(tst[i][0], testing_expctd[i, 0])
            self.assertIsInstance(tst[i][1], geometry.Point)
            self.assertListEqual([tst[i][1].x, tst[i][1].y], list(testing_expctd[i, 1:]))

        # change cutoff_t
        cutoff_te = 0.3
        vb.set_t_cutoff(cutoff_te)
        testing_expctd = self.data[self.data[:, 0] > cutoff_te]
        self.assertEqual(vb.testing().ndata, len(testing_expctd))
        self.assertTrue(np.all(vb.testing() == testing_expctd))

        # testing dataset with dt_plus specified
        dt_plus = testing_expctd[2, 0] - cutoff_te
        testing_expctd = testing_expctd[:3]  # three results
        self.assertEqual(vb.testing(dt_plus=dt_plus).ndata, len(testing_expctd))
        self.assertTrue(np.all(vb.testing(dt_plus=dt_plus) == testing_expctd))

        # testing dataset with dt_plus and dt_minus
        dt_plus = self.data[self.data[:, 0] > cutoff_te][17, 0] - cutoff_te
        dt_minus = self.data[self.data[:, 0] > cutoff_te][10, 0] - cutoff_te
        testing_expctd = self.data[(self.data[:, 0] > (cutoff_te + dt_minus)) & (self.data[:, 0] <= (cutoff_te + dt_plus))]
        self.assertEqual(vb.testing(dt_plus=dt_plus, dt_minus=dt_minus).ndata, 7)
        self.assertEqual(vb.testing(dt_plus=dt_plus, dt_minus=dt_minus).ndata, len(testing_expctd))
        self.assertTrue(np.all(vb.testing(dt_plus=dt_plus, dt_minus=dt_minus) == testing_expctd))

    def test_training(self):

        stk = mock.create_autospec(hotspot.STKernelBowers)
        vb = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))
        # check that model is NOT trained initially
        self.assertEqual(stk.train.call_count, 0)

        # spoof train
        vb.train_model()
        self.assertEqual(stk.train.call_count, 1)
        self.assertEqual(len(stk.train.call_args[0]), 1)
        self.assertTrue(np.all(stk.train.call_args[0][0] == vb.training))

        vb.train_model()
        self.assertEqual(stk.train.call_count, 2)

        # set cutoff time, no training
        vb.set_t_cutoff(0.1, b_train=False)
        self.assertEqual(stk.train.call_count, 2)

        # set cutoff time, training
        vb.set_t_cutoff(0.1, b_train=True)
        self.assertEqual(stk.train.call_count, 3)
        self.assertEqual(len(stk.train.call_args[0]), 1)
        self.assertTrue(np.all(stk.train.call_args[0][0] == vb.training))

    def test_predict(self):

        stk = hotspot.SKernelHistoric(1, bdwidth=0.3)
        vb = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))
        vb.train_model()
        vb.set_grid(0.1)

        # spoof predict at all grid centroids
        stk = mock.create_autospec(hotspot.STKernelBowers)
        vb.model = stk
        pop = vb.predict(vb.cutoff_t)

        self.assertEqual(stk.predict.call_count, 1)
        sp = vb.sample_points
        pred_call_arg = DataArray.from_args(
            vb.cutoff_t * np.ones(sp.ndata),
            sp.toarray(0),
            sp.toarray(1),
        )
        self.assertTrue(np.all(stk.predict.call_args[0] == pred_call_arg))


    def test_assess(self):

        stk = hotspot.SKernelHistoric(1, bdwidth=0.3)
        vb = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))
        vb.train_model()

        # mock roc object with grid
        mocroc = mock.create_autospec(roc.RocSpatialGrid)
        mocroc.centroids = np.array([[0., 0.],
                                     [1., 1.]])
        mocroc.egrid = range(2) # needs to have the correct length, contents not used
        mocroc.sample_points = DataArray([[0., 0.],
                                         [1., 1.]])
        vb.roc = mocroc

        res = vb._iterate_run(pred_dt_plus=0.2, true_dt_plus=None, true_dt_minus=0.)

        # set data
        self.assertTrue(vb.roc.set_data.called)
        self.assertEqual(vb.roc.set_data.call_count, 1)
        self.assertTrue(np.all(vb.roc.set_data.call_args[0] == vb.testing(dt_plus=0.2)[:, 1:]))

        # set prediction
        self.assertTrue(vb.roc.set_prediction.called)
        self.assertEqual(vb.roc.set_prediction.call_count, 1)
        self.assertEqual(len(vb.roc.set_prediction.call_args[0]), 1)
        t = (vb.cutoff_t + 0.2)
        # expected prediction values at (0,0), (1,1)
        pred_arg = DataArray(
            np.array([
                [t, 0., 0.],
                [t, 1., 1.]
            ])
        )
        pred_expctd = vb.model.predict(pred_arg)
        self.assertTrue(np.all(vb.roc.set_prediction.call_args[0][0] == pred_expctd))

        # set grid
        self.assertFalse(vb.roc.set_grid.called)
        vb.set_grid(0.1)
        self.assertTrue(vb.roc.set_grid.called)
        self.assertEqual(vb.roc.set_grid.call_count, 1)
        self.assertTupleEqual(vb.roc.set_grid.call_args[0], (0.1,))

        # evaluate
        self.assertTrue(vb.roc.evaluate.called)
        self.assertEqual(vb.roc.evaluate.call_count, 1)

    def test_run_calls_iterate_run(self):
        stk = hotspot.SKernelHistoric(1, bdwidth=0.3)

        # check correct calls are made to _iterate_run with no t_upper
        with mock.patch.object(validation.ValidationBase, '_iterate_run',
                               return_value=collections.defaultdict(list)) as m:
            vb = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))
            vb.set_grid(0.1)
            t0 = vb.cutoff_t
            vb.run(time_step=0.1)
            expct_call_count = math.ceil((1 - t0) / 0.1)
            self.assertEqual(m.call_count, expct_call_count)
            for c in m.call_args_list:
                self.assertEqual(c, mock.call(0.1, 0.1, 0.))  # signature: pred_dt_plus, true_dt_plus, true_dt_minus

    def test_run_calls_initial_setup(self):
        stk = hotspot.SKernelHistoric(1, bdwidth=0.3)

        # need to train model before running, otherwise it won't get past the call to the mocked function
        vb = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))
        vb.set_grid(0.1)
        vb.train_model()

        # check correct calls being made to _initial_setup
        with mock.patch.object(validation.ValidationBase, '_initial_setup') as m:

            t0 = vb.cutoff_t
            res = vb.run(time_step=0.1)
            self.assertEqual(m.call_count, 1)  # initial setup only called once
            self.assertEqual(m.call_args, mock.call())  # no arguments passed

            # test passing training keywords
            train_kwargs = {'foo': 'bar'}
            res = vb.run(time_step=0.1, train_kwargs=train_kwargs)
            self.assertEqual(m.call_args, mock.call(**train_kwargs))

    def test_run_calls_update(self):
        stk = hotspot.SKernelHistoric(1, bdwidth=0.3)

        # need to train model before running, otherwise it won't get past the call to the mocked function
        vb = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))
        vb.set_grid(0.1)
        vb.train_model()

        # check correct calls being made to _update
        with mock.patch.object(validation.ValidationBase, '_update') as m:
            t0 = vb.cutoff_t
            res = vb.run(time_step=0.1)

            # _update is the time updater, so cutoff time should not have changed
            self.assertEqual(vb.cutoff_t, t0)

            expct_call_count = math.ceil((1 - t0) / 0.1)
            self.assertEqual(m.call_count, expct_call_count)
            self.assertEqual(m.call_args_list[0], mock.call(time_step=0.0))  # _initial_setup routes to here
            self.assertEqual(m.call_args, mock.call(0.1))  # normal operation

        stk = hotspot.SKernelHistoric(1, bdwidth=0.3)
        vb = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))
        expct_call_count = math.ceil((1 - vb.cutoff_t) / 0.1)

    def test_run_no_grid(self):
        stk = hotspot.SKernelHistoric(1, bdwidth=0.3)
        vb = validation.ValidationBase(self.data, hotspot.Hotspot, model_args=(stk,))
        t0 = vb.cutoff_t
        # no grid specified raises error
        with self.assertRaises(AttributeError):
            res = vb.run(time_step=0.1)
        vb.set_grid(0.1)
        res = vb.run(time_step=0.1)
        expct_call_count = math.ceil((1 - t0) / 0.1)
        for v in res.values():
            # only test the length of iterable values, as the res dict also contains some parameter values
            # that are float or int
            if hasattr(v, '__iter__'):
                self.assertEqual(len(v), expct_call_count)
