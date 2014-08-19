__author__ = 'gabriel'
from django.contrib.gis import geos
import numpy as np
import mcint
import math
from spatial import create_spatial_grid
import collections


def mc_sampler(poly):
    x_min, y_min, x_max, y_max = poly.extent
    while True:
        x = np.random.random() * (x_max - x_min) + x_min
        y = np.random.random() * (y_max - y_min) + y_min
        if poly.intersects(geos.Point([x, y])):
            yield (x, y)


class ValidationBase(object):

    def __init__(self, data, model_class, spatial_domain=None, grid_length=None, tmax_initial=None, model_args=None, model_kwargs=None):
        # sort data in increasing time
        self.data = data
        self.data = np.array(data)[np.argsort(self.t)]
        self.spatial_domain = spatial_domain

        self.model_args = model_args or []
        self.model_kwargs = model_kwargs or {}
        self.model = model_class(*self.model_args, **self.model_kwargs)

        # set initial time cut point
        self.cutoff_t = tmax_initial or self.t[int(self.ndata / 2)]
        # self.set_t_cutoff(tmax_initial or self.t[int(self.ndata / 2)])

        # set grid for evaluation
        self._grid = []
        self.centroids = np.array([], dtype=float)
        self.a = []
        if grid_length:
            self.set_grid(grid_length)

    def set_t_cutoff(self, cutoff_t, b_train=True, **kwargs):
        """
        Set cutoff time that divides dataset into training and testing portions.
        :param cutoff_t: New value for cutoff time.
        :param b_train: Boolean indicating whether the model should be (re)trained after setting the new cutoff
        :param kwargs: kwargs to pass to training function.
        """
        self.cutoff_t = cutoff_t
        if b_train:
            self.train_model(**kwargs)

    def set_grid(self, grid_length):
        if not self.spatial_domain:
            # find minimal bounding rectangle
            xmin, ymin = np.min(self.data[:, 1:], axis=0)
            xmax, ymax = np.max(self.data[:, 1:], axis=0)
            self.spatial_domain = geos.Polygon([
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
            ])
        self._grid = create_spatial_grid(self.spatial_domain, grid_length)
        self.centroids = np.array([t.centroid.coords for t in self._grid])
        self.a = np.array([t.area for t in self._grid])

    @property
    def grid(self):
        if len(self._grid):
            return self._grid
        raise AttributeError("Grid has not been computed, run set_grid with grid length")

    @property
    def A(self):
        return sum([t.area for t in self.grid])

    @property
    def ndata(self):
        return self.data.shape[0]

    @property
    def t(self):
        return self.data[:, 0]

    @property
    def training(self):
        return self.data[self.t <= self.cutoff_t]

    def testing(self, dt_plus=None, dt_minus=0., as_point=False):
        """
        :param dt_plus: Number of time units ahead of cutoff_t to take as maximum testing data.  If None, take ALL data.
        :param dt_minus: Number of time units ahead of cutoff_t to take as minimum testing data.  Defaults to 0.
        :param as_point: If True, return N length list of (time, geos.Point) tuples, else return N x 3 matrix
        :return: Testing data for comparison with predictions, based on value of cutoff_t.
        """
        bottom = self.cutoff_t + dt_minus
        if dt_plus:
            d = self.data[(self.t > bottom) & (self.t <= (self.cutoff_t + dt_plus))]
        else:
            d = self.data[self.t > bottom]
        if as_point:
            return [(t[0], geos.Point(list(t[1:]))) for t in d]
        else:
            return d

    def train_model(self, *args, **kwargs):
        """ Train the predictor on training data """
        self.model.train(self.training, *args, **kwargs)

    def predict_on_poly(self, t, poly, *args, **kwargs):
        # FIXME: disabled as too slow
        method = kwargs.pop('method', 'centroid')
        if method == 'int':
            res, err = mcint.integrate(lambda x: self.model.predict(t, x[0], x[1]), mc_sampler(poly), n=100)
        elif method == 'centroid':
            res = self.model.predict(t, *poly.centroid.coords)
        else:
            raise NotImplementedError("Unsupported method %s", method)
        return res

    def predict(self, t, **kwargs):
        print "predict"
        # estimate total propensity in each grid poly
        # use centroid method for speed
        ts = np.ones(len(self.grid)) * t
        return self.model.predict(ts, self.centroids[:, 0], self.centroids[:, 1])

    def true_values(self, dt_plus, dt_minus):
        # count actual crimes in testing dataset on grid
        n = np.zeros(len(self.grid))
        testing = self.testing(dt_plus=dt_plus, dt_minus=dt_minus, as_point=True)
        for i in range(len(self.grid)):
            n[i] += sum([t[1].intersects(self.grid[i]) for t in testing])
        return n

    def predict_assess(self, pred_dt_plus, true_dt_plus=None, true_dt_minus=0, *args, **kwargs):
        """
        Assess the trained model on the polygonal grid.  Return the PAI metric for varying hit rate.
        :param pred_dt_plus: Time units ahead of cutoff_t to use when computing the prediction
        :param true_dt_plus: Time units ahead of cutoff_t to take as the maximum test data, defaults to same value as
        pred_dt_plus
        :param true_dt_minus: Time units ahead of cutoff_t to take as the minimum test data, defaults to 0
        """
        print "predict_assess"
        true_dt_plus = true_dt_plus or pred_dt_plus
        pred = self.predict(self.cutoff_t + pred_dt_plus, **kwargs)

        # count actual crimes in testing dataset on same grid
        true = self.true_values(true_dt_plus, true_dt_minus)

        # sort by descending predicted values
        rank = np.argsort(pred)[::-1]
        true = true[rank]
        pred = pred[rank]

        N = sum(true)
        a = [self.a[i] for i in rank]
        cfrac = np.cumsum(true) / N
        carea = np.cumsum(a) / self.A
        pai = cfrac * (self.A / np.cumsum(a))

        return rank, pred, carea, cfrac, pai

    def _iterate_run(self, pred_dt_plus, true_dt_plus, true_dt_minus, **kwargs):
        print "_iterate_run"
        rank, _, carea, cfrac, pai = self.predict_assess(pred_dt_plus=pred_dt_plus,
                                                         true_dt_plus=true_dt_plus,
                                                         true_dt_minus=true_dt_minus,
                                                         **kwargs)
        return {
            'rank': rank,
            'carea': carea,
            'cfrac': cfrac,
            'pai': pai
        }

    def run(self, time_step, pred_dt_plus=None, true_dt_plus=None, true_dt_minus=0, t_upper=None, pred_kwargs=None,
            train_kwargs=None, **kwargs):
        """
        Run the required train / predict / assess sequence
        Take the mean of the metrics returned
        :param time_step: Time step to use in each successive train-predict-assess cycle
        :param pred_dt_plus: Time units ahead of cutoff_t to use when computing the prediction
        :param true_dt_plus: Time units ahead of cutoff_t to take as the maximum test data, defaults to same value as
        pred_dt_plus
        :param true_dt_minus: Time units ahead of cutoff_t to take as the minimum test data, defaults to 0
        :param t_upper: Maximum time to use in data.  Upon reaching this, the run ceases.
        :param pred_kwargs: kwargs passed to prediction function
        :param train_kwargs: kwargs passed to train function
        :param kwargs: kwargs for the run function itself
        :return: results dictionary.
        """
        print "run"
        verbose = kwargs.pop('verbose', True)
        pred_kwargs = pred_kwargs or {}
        train_kwargs = train_kwargs or {}
        pred_dt_plus = pred_dt_plus or time_step
        true_dt_plus = true_dt_plus or pred_dt_plus
        t_upper = min(t_upper or np.inf, self.testing()[-1, 0])

        # store current cutoff so it can be restored after this run
        t0 = self.cutoff_t

        # precompute number of iterations
        n_iter = math.ceil((t_upper - t0) / time_step)

        res = collections.defaultdict(list)

        if verbose:
            print "Running %d validation iterations..." % n_iter

        count = 0

        try:
            while self.cutoff_t < t_upper:
                if verbose:
                    print "Running validation with cutoff time %s (iteration %d / %d)" % (str(self.cutoff_t),
                                                                                          count + 1,
                                                                                          n_iter)

                # initial training
                if count == 0:
                    self._initial_setup(**train_kwargs)

                # predict and assess iteration
                this_res = self._iterate_run(pred_dt_plus, true_dt_plus, true_dt_minus, **pred_kwargs)
                for k, v in this_res.items():
                    res[k].append(v)

                # update model as required and update the cutoff time
                self._update(time_step, **train_kwargs)

                count += 1

        except KeyboardInterrupt:
            # this breaks the loop, now return res as it stands
            pass
        finally:
            self.set_t_cutoff(t0, b_train=False)

        return res

    def _update(self, time_step, **train_kwargs):
        """
        This function gets called after a successful train-predict-assess cycle.
        It is used to setup and TRAIN the model for the next cycle.
        It should ALWAYS update the cutoff time
        """
        # training is implied here
        self.set_t_cutoff(self.cutoff_t + time_step, **train_kwargs)

    def _initial_setup(self, **train_kwargs):
        self._update(time_step=0, **train_kwargs)


if __name__ == "__main__":
    from database import logic, models
    from scipy.stats import multivariate_normal
    from point_process import models as pp_models
    from point_process import simulate
    import hotspot
    from analysis import plotting
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    camden = models.Division.objects.get(name='Camden')
    xm = 526500
    ym = 186000
    nd = 1000
    # nice normal data
    # data = np.hstack((np.random.normal(loc=5, scale=5, size=(nd, 1)),
    #                   multivariate_normal.rvs(mean=[xm, ym], cov=np.diag([1e5, 1e5]), size=(nd, 1))))

    # moving cluster
    data = np.hstack((
        np.linspace(0, 10, nd).reshape((nd, 1)),
        xm + np.linspace(0, 5000, nd).reshape((nd, 1)) + np.random.normal(loc=0, scale=100, size=(nd, 1)),
        ym + np.linspace(0, -4000, nd).reshape((nd, 1)) + np.random.normal(loc=0, scale=100, size=(nd, 1)),
    ))

    # use Bowers kernel
    # stk = hotspot.STKernelBowers(1, 1e-4)
    # vb = ValidationBase(data, hotspot.Hotspot, camden.mpoly, model_args=(stk,))
    # vb.set_grid(grid_length=200)
    # vb.set_t_cutoff(4.0)

    # use basic historic data spatial hotspot
    # sk = hotspot.SKernelHistoric(2) # use heatmap from final 2 days data
    # vb = ValidationBase(data, hotspot.Hotspot, camden.mpoly, model_args=(sk,))
    # vb.set_grid(grid_length=200)
    # vb.set_t_cutoff(4.0)

    # use point process model
    c = simulate.MohlerSimulation()
    c.run()
    data = np.array(c.data)[:, :3]  # (t, x, y, b_is_BG)
    data = data[data[:, 0].argsort()]
    vb = ValidationBase(data, pp_models.PointProcess,
                        model_kwargs={'max_trigger_t': 80, 'max_trigger_d': 0.75})
    vb.set_grid(grid_length=5)
    vb.train_model(niter=20)


    rank, res, carea, cfrac, pai = vb.predict_assess(pred_dt_plus=1) # predict one day ahead
    polys = [vb.grid[i] for i in rank]
    n = vb.true_values(dt_plus=1, dt_minus=0)

    norm = mpl.colors.Normalize(min(res), max(res))
    cmap = mpl.cm.jet
    sm = mpl.cm.ScalarMappable(norm, cmap)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (p, r) in zip(polys, res):
        plotting.plot_geodjango_shapes(shapes=(p,), ax=ax, facecolor=sm.to_rgba(r), set_axes=False)
    plotting.plot_geodjango_shapes((vb.spatial_domain,), ax=ax, facecolor='none')

    norm = mpl.colors.Normalize(min(n), max(n))
    sm = mpl.cm.ScalarMappable(norm, cmap)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (p, r) in zip(vb.grid, n):
        plotting.plot_geodjango_shapes(shapes=(p,), ax=ax, facecolor=sm.to_rgba(r), set_axes=False)
    plotting.plot_geodjango_shapes((vb.spatial_domain,), ax=ax, facecolor='none')

