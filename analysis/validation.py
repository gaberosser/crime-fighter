__author__ = 'gabriel'
from django.contrib.gis import geos
import numpy as np
import mcint
import math
from spatial import create_spatial_grid


def mc_sampler(poly):
    x_min, y_min, x_max, y_max = poly.extent
    while True:
        x = np.random.random() * (x_max - x_min) + x_min
        y = np.random.random() * (y_max - y_min) + y_min
        if poly.intersects(geos.Point([x, y])):
            yield (x, y)


class ValidationBase(object):

    def __init__(self, data, model_class, spatial_domain, tmax_initial=None, model_args=None, model_kwargs=None):
        # sort data in increasing time
        self.data = data
        self.data = np.array(data)[np.argsort(self.t)]
        self.model = model_class
        self.spatial_domain = spatial_domain
        self.A = self.spatial_domain.area
        self.model_args = model_args or []
        self.model_kwargs = model_kwargs or {}
        self.model = model_class([], *self.model_args, **self.model_kwargs)
        self.set_t_cutoff(tmax_initial or self.t[int(self.ndata / 2)])

    def set_t_cutoff(self, cutoff_t):
        self.cutoff_t = cutoff_t
        self.train_model()

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
        self.model.train(self.training)

    def predict_on_poly(self, t, poly, *args, **kwargs):
        method = kwargs.pop('method', 'centroid')
        if method == 'int':
            res, err = mcint.integrate(lambda x: self.model.predict(t, x[0], x[1]), mc_sampler(poly), n=100)
        elif method == 'centroid':
            res = self.predict(t, *poly.centroid.coords)
        else:
            raise NotImplementedError("Unsupported method %s", method)
        return res

    def predict(self, t, x, y, *args, **kwargs):
        """ Run prediction using the trained model """
        return self.model.predict(t, x, y)

    def prediction_accuracy_index(self, dt_plus, grid, dt_minus=0., *args, **kwargs):
        """
        Assess the trained model on a grid of supplied size.  Return the PAI metric for varying hit rate
        """
        # estimate total propensity in each grid poly
        res = []
        for p in grid:
            res.append(self.predict_on_poly(self.cutoff_t + dt_plus, p, **kwargs))

        # sort by intensity
        res = np.array(res)
        sort_idx = np.argsort(res)[::-1]
        polys = [grid[i] for i in sort_idx]
        res = res[sort_idx]

        # count actual crimes in testing dataset on same grid
        true_counts = np.zeros(len(polys))
        testing = self.testing(dt_plus=dt_plus, dt_minus=dt_minus, as_point=True)
        for i in range(len(polys)):
            true_counts[i] += sum([t[1].intersects(polys[i]) for t in testing])
        a = np.array([t.area for t in polys])
        N = np.sum(true_counts)

        cfrac = np.cumsum(true_counts) / N
        pai = np.cumsum(true_counts) * self.A / (np.cumsum(a) * N)
        carea = np.cumsum(a) / self.A

        return polys, res, carea, cfrac, pai

    def run(self, grid_size, dt, t_upper=None):
        """
        Run the required train / predict / assess sequence
        Take the mean of the metrics returned
        """
        t0 = self.cutoff_t
        t = dt
        t_upper = min(t_upper or np.inf, self.testing()[-1, 0])

        grid = create_spatial_grid(self.spatial_domain, grid_length=grid_size)
        cfrac = []
        pai = []

        try:
            while self.cutoff_t < t_upper:
                # predict and assess
                polys, _, carea, cfrac_, pai_ = self.prediction_accuracy_index(dt_plus=dt, dt_minus=0, grid=grid)
                cfrac.append(cfrac_)
                pai.append(pai_)
                self.set_t_cutoff(self.cutoff_t + dt)

        finally:
            self.set_t_cutoff(t0)

        return polys, carea, cfrac, pai


if __name__ == "__main__":
    from database import logic, models
    from scipy.stats import multivariate_normal
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

    stk = hotspot.STKernelBowers(1, 1e-4)
    vb = ValidationBase(data, hotspot.Hotspot, camden.mpoly, model_args=(stk,))
    grid = create_spatial_grid(camden.mpoly, grid_length=200)
    polys, res, carea, cfrac, pai = vb.prediction_accuracy_index(dt_plus=5, dt_minus=4, grid=grid)
    x = [a.centroid.coords[0] for a in polys]
    y = [a.centroid.coords[1] for a in polys]

    norm = mpl.colors.Normalize(min(res), max(res))
    cmap = mpl.cm.jet
    sm = mpl.cm.ScalarMappable(norm, cmap)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (p, r) in zip(polys, res):
        plotting.plot_geodjango_shapes(shapes=(p,), ax=ax, facecolor=sm.to_rgba(r), set_axes=False)
    plotting.plot_geodjango_shapes((camden.mpoly,), ax=ax, facecolor='none')

    # plotting.plot_surface_on_polygon(camden.mpoly, lambda x,y: vb.predict(1.0, x, y), n=250)

    fig = plt.figure()
    plt.plot(carea, cfrac)