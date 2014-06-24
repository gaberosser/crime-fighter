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

    def testing(self, dt=None, as_point=False):
        if dt:
            d = self.data[(self.t > self.cutoff_t) & (self.t <= (self.cutoff_t + dt))]
        else:
            d = self.data[self.t > self.cutoff_t]
        if as_point:
            return [(t[0], geos.Point(list(t[1:]))) for t in d]
        else:
            return d

    def train_model(self, *args, **kwargs):
        """ Train the predictor on training data """
        self.model.train(self.training)

    def predict_on_poly(self, t, poly):
        res, err = mcint.integrate(lambda x: self.predict(t, x[0], x[1]), mc_sampler(poly), n=100)
        return res

    def predict(self, t, x, y, *args, **kwargs):
        """ Run prediction using the trained model """
        return self.model.predict(t, x, y)

    def prediction_accuracy_index(self, dt, grid_size, *args, **kwargs):
        """
        Assess the trained model on a grid of supplied size.  Return the PAI metric for varying hit rate
        """
        main_grid_polys = create_spatial_grid(self.spatial_domain, grid_length=grid_size)
        method = kwargs.pop('method', 'centroid')

        # estimate total propensity in each grid poly
        res = []
        for p in main_grid_polys:
            if method == 'int':
                res.append(self.predict_on_poly(self.cutoff_t + dt, p))
            elif method == 'centroid':
                c = p.centroid.coords
                res.append(self.predict(self.cutoff_t + dt, c[0], c[1]))

        # sort by intensity
        res = np.array(res)
        sort_idx = np.argsort(res)[::-1]
        polys = [main_grid_polys[i] for i in sort_idx]
        res = res[sort_idx]

        # count actual crimes in testing dataset on same grid
        true_counts = np.zeros(len(polys))
        testing = self.testing(dt, as_point=True)
        for i in range(len(polys)):
            true_counts[i] += sum([t[1].intersects(polys[i]) for t in testing])
        a = np.array([t.area for t in polys])
        N = np.sum(true_counts)
        A = self.spatial_domain.area

        cfrac = np.cumsum(true_counts) / N
        pai = np.cumsum(true_counts) * A / (np.cumsum(a) * N)
        carea = np.cumsum(a) / A

        return polys, res, carea, cfrac, pai

    def run(self):
        """
        Run the required train / predict / assess sequence
        """
        raise NotImplementedError()


if __name__ == "__main__":
    from database import logic, models
    from scipy.stats import multivariate_normal
    import hotspot
    from analysis import plotting
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    camden = models.Division.objects.get(name='Camden')
    xm = 527753
    ym = 184284
    nd = 1000
    data = np.hstack((np.random.random((nd, 1)) * 10,
                      multivariate_normal.rvs(mean=[xm, ym], cov=np.diag([1e4, 1e4]), size=(nd, 1))))
    stk = hotspot.STKernelBowers(1, 1e-4)
    vb = ValidationBase(data, hotspot.Hotspot, camden.mpoly, model_args=(stk,))
    polys, res, carea, cfrac, pai = vb.prediction_accuracy_index(1, 200)
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

    plotting.plot_surface_on_polygon(camden.mpoly, lambda x,y: vb.predict(1.0, x, y), n=250)

    fig = plt.figure()
    plt.plot(carea, cfrac)