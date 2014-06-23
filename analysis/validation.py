__author__ = 'gabriel'
from django.contrib.gis import geos
import numpy as np
import mcint
import math


def create_spatial_grid(spatial_domain, grid_length, offset_coords=None):
    """
    Compute a grid on the spatial domain.
    :param grid_length: the length of one side of the grid square
    :param offset_coords: tuple giving the (x, y) coordinates of the bottom LHS of a gridsquare, default = (0, 0)
    :return: list of grid vertices and centroids (both (x,y) pairs)
    """
    offset_coords = offset_coords or (0, 0)
    xmin, ymin, xmax, ymax = spatial_domain.extent

    # 1) create grid over entire bounding box
    sq_x_l = math.ceil((offset_coords[0] - xmin) / grid_length)
    sq_x_r = math.ceil((xmax - offset_coords[0]) / grid_length)
    sq_y_l = math.ceil((offset_coords[1] - ymin) / grid_length)
    sq_y_r = math.ceil((ymax - offset_coords[1]) / grid_length)
    edges_x = grid_length * np.arange(-sq_x_l, sq_x_r + 1) + offset_coords[0]
    edges_y = grid_length * np.arange(-sq_y_l, sq_y_r + 1) + offset_coords[1]
    polys = []
    for ix in range(len(edges_x) - 1):
        for iy in range(len(edges_y) - 1):
            p = geos.Polygon((
                (edges_x[ix], edges_y[iy]),
                (edges_x[ix+1], edges_y[iy]),
                (edges_x[ix+1], edges_y[iy+1]),
                (edges_x[ix], edges_y[iy+1]),
                (edges_x[ix], edges_y[iy]),
            ))
            if spatial_domain.intersects(p):
                polys.append(spatial_domain.intersection(p))

    return polys


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
        self.cutoff_t = tmax_initial or self.t[int(self.ndata / 2)]
        self.model_args = model_args or []
        self.model_kwargs = model_kwargs or {}
        self.model = model_class(data, *self.model_args, **self.model_kwargs)

    @property
    def ndata(self):
        return self.data.shape[0]

    @property
    def t(self):
        return self.data[:, 0]

    @property
    def training(self):
        return self.data[self.t <= self.cutoff_t]

    @property
    def testing(self):
        return self.data[self.t > self.cutoff_t]

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
        Test the trained predictor using a metric, etc.
        """
        domain_area = self.spatial_domain.area
        main_grid_polys = create_spatial_grid(self.spatial_domain, grid_length=grid_size)
        method = kwargs.pop('method', 'centroid')

        res = []
        for p in main_grid_polys:
            if method == 'int':
                res.append(self.predict_on_poly(self.cutoff_t + dt, p) * p.area / domain_area)
            elif method == 'centroid':
                c = p.centroid.coords
                res.append(self.predict(self.cutoff_t + dt, c[0], c[1]) * p.area / domain_area)
        return np.array(res)

    def run(self):
        """
        Run the required train / predict / assess sequence
        """
        raise NotImplementedError()