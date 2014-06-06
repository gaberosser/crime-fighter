__author__ = 'gabriel'
import numpy as np
import math
import runner
from database import models, logic

class ValidationBase(object):

    def __init__(self, data, predictor, spatial_domain, pp_class=None, pp_init_args=None, pp_init_kwargs=None):
        self.pp_class = pp_class or runner.PointProcess
        self.pp_init_args = pp_init_args or ()
        self.pp_init_kwargs = pp_init_kwargs or {}
        self.data = data
        self.predictor = predictor
        self.spatial_domain = spatial_domain
        self.train_idx = []
        self.test_idx = []
        self.pp = None

    @property
    def ndata(self):
        return self.data.shape[0]

    @property
    def training(self):
        return self.data[self.train_idx]

    @property
    def testing(self):
        return self.data[self.test_idx]

    def create_spatial_grid(self, grid_length, offset_coords=None):
        """
        Compute a grid on the spatial domain.
        :param grid_length: the length of one side of the grid square
        :param offset_coords: tuple giving the (x, y) coordinates of the bottom LHS of a gridsquare, default = (0, 0)
        :return: list of grid vertices and centroids (both (x,y) pairs)
        """
        offset_coords = offset_coords or (0, 0)
        xmin, ymin, xmax, ymax = self.spatial_domain.extent
        sq_x_l = math.ceil((offset_coords[0] - xmin) / grid_length)
        sq_x_r = math.ceil((xmax - offset_coords[0]) / grid_length)
        sq_y_l = math.ceil((offset_coords[1] - ymin) / grid_length)
        sq_y_r = math.ceil((ymax - offset_coords[1]) / grid_length)
        edges_x = grid_length * np.arange(-sq_x_l, sq_x_r + 1) + offset_coords[0]
        edges_y = grid_length * np.arange(-sq_y_l, sq_y_r + 1) + offset_coords[1]
        centres_x = grid_length * 0.5 + edges_x[:-1]
        centres_y = grid_length * 0.5 + edges_y[:-1]
        ex, ey = np.meshgrid(edges_x, edges_y, copy=False)
        cx, cy = np.meshgrid(centres_x, centres_y, copy=False)
        return zip(ex.flatten(), ey.flatten()), zip(cx.flatten(), cy.flatten())

    def split_data(self, test_frac=0.4, *args, **kwargs):
        """ Split the data into training and test """
        m = np.random.random(self.ndata) < test_frac
        self.train_idx = np.where(~m)
        self.test_idx = np.where(m)

    def train_model(self, *args, **kwargs):
        """ Train the predictor on training data """
        # return risk function
        self.pp = self.pp_class(self.training, *self.pp_init_args, **self.pp_init_kwargs)
        self.pp.train(*args, **kwargs)

    def predict(self, t, x, y, *args, **kwargs):
        """ Run prediction using the trained model """
        return self.pp.evaluate(t, x, y)

    def comparison_metric_1(self, *args, **kwargs):
        """
        Test the trained predictor using a metric, etc.
        """
        raise NotImplementedError()

    def run(self):
        """
        Run the required train / predict / assess sequence
        """
        raise NotImplementedError()


class ValidationForecastSequential(ValidationBase):

    def split_data(self, t_max=None, *args, **kwargs):
        """ Split the data into training and test based on the threshold time t_max """
        if t_max is None:
            t_max = self.data[int(self.ndata / 2), 0] # half readings
        idx = self.data[:, 0] < t_max
        self.train_idx  = np.where(idx)
        self.test_idx = np.where(~idx)

    def predict(self, t, x, y, *args, **kwargs):
        pass