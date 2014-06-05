__author__ = 'gabriel'
import numpy as np
import math
from database import models, logic

class ValidationBase(object):

    def __init__(self, data, predictor, spatial_domain):
        self.data = data
        self.predictor = predictor
        self.spatial_domain = spatial_domain

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
        return zip(edges_x, edges_y), zip(centres_x, centres_y)

    def split_data(self, *args, **kwargs):
        """ Split the data into training and test """
        raise NotImplementedError()

    def train_model(self, *args, **kwargs):
        """ Train the predictor on training data """
        # return risk function
        raise NotImplementedError()

    def comparison_metric_1(self, *args, **kwargs):
        """
        Test the trained predictor using a metric, etc.
        """
        raise NotImplementedError()