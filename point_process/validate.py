__author__ = 'gabriel'
import numpy as np
from database import models, logic

class ValidationBase(object):

    def __init__(self, data, predictor, spatial_domain):
        self.data = data
        self.predictor = predictor
        self.spatial_domain = spatial_domain

    def create_spatial_grid(self, grid_length):
        """
        Compute a grid on the spatial domain.
        :param grid_length: the length of one side of the grid square
        :return: list of grid vertices and centroids (both (x,y) pairs)
        """
        pass

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