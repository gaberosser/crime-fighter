__author__ = 'gabriel'
import numpy as np
import math
import runner
from database import models, logic
from analysis import validation


# TODO: refactor this, inheriting almost everything from analysis.validation.ValidationBase
# main addition is probably a more efficient version of the run() routine in which the previous estimate is applied to
# improve the convergence of the train algorithm.  Also reduce the number of iterations.

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
