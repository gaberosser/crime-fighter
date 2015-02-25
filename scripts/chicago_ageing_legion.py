__author__ = 'gabriel'
from point_process import models as pp_models, estimation, validate
import datetime
import numpy as np
import os
import pickle
import io
import sys


# sys.argv[1] etc...

OUTPUT_DIR = sys.argv[4]
num_sample_points = 20

estimate_kwargs = {
    'ct': 1,
    'cd': 0.02
}
model_kwargs = {
    'max_delta_t': 60,
    'max_delta_d': 400,
    'bg_kde_kwargs': {'number_nn': [100, 15],
                      'min_bandwidth': [0.5, 20, 20],
                      'strict': False},
    'trigger_kde_kwargs': {'number_nn': 15,
                           'min_bandwidth': [0.5, 20, 20],
                           'strict': False},
    'estimation_function': lambda x, y: estimation.estimator_bowers(x, y, **estimate_kwargs),
    'seed': 42,  # doesn't matter what this is, just want it fixed
}

niter = 75

