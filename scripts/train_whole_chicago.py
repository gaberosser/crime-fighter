__author__ = 'gabriel'
import datetime
import numpy as np
import pickle
import os
from point_process import models as pp_models, estimation, validate
from . import OUT_DIR, IN_DIR
import logging


def data_by_date_range(t0, start, end):
    dt0 = (start - t0).days
    dt1 = (end - t0).days

num_sample_points = 20
sample_unit_length = 100

if __name__ == '__main__':
    OUT_SUBDIR = os.path.join(OUT_DIR, 'chicago')
    LOG_FILE = os.path.join(OUT_SUBDIR, '%s.log' % datetime.datetime.now().strftime('%Y-%m-%d_%H:%M'))

    if not os.path.isdir(OUT_SUBDIR):
        os.makedirs(OUT_SUBDIR)

    logger = logging.getLogger('train_whole_chicago')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("Logger set.  Script started.")

    for t in ('point_process.models', 'validation.validation'):
        this_logger = logging.getLogger(t)
        this_logger.setLevel(logging.DEBUG)
        this_logger.handlers = []  # clear existing handlers
        this_logger.addHandler(fh)  # replace with the same file output


    logger.info("Loading boundary.")
    poly_file = os.path.join(IN_DIR, 'boundaries.pickle')
    with open(poly_file, 'r') as f:
        boundaries = pickle.load(f)
    domain = boundaries[loc]