__author__ = 'gabriel'
import datetime
from point_process import validate
from validation import hotspot, validation
import copy
import os
import sys
import numpy as np

from . import OUT_DIR, IN_DIR
import logging
import dill as pickle
INITIAL_CUTOFF = 212
num_sample_points_per_unit = 20
sample_unit_length = 100

if __name__ == '__main__':
    assert len(sys.argv) == 4, "Three input arguments required"
    loc = sys.argv[1]
    num_nn = (int(sys.argv[2]), int(sys.argv[3]))
    filename = 'nn_%d_%d.pickle' % num_nn

    out_dir = os.path.join(OUT_DIR, 'vary_num_nn', 'burglary', loc)
    log_file = os.path.join(out_dir, filename + '.log')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    logger = logging.getLogger('vary_num_nn_burglary.%s' % loc)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
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

    filename = 'nn_%d_%d.pickle' % num_nn
    fullfile = os.path.join(IN_DIR, 'vary_num_nn', 'burglary', loc, filename)

    logger.info("Running validation.")
    this_vb, this_res = validate.validate_pickled_model(fullfile,
                                                        sample_unit_length,
                                                        n_sample_per_grid=num_sample_points_per_unit,
                                                        domain=domain,
                                                        cutoff_t=INITIAL_CUTOFF,
                                                        n_iter=100,
                                                        parallel=False)

    logger.info("Saving results.")
    outfile = os.path.join(out_dir, filename)
    with open(outfile, 'w') as f:
        pickle.dump({
            'results': this_res,
            'vb_obj': this_vb
        }, f)
