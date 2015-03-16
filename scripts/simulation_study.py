__author__ = 'gabriel'
import datetime
import os
# import pickle
import dill as pickle
import logging
import sys
import getopt
from point_process import models as pp_models, estimation, validate
from . import OUT_DIR, IN_DIR

# global parameters
num_sample_points = 20
grid_size = 250  # metres

estimate_kwargs = {
    'ct': 1,
    'cd': 10.
}

bg_kde_kwargs = {'number_nn': [100, 15],
                 'strict': False}

trigger_kde_kwargs = {'number_nn': 15,
                      'strict': False}

niter = 25  # number of SEPP iterations before convergence is assumed

def run_me(location_dir, max_t, max_d):
    data_file = os.path.join(IN_DIR, location_dir, 'simulation.pickle')
    out_dir = os.path.join(OUT_DIR, location_dir)
    log_file = os.path.join(out_dir, 'simulation_%.2f-%.2f.log' % (max_t, max_d))

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # set loggers
    logger = logging.getLogger('simulation_study')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    for t in ('point_process.models', 'kde.models'):
        this_logger = logging.getLogger(t)
        this_logger.setLevel(logging.DEBUG)
        this_logger.handlers = []  # clear existing handlers
        this_logger.addHandler(fh)  # replace with the same file output

    logger.info("Logger set.  Script started.")
    logger.info("Simulation study.")

    # load data
    with open(data_file, 'r') as f:
        data = pickle.load(f)
    logger.info("Loaded data.")

    logger.info("Instantiating SEPP object")

    r = pp_models.SeppStochasticNn(data=data, max_delta_d=max_d, max_delta_t=max_t,
                                bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)
    p = estimation.estimator_bowers(data, r.linkage)
    r.p = p
    r.set_seed(42)

    try:
        logger.info("Starting training run.")
        r.train(niter=niter)
    except Exception as exc:
        logger.error(repr(exc))
        res = None
    finally:
        file_stem = os.path.join(out_dir, 'simulation_%.2f-%.2f' % (max_t, max_d))
        logger.info("Saving results (or None).")
        with open(file_stem + '-sepp_obj.pickle', 'w') as f:
            pickle.dump(r, f)



if __name__ == '__main__':

    max_t = float(sys.argv[1])
    max_d = float(sys.argv[2])
    location_dir = 'simulation'
    run_me(location_dir, max_t, max_d)
