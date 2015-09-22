__author__ = 'gabriel'
import datetime
import time
import os
import dill as pickle
import logging
import sys
from pytz import utc
from point_process import models as pp_models, estimation, validate
from . import OUT_DIR, IN_DIR

OUT_SUBDIR = 'validate_chicago_ani_vs_iso_refl_keep_coincident'

# global parameters
num_sample_points = 50
grid_size = 250  # metres
niter = 150  # number of SEPP iterations before convergence is assumed
num_validation = 100  # number of predict - assess cycles
start_date = datetime.datetime(2011, 3, 1)  # first date for which data are required
start_day_number = 366  # number of days (after start date) on which first prediction is made

estimate_kwargs = {
    'ct': 0.1,
    'cd': 150,
    'frac_bg': None,
}

model_kwargs = {
    'parallel': False,
    'max_delta_t': 90, # set on each iteration
    'max_delta_d': 500, # set on each iteration
    'bg_kde_kwargs': {'number_nn': [100, 15],
                      'min_bandwidth': None,
                      'strict': False},
    'trigger_kde_kwargs': {'number_nn': 15,
                           'min_bandwidth': None,
                           'strict': False},
    'estimation_function': lambda x, y: estimation.estimator_exp_gaussian(x, y, **estimate_kwargs),
    'seed': 42,  # doesn't matter what this is, just want it fixed
    'remove_coincident_pairs': False
}

pred_include = ('full_static',)  # only use this method for prediction

## DEBUGGING:
#niter = 2  # number of SEPP iterations before convergence is assumed
#num_validation = 2  # number of predict - assess cycles
#num_sample_points = 2

# end_date is the maximum required date
end_date = start_date + datetime.timedelta(days=start_day_number + num_validation)


def run_me(data, domain, out_dir, run_name, pp_class):
    # data_file = os.path.join(IN_DIR, location_dir, '%s.pickle' % crime_type)
    # out_dir = os.path.join(OUT_DIR, location_dir, OUT_SUBDIR)
    log_file = os.path.join(out_dir, '%s.log' % run_name)

    if not os.path.isdir(out_dir):
        try:
            os.makedirs(out_dir)
        except OSError:
            # wait a moment, just in case another process has just done the folder creation
            time.sleep(1)
            if not os.path.isdir(out_dir):
                raise

    # set loggers
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    for t in ('point_process.models', 'validation.validation'):
        this_logger = logging.getLogger(t)
        this_logger.setLevel(logging.DEBUG)
        this_logger.handlers = []  # clear existing handlers
        this_logger.addHandler(fh)  # replace with the same file output

    logger.info("Logger set.  Script started.")

    # check that num_validation iterations is feasible
    if start_day_number + num_validation - 1 > data[-1, 0]:
        this_num_validation = int(data[-1, 0]) - start_day_number + 1
        logger.info("Can only do %d validation runs" % this_num_validation)
    else:
        this_num_validation = num_validation

    logger.info("Instantiating validation object")

    sepp = pp_class(data=data, **model_kwargs)
    vb = validate.SeppValidationFixedModelIntegration(
        data=data,
        model=sepp,
        spatial_domain=domain,
        cutoff_t=start_day_number,
    )

    logger.info("Setting validation grid")
    vb.set_sample_units(grid_size, num_sample_points)
    file_stem = os.path.join(out_dir, run_name)
    try:
        logger.info("Starting validation run.")
        res = vb.run(time_step=1,
                     n_iter=this_num_validation,
                     verbose=True,
                     train_kwargs={'niter': niter},
                     pred_kwargs={'include': pred_include})
    except Exception as exc:
        logger.error(repr(exc))
        res = None
    finally:
        logger.info("Saving results (or None).")
        with open(file_stem + '-validation.pickle', 'w') as f:
            pickle.dump(res, f)


if __name__ == '__main__':
    assert len(sys.argv) == 4, "Three input arguments required"

    # arg 1: region (chicago_south, chicago_northwest, ...)
    poly_file = os.path.join(IN_DIR, 'boundaries.pickle')
    with open(poly_file, 'r') as f:
        boundaries = pickle.load(f)
        domain = boundaries[sys.argv[1]]

    # arg 2: crime type (burglary, assault, motor_vehicle_theft, ...)
    crime_type = sys.argv[2]
    data_infile = os.path.join(IN_DIR, 'chicago', sys.argv[1], '%s.pickle' % sys.argv[2])
    with open(data_infile, 'r') as f:
        data, t0, cid = pickle.load(f)

    if sys.argv[3] == 'ani':
        pp_class = pp_models.SeppStochasticNn
    elif sys.argv[3] == 'iso':
        pp_class = pp_models.SeppStochasticNnIsotropicTrigger
    elif sys.argv[3] == 'ani_refl':
        pp_class = pp_models.SeppStochasticNnReflected
    elif sys.argv[3] == 'iso_refl':
        pp_class = pp_models.SeppStochasticNnIsotropicReflectedTrigger
    else:
        raise AttributeError("Unsupported pp_class type")

    # cut data to selected date range
    sd = (start_date.replace(tzinfo=utc) - t0).days
    ed = (end_date.replace(tzinfo=utc) - t0).days
    data = data[(data[:, 0] >= sd) & (data[:, 0] < ed + 1)]
    data[:, 0] -= min(data[:, 0])

    out_dir = os.path.join(OUT_DIR, OUT_SUBDIR, sys.argv[1], sys.argv[2])
    run_name = '%s-%s-%s' % tuple(sys.argv[1:4])

    run_me(data, domain, out_dir, run_name, pp_class)
