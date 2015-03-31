__author__ = 'gabriel'
import datetime
import os
import dill as pickle
import logging
import sys
from point_process import models as pp_models, estimation, validate
from . import OUT_DIR, IN_DIR

LOG_DIR = os.path.join(OUT_DIR, 'logs')

# global parameters
num_sample_points = 20
grid_size = 250  # metres

estimate_kwargs = {
    'ct': 1,
    'cd': 0.02
}
model_kwargs = {
    'parallel': False,
    # 'max_delta_t': 60, # set on each iteration
    # 'max_delta_d': 400, # set on each iteration
    'bg_kde_kwargs': {'number_nn': [100, 15],
                      'min_bandwidth': None,  ## FIXED
                      'strict': False},
    'trigger_kde_kwargs': {'number_nn': 15,
                           'min_bandwidth': [0.5, 20, 20],  # FIXED
                           'strict': False},
    'estimation_function': lambda x, y: estimation.estimator_bowers(x, y, **estimate_kwargs),
    'seed': 42,  # doesn't matter what this is, just want it fixed
}

min_bandwidth_by_crime_type_camden = {
    'burglary': [0, 10, 10],
    'robbery': [0.25, 0, 0],
    'violence': [0.25, 0, 0],
    'theft_of_vehicle': [0.25, 10, 10],
}

niter = 75  # number of SEPP iterations before convergence is assumed
num_validation = 120  # number of predict - assess cycles

## DEBUGGING:
# niter = 5  # number of SEPP iterations before convergence is assumed
# num_validation = 5  # number of predict - assess cycles

# start_date is the FIRST DATE FOR WHICH DATA ARE USED
start_date = datetime.datetime(2011, 3, 1)

# number of days from t0 (1/3/2011) at which we start predictions
start_day_number = 277

# end_date is the maximum required date
end_date = start_date + datetime.timedelta(days=start_day_number + num_validation)


def run_me(location_dir, location_poly, max_delta_t, max_delta_d, crime_type):
    data_file = os.path.join(IN_DIR, location_dir, '%s.pickle' % crime_type)
    out_dir = os.path.join(OUT_DIR, location_dir, 'max_triggers')
    log_file = os.path.join(out_dir, crime_type + '_' + '%d-%d.log' % (max_delta_t, max_delta_d))

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # set loggers
    logger = logging.getLogger('vary_max_triggers.%s' % location_dir)
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
    logger.info("Crime type: %s. Max delta t %d, max delta d %d" % (crime_type, max_delta_t, max_delta_d))

    # load data
    with open(data_file, 'r') as f:
        data = pickle.load(f)
    logger.info("Loaded data.")

    # check that num_validation iterations is feasible
    if start_day_number + num_validation - 1 > data[-1, 0]:
        this_num_validation = int(data[-1, 0]) - start_day_number + 1
        logger.info("Can only do %d validation runs" % this_num_validation)
    else:
        this_num_validation = num_validation

    model_kwargs['max_delta_t'] = max_delta_t
    model_kwargs['max_delta_d'] = max_delta_d
    logger.info("Instantiating validation object")
    vb = validate.SeppValidationFixedModelIntegration(
        data=data,
        pp_class=pp_models.SeppStochasticNn,
        spatial_domain=location_poly,
        cutoff_t=start_day_number,
        model_kwargs=model_kwargs,
    )

    logger.info("Setting validation grid")
    vb.set_grid(250, num_sample_points)
    file_stem = os.path.join(out_dir, crime_type + '_' + '%d-%d' % (max_delta_t, max_delta_d))
    try:
        logger.info("Starting validation run.")
        res = vb.run(time_step=1, n_iter=this_num_validation, verbose=True, train_kwargs={'niter': niter})
    except Exception as exc:
        logger.error(repr(exc))
        res = None
    finally:
        logger.info("Saving results (or None).")
        with open(file_stem + '-validation.pickle', 'w') as f:
            pickle.dump(res, f)
        with open(file_stem + '-vb_obj.pickle', 'w') as f:
            pickle.dump(vb, f)


def chicago_south_side(max_delta_t, max_delta_d, crime_type):
    location_dir = 'chicago_south'
    poly_file = os.path.join(IN_DIR, 'boundaries.pickle')
    with open(poly_file, 'r') as f:
        boundaries = pickle.load(f)
        location_poly = boundaries['chicago_south']

    # min_bandwidths not required
    model_kwargs['trigger_kde_kwargs']['min_bandwidth'] = model_kwargs['bg_kde_kwargs']['min_bandwidth'] = None

    run_me(location_dir, location_poly, max_delta_t, max_delta_d, crime_type)


def camden(max_delta_t, max_delta_d, crime_type):
    location_dir = 'camden'
    poly_file = os.path.join(IN_DIR, 'boundaries.pickle')
    with open(poly_file, 'r') as f:
        boundaries = pickle.load(f)
        location_poly = boundaries['camden']

    # set min_bandwidths
    model_kwargs['trigger_kde_kwargs']['min_bandwidth'] = model_kwargs['bg_kde_kwargs']['min_bandwidth'] = \
        min_bandwidth_by_crime_type_camden[crime_type]

    run_me(location_dir, location_poly, max_delta_t, max_delta_d, crime_type)


if __name__ == '__main__':
    assert len(sys.argv) == 5, "Four input arguments required"
    crime_type = sys.argv[2]
    t = float(sys.argv[3])
    d = float(sys.argv[4])
    if sys.argv[1] == 'camden':
        camden(t, d, crime_type)
    elif sys.argv[1] == 'chicago_south':
        chicago_south_side(t, d, crime_type)
