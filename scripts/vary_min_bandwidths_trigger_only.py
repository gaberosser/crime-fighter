__author__ = 'gabriel'
import datetime
import os
# import pickle
import dill as pickle
import logging
import sys
from point_process import models as pp_models, estimation, validate
# from settings import OUT_DIR, IN_DIR
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
    'max_delta_t': 60,
    'max_delta_d': 400,
    'bg_kde_kwargs': {'number_nn': [100, 15],
                      'min_bandwidth': None,  # doesn't change
                      'strict': False},
    'trigger_kde_kwargs': {'number_nn': 15,
                           'min_bandwidth': None,  # replace this on each iteration
                           'strict': False},
    'estimation_function': lambda x, y: estimation.estimator_bowers(x, y, **estimate_kwargs),
    'seed': 42,  # doesn't matter what this is, just want it fixed
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


def chicago_south_side(min_bandwidth, crime_type):
    data_file = os.path.join(IN_DIR, 'chicago_south', '%s.pickle' % crime_type)
    poly_file = os.path.join(IN_DIR, 'boundaries.pickle')
    out_dir = os.path.join(OUT_DIR, 'chicago_south', 'min_bandwidth_trigger_only')
    log_file = os.path.join(out_dir, crime_type + '-' + '-'.join(['%.2f' % t for t in min_bandwidth]) + '.log')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # set loggers
    logger = logging.getLogger('vary_min_bandwidths.chicago_south_side')
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
    logger.info("Crime type: %s. Min bandwidths %s" % (crime_type, str(min_bandwidth)))

    # load data
    with open(data_file, 'r') as f:
        data = pickle.load(f)
    with open(poly_file, 'r') as f:
        boundaries = pickle.load(f)
        south_side_poly = boundaries['chicago_south']
    logger.info("Loaded data.")

    # check that num_validation iterations is feasible
    if start_day_number + num_validation - 1 > data[-1, 0]:
        this_num_validation = int(data[-1, 0]) - start_day_number + 1
        logger.info("Can only do %d validation runs" % this_num_validation)
    else:
        this_num_validation = num_validation

    model_kwargs['trigger_kde_kwargs']['min_bandwidth'] = [min_bandwidth[0], min_bandwidth[1], min_bandwidth[1]]
    # model_kwargs['bg_kde_kwargs']['min_bandwidth'] = model_kwargs['trigger_kde_kwargs']['min_bandwidth']
    logger.info("Instantiating validation object")
    vb = validate.SeppValidationFixedModelIntegration(
        data=data,
        pp_class=pp_models.SeppStochasticNn,
        spatial_domain=south_side_poly,
        cutoff_t=start_day_number,
        model_kwargs=model_kwargs,
    )

    logger.info("Setting validation grid")
    vb.set_sample_units(250, num_sample_points)
    try:
        logger.info("Starting validation run.")
        res = vb.run(time_step=1, n_iter=this_num_validation, verbose=True, train_kwargs={'niter': niter})
    except Exception as exc:
        logger.error(repr(exc))
        raise exc
    else:
        logger.info("Saving results.")
        file_stem = os.path.join(out_dir, crime_type + '_' + '-'.join(['%.2f' % t for t in min_bandwidth]))
        with open(file_stem + '-validation.pickle', 'w') as f:
            pickle.dump(res, f)
        with open(file_stem + '-vb_obj.pickle', 'w') as f:
            pickle.dump(vb, f)


def camden(min_bandwidth, crime_type):
    data_file = os.path.join(IN_DIR, 'camden', '%s.pickle' % crime_type)
    poly_file = os.path.join(IN_DIR, 'boundaries.pickle')
    out_dir = os.path.join(OUT_DIR, 'camden', 'min_bandwidth_trigger_only')
    log_file = os.path.join(out_dir, crime_type + '-' + '-'.join(['%.2f' % t for t in min_bandwidth]) + '.log')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # set loggers
    logger = logging.getLogger('vary_min_bandwidths.camden')
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
    logger.info("Crime type: %s. Min bandwidths %s" % (crime_type, str(min_bandwidth)))

    # load data
    with open(data_file, 'r') as f:
        data = pickle.load(f)
    with open(poly_file, 'r') as f:
        boundaries = pickle.load(f)
        boundary_poly = boundaries['camden']
    logger.info("Loaded data.")

    # check that num_validation iterations is feasible
    if start_day_number + num_validation - 1 > data[-1, 0]:
        this_num_validation = int(data[-1, 0]) - start_day_number + 1
        logger.info("Can only do %d validation runs" % this_num_validation)
    else:
        this_num_validation = num_validation

    model_kwargs['trigger_kde_kwargs']['min_bandwidth'] = [min_bandwidth[0], min_bandwidth[1], min_bandwidth[1]]
    # model_kwargs['bg_kde_kwargs']['min_bandwidth'] = model_kwargs['trigger_kde_kwargs']['min_bandwidth']
    logger.info("Instantiating validation object")
    vb = validate.SeppValidationFixedModelIntegration(
        data=data,
        pp_class=pp_models.SeppStochasticNn,
        spatial_domain=boundary_poly,
        cutoff_t=start_day_number,
        model_kwargs=model_kwargs,
    )

    logger.info("Setting validation grid")
    vb.set_sample_units(250, num_sample_points)
    try:
        logger.info("Starting validation run.")
        res = vb.run(time_step=1, n_iter=this_num_validation, verbose=True, train_kwargs={'niter': niter})
    except Exception as exc:
        logger.error(repr(exc))
        raise exc
    else:
        logger.info("Saving results.")
        file_stem = os.path.join(out_dir, crime_type + '_' + '-'.join(['%.2f' % t for t in min_bandwidth]))
        with open(file_stem + '-validation.pickle', 'w') as f:
            pickle.dump(res, f)
        with open(file_stem + '-vb_obj.pickle', 'w') as f:
            pickle.dump(vb, f)


if __name__ == '__main__':
    assert len(sys.argv) == 5, "Four input arguments required"
    crime_type = sys.argv[2]
    t = float(sys.argv[3])
    d = float(sys.argv[4])
    if sys.argv[1] == 'camden':
        camden([t, d], crime_type)
    elif sys.argv[1] == 'chicago_south':
        chicago_south_side([t, d], crime_type)
