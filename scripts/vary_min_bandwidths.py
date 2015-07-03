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
    'cd': 0.02
    # 'ct': 1/7.,
    # 'cd': 1/2000.
}

model_kwargs = {
    'parallel': False,
    'max_delta_t': 120,
    'max_delta_d': 1000,
    'bg_kde_kwargs': {'number_nn': [100, 15],
                      'min_bandwidth': None,  # replace this on each iteration
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


def run_me(location_dir, location_poly, min_bandwidth, crime_type, bjiggle=False):
    if bjiggle:
        loc_file = '%s_jiggle.pickle' % crime_type
    else:
        loc_file = '%s.pickle' % crime_type
    data_file = os.path.join(IN_DIR, location_dir, loc_file)
    out_dir = os.path.join(OUT_DIR, location_dir + ('_jiggle' if bjiggle else ''), 'min_bandwidth')
    log_file = os.path.join(out_dir, crime_type + '_' + '-'.join(['%.2f' % t for t in min_bandwidth]) + '.log')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # set loggers
    logger = logging.getLogger('vary_min_bandwidth.%s' % location_dir)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    for t in ('point_process.models', 'validation.validation', 'kde.models'):
        this_logger = logging.getLogger(t)
        this_logger.setLevel(logging.DEBUG)
        this_logger.handlers = []  # clear existing handlers
        this_logger.addHandler(fh)  # replace with the same file output

    logger.info("Logger set.  Script started.")
    logger.info("Crime type: %s. Min bandwidths %s. Jiggle: %s" % (crime_type, str(min_bandwidth), str(bjiggle)))

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

    model_kwargs['trigger_kde_kwargs']['min_bandwidth'] = [min_bandwidth[0], min_bandwidth[1], min_bandwidth[1]]
    model_kwargs['bg_kde_kwargs']['min_bandwidth'] = model_kwargs['trigger_kde_kwargs']['min_bandwidth']
    logger.info("Instantiating validation object")
    vb = validate.SeppValidationFixedModelIntegration(
        data=data,
        pp_class=pp_models.SeppStochasticNn,
        spatial_domain=location_poly,
        cutoff_t=start_day_number,
        model_kwargs=model_kwargs,
    )

    logger.info("Setting validation grid")
    vb.set_sample_units(250, num_sample_points)
    file_stem = os.path.join(out_dir, crime_type + '_' + '-'.join(['%.2f' % t for t in min_bandwidth]))
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


def chicago_south_side(min_bandwidth, crime_type, bjiggle):
    location_dir = 'chicago_south'
    poly_file = os.path.join(IN_DIR, 'boundaries.pickle')
    with open(poly_file, 'r') as f:
        boundaries = pickle.load(f)
        location_poly = boundaries['chicago_south']
    run_me(location_dir, location_poly, min_bandwidth, crime_type, bjiggle)


def camden(min_bandwidth, crime_type, bjiggle):
    location_dir = 'camden'
    poly_file = os.path.join(IN_DIR, 'boundaries.pickle')
    with open(poly_file, 'r') as f:
        boundaries = pickle.load(f)
        location_poly = boundaries['camden']
    run_me(location_dir, location_poly, min_bandwidth, crime_type, bjiggle)


# def chicago_south_side(min_bandwidth, crime_type):
#     data_file = os.path.join(IN_DIR, 'chicago_south', '%s.pickle' % crime_type)
#     poly_file = os.path.join(IN_DIR, 'boundaries.pickle')
#     out_dir = os.path.join(OUT_DIR, 'chicago_south', 'min_bandwidth')
#     log_file = os.path.join(out_dir, crime_type + '-' + '-'.join(['%.2f' % t for t in min_bandwidth]) + '.log')
#
#     if not os.path.isdir(out_dir):
#         os.makedirs(out_dir)
#
#     # set loggers
#     logger = logging.getLogger('vary_min_bandwidths.chicago_south_side')
#     logger.setLevel(logging.DEBUG)
#     fh = logging.FileHandler(log_file)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)
#
#     for t in ('point_process.models', 'validation.validation'):
#         this_logger = logging.getLogger(t)
#         this_logger.setLevel(logging.DEBUG)
#         this_logger.handlers = []  # clear existing handlers
#         this_logger.addHandler(fh)  # replace with the same file output
#
#     logger.info("Logger set.  Script started.")
#     logger.info("Crime type: %s. Min bandwidths %s" % (crime_type, str(min_bandwidth)))
#
#     # load data
#     with open(data_file, 'r') as f:
#         data = pickle.load(f)
#     with open(poly_file, 'r') as f:
#         boundaries = pickle.load(f)
#         south_side_poly = boundaries['chicago_south']
#     logger.info("Loaded data.")
#
#     # check that num_validation iterations is feasible
#     if start_day_number + num_validation - 1 > data[-1, 0]:
#         this_num_validation = int(data[-1, 0]) - start_day_number + 1
#         logger.info("Can only do %d validation runs" % this_num_validation)
#     else:
#         this_num_validation = num_validation
#
#     model_kwargs['trigger_kde_kwargs']['min_bandwidth'] = [min_bandwidth[0], min_bandwidth[1], min_bandwidth[1]]
#     model_kwargs['bg_kde_kwargs']['min_bandwidth'] = model_kwargs['trigger_kde_kwargs']['min_bandwidth']
#     logger.info("Instantiating validation object")
#     vb = validate.SeppValidationFixedModelIntegration(
#         data=data,
#         pp_class=pp_models.SeppStochasticNn,
#         spatial_domain=south_side_poly,
#         cutoff_t=start_day_number,
#         model_kwargs=model_kwargs,
#     )
#
#     logger.info("Setting validation grid")
#     vb.set_sample_units(250, num_sample_points)
#     try:
#         logger.info("Starting validation run.")
#         res = vb.run(time_step=1, n_iter=this_num_validation, verbose=True, train_kwargs={'niter': niter})
#     except Exception as exc:
#         logger.error(repr(exc))
#         raise exc
#     else:
#         logger.info("Saving results.")
#         file_stem = os.path.join(out_dir, crime_type + '_' + '-'.join(['%.2f' % t for t in min_bandwidth]))
#         with open(file_stem + '-validation.pickle', 'w') as f:
#             pickle.dump(res, f)
#         with open(file_stem + '-vb_obj.pickle', 'w') as f:
#             pickle.dump(vb, f)
#
#
# def camden(min_bandwidth, crime_type):
#     data_file = os.path.join(IN_DIR, 'camden', '%s.pickle' % crime_type)
#     poly_file = os.path.join(IN_DIR, 'boundaries.pickle')
#     out_dir = os.path.join(OUT_DIR, 'camden', 'min_bandwidth')
#     log_file = os.path.join(out_dir, crime_type + '-' + '-'.join(['%.2f' % t for t in min_bandwidth]) + '.log')
#
#     if not os.path.isdir(out_dir):
#         os.makedirs(out_dir)
#
#     # set loggers
#     logger = logging.getLogger('vary_min_bandwidths.camden')
#     logger.setLevel(logging.DEBUG)
#     fh = logging.FileHandler(log_file)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)
#
#     for t in ('point_process.models', 'validation.validation'):
#         this_logger = logging.getLogger(t)
#         this_logger.setLevel(logging.DEBUG)
#         this_logger.handlers = []  # clear existing handlers
#         this_logger.addHandler(fh)  # replace with the same file output
#
#     logger.info("Logger set.  Script started.")
#     logger.info("Crime type: %s. Min bandwidths %s" % (crime_type, str(min_bandwidth)))
#
#     # load data
#     with open(data_file, 'r') as f:
#         data = pickle.load(f)
#     with open(poly_file, 'r') as f:
#         boundaries = pickle.load(f)
#         boundary_poly = boundaries['camden']
#     logger.info("Loaded data.")
#
#     # check that num_validation iterations is feasible
#     if start_day_number + num_validation - 1 > data[-1, 0]:
#         this_num_validation = int(data[-1, 0]) - start_day_number + 1
#         logger.info("Can only do %d validation runs" % this_num_validation)
#     else:
#         this_num_validation = num_validation
#
#     model_kwargs['trigger_kde_kwargs']['min_bandwidth'] = [min_bandwidth[0], min_bandwidth[1], min_bandwidth[1]]
#     model_kwargs['bg_kde_kwargs']['min_bandwidth'] = model_kwargs['trigger_kde_kwargs']['min_bandwidth']
#     logger.info("Instantiating validation object")
#     vb = validate.SeppValidationFixedModelIntegration(
#         data=data,
#         pp_class=pp_models.SeppStochasticNn,
#         spatial_domain=boundary_poly,
#         cutoff_t=start_day_number,
#         model_kwargs=model_kwargs,
#     )
#
#     logger.info("Setting validation grid")
#     vb.set_sample_units(250, num_sample_points)
#     try:
#         logger.info("Starting validation run.")
#         res = vb.run(time_step=1, n_iter=this_num_validation, verbose=True, train_kwargs={'niter': niter})
#     except Exception as exc:
#         logger.error(repr(exc))
#         raise exc
#     else:
#         logger.info("Saving results.")
#         file_stem = os.path.join(out_dir, crime_type + '_' + '-'.join(['%.2f' % t for t in min_bandwidth]))
#         with open(file_stem + '-validation.pickle', 'w') as f:
#             pickle.dump(res, f)
#         with open(file_stem + '-vb_obj.pickle', 'w') as f:
#             pickle.dump(vb, f)


if __name__ == '__main__':

    location_dict = {
        'camden': camden,
        'chicago_south': chicago_south_side,
    }

    try:
        opts, args = getopt.getopt(sys.argv[1:], "jl:c:", ["jiggle", "location=", "ct="])
    except getopt.GetoptError:
        sys.exit(2)

    bjiggle = False
    crime_type = None
    func = None

    for opt, arg in opts:
        if opt in ('-j', '--jiggle'):
            bjiggle = True
        elif opt in ('-l', '--location'):
            try:
                func = location_dict[arg]
            except KeyError:
                print "Unrecognised location. Acceptable options are: %s" % ', '.join(location_dict.keys())
                sys.exit(2)
        elif opt in ('-c', '--ct'):
            crime_type = arg

    assert crime_type, "Must define crime_type with -c or --ct"
    assert func, "Must define location with -l or --location"

    # pick up remaining args
    assert len(args) == 2, "Two unlabelled input args required"
    t = float(args[0])
    d = float(args[1])

    func([t, d], crime_type, bjiggle)


