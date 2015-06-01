__author__ = 'gabriel'
import os
import sys
import dill as pickle
from . import OUT_DIR, IN_DIR
import logging


# Global parameters
num_sample_points = 20


def run_me(location_dir, crime_type, grid_size, max_delta_t, max_delta_d):


    data_dir = os.path.join(OUT_DIR, location_dir, 'max_triggers')
    out_dir = os.path.join(OUT_DIR, location_dir, 'max_triggers_grid-%d' % grid_size)
    log_file = os.path.join(out_dir, '%s_%d-%d.log' % (crime_type, max_delta_t, max_delta_d))

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # set loggers
    logger = logging.getLogger('vary_maximum_triggers_grid_size.%s' % location_dir)
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
    logger.info("Crime type: %s. Max delta t: %d, max delta d: %d. Grid size %d." % (crime_type,
                                                                                     max_delta_t,
                                                                                     max_delta_d,
                                                                                     grid_size))

    # load data
    data_file = os.path.join(data_dir, '%s_%d-%d-vb_obj.pickle' % (crime_type, max_delta_t, max_delta_d))
    with open(data_file, 'r') as f:
        vb = pickle.load(f)

    data_file = os.path.join(data_dir, '%s_%d-%d-validation.pickle' % (crime_type, max_delta_t, max_delta_d))
    with open(data_file, 'r') as f:
        vb_res = pickle.load(f)

    logger.info("Loaded data.")

    logger.info("Setting validation grid")
    vb.set_sample_units(grid_size, num_sample_points)

    file_stem = os.path.join(out_dir, crime_type + '_' + '%d-%d' % (max_delta_t, max_delta_d))
    res = None
    try:
        logger.info("Starting validation run.")
        res = vb.repeat_run(vb_res)
    except Exception as exc:
        logger.error(repr(exc))
    finally:
        logger.info("Saving results (or None).")
        with open(file_stem + '-validation.pickle', 'w') as f:
            pickle.dump(res, f)


def chicago_south(crime_type, grid_size, max_delta_t, max_delta_d):
    location_dir = 'chicago_south'
    run_me(location_dir, crime_type, grid_size, max_delta_t, max_delta_d)


def camden(crime_type, grid_size, max_delta_t, max_delta_d):
    location_dir = 'camden'
    run_me(location_dir, crime_type, grid_size, max_delta_t, max_delta_d)


if __name__ == '__main__':
    assert len(sys.argv) == 6, "Five input arguments required"
    crime_type = sys.argv[2]
    t = float(sys.argv[3])
    d = float(sys.argv[4])
    grid = float(sys.argv[5])
    if sys.argv[1] == 'camden':
        camden(crime_type, grid, t, d)
    elif sys.argv[1] == 'chicago_south':
        chicago_south(crime_type, grid, t, d)
