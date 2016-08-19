from database.birmingham.loader import load_network, BirminghamCrimeLoader, load_boundary_file
from data.models import NetworkSpaceTimeData
import dill
import numpy as np
from validation import validation, hotspot
import datetime
from time import time
import sys
import logging
from scripts import OUT_DIR
import os


START_DATE = datetime.date(2013, 7, 1)
START_DAY_NUMBER = 240
# D_BANDWIDTH_NETWORK = 820.
# T_BANDWIDTH_NETWORK = 78.
# D_BANDWIDTH_PLANAR = 610.
# T_BANDWIDTH_PLANAR = 55.
NUM_VALIDATION = 90
# BETWEEN_SAMPLE_POINTS = 30
BETWEEN_SAMPLE_POINTS = 100
GRID_LENGTH = 150
N_SAMPLES_PER_GRID = 15

## optimal bandwidths
## these come from running the mangle_birmingham_... script and copying the dictionary here
OPTIMAL_BANDWIDTHS = {
     ('network', 'end', None): (77.285714285714292, 925.51020408163254),
     ('network', 'start', None): (77.414141414141412, 916.66666666666663),
     ('network', 'start', 24): (79.102040816326536, 925.51020408163254),
     ('planar', 'end', None): (50.04081632653061, 806.12244897959181),
     ('planar', 'start', None): (50.04081632653061, 806.12244897959181),
     ('planar', 'start', 24): (53.673469387755105, 766.32653061224482)
}


def load_data(
    start_date=START_DATE,
    end_date=None,
    aoristic_method='start',
    max_time_window=None
):
    obj = BirminghamCrimeLoader(aoristic_method=aoristic_method, max_time_window=max_time_window)
    # obj = BirminghamCrimeFileLoader(CRIME_DATA_FILE, fmt='csv')
    return obj.get_data(start_date=start_date,
                        end_date=end_date)


def snap_data(data, cid):

    # load network
    net = load_network()

    # snap
    snapped_data, failed = NetworkSpaceTimeData.from_cartesian(net, data, return_failure_idx=True, grid_size=100, radius=100)

    # filter cid to those that snapped correctly
    idx = sorted(list(set(range(data.shape[0])) - set(failed)))
    filtered_cid = np.array(cid)[idx]

    return snapped_data, filtered_cid


def run_network_validation(data, topt, dopt):
    sk = hotspot.STNetworkLinearSpaceExponentialTime(radius=dopt, time_decay=topt)
    vb = validation.NetworkValidationMean(data, sk, spatial_domain=None, include_predictions=True)
    vb.set_t_cutoff(START_DAY_NUMBER)
    vb.set_sample_units(None, BETWEEN_SAMPLE_POINTS)  # 2nd argument refers to interval between sample points

    tic = time()
    vb_res = vb.run(1, n_iter=NUM_VALIDATION)
    toc = time()
    print toc - tic
    return vb_res


def run_planar_validation_compare_by_grid(data, topt, dopt):
    poly = load_boundary_file()
    sk_planar = hotspot.STLinearSpaceExponentialTime(radius=dopt, mean_time=topt)
    vb_planar = validation.ValidationIntegration(data, sk_planar, spatial_domain=poly, include_predictions=True)
    vb_planar.set_t_cutoff(START_DAY_NUMBER)
    vb_planar.set_sample_units(GRID_LENGTH, N_SAMPLES_PER_GRID)

    tic = time()
    vb_res = vb_planar.run(1, n_iter=NUM_VALIDATION)
    toc = time()
    print toc - tic
    return vb_res


def run_planar_validation_compare_by_grid_network_intersection(data, net, topt, dopt, debug=False):
    if debug:
        # set logger to verbose output
        sh = logging.StreamHandler()
        validation.roc.logger.handlers = [sh]
        validation.roc.logger.setLevel(logging.DEBUG)
        validation.roc.logger.debug("Initiated debugging logger for ROC")
    poly = load_boundary_file()
    sk_planar = hotspot.STLinearSpaceExponentialTime(radius=dopt, mean_time=topt)
    vb = validation.ValidationIntegrationByNetworkSegment(
        data, sk_planar, spatial_domain=poly, graph=net, include_predictions=True
    )
    vb.set_t_cutoff(START_DAY_NUMBER)
    vb.set_sample_units(GRID_LENGTH, N_SAMPLES_PER_GRID)

    tic = time()
    vb_res = vb.run(1, n_iter=NUM_VALIDATION)
    toc = time()
    print toc - tic
    return vb_res


if __name__ == '__main__':

    # method can be one of 'net', 'grid', 'grid_net'
    method = 'net'
    aoristic_method = 'start'
    max_time_window = None

    filestem = '_validation_start_day_%d_%d_iterations_%s%s.dill' % (
        START_DAY_NUMBER,
        NUM_VALIDATION,
        aoristic_method,
        '_max%dhours' % max_time_window if max_time_window is not None else ''
    )

    # get optimal bandwidths
    if method == 'net':
        topt, dopt = OPTIMAL_BANDWIDTHS[('network', aoristic_method, max_time_window)]
    else:
        topt, dopt = OPTIMAL_BANDWIDTHS[('planar', aoristic_method, max_time_window)]

    end_date = START_DATE + datetime.timedelta(days=START_DAY_NUMBER + NUM_VALIDATION + 1)
    data, t0, cid = load_data(
        start_date=START_DATE,
        end_date=end_date,
        aoristic_method=aoristic_method,
        max_time_window=max_time_window
    )

    if method == 'net':
        data_snap, cid_snap = snap_data(data, cid)
        res = run_network_validation(data_snap, topt, dopt)
        fn = 'net' + filestem
    elif method == 'grid':
        res = run_planar_validation_compare_by_grid(data, topt, dopt)
        fn = 'grid' + filestem
    elif method == 'net_grid':
        net = load_network()
        res = run_planar_validation_compare_by_grid_network_intersection(
            data,
            net,
            topt=topt,
            dopt=dopt,
            debug=True)
        fn = 'grid_net' + filestem
    else:
        raise ValueError("Method not recognised")


    with open(os.path.join(OUT_DIR, 'birmingham', fn), 'w') as f:
        dill.dump(res, f)
