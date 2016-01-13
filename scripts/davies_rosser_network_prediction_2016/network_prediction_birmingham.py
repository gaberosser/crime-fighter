from database.birmingham.loader import load_network, BirminghamCrimeFileLoader, load_boundary_file
from database.birmingham.consts import CRIME_DATA_FILE
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
D_BANDWIDTH_NETWORK = 820.
T_BANDWIDTH_NETWORK = 78.
D_BANDWIDTH_PLANAR = 610.
T_BANDWIDTH_PLANAR = 55.
NUM_VALIDATION = 180
BETWEEN_SAMPLE_POINTS = 30
GRID_LENGTH = 150
N_SAMPLES_PER_GRID = 15


def load_data(start_date=START_DATE,
              end_date=None):
    obj = BirminghamCrimeFileLoader(CRIME_DATA_FILE, fmt='csv')
    return obj.get_data(start_date=start_date,
                        end_date=end_date)


def snap_data(data, cid):

    # load network
    net = load_network()

    # snap
    snapped_data, failed = NetworkSpaceTimeData.from_cartesian(net, data, return_failure_idx=True)

    # filter cid to those that snapped correctly
    idx = sorted(list(set(range(data.shape[0])) - set(failed)))
    filtered_cid = np.array(cid)[idx]

    return snapped_data, filtered_cid


def run_network_validation(data):
    sk = hotspot.STNetworkLinearSpaceExponentialTime(radius=D_BANDWIDTH_NETWORK, time_decay=T_BANDWIDTH_NETWORK)
    vb = validation.NetworkValidationMean(data, sk, spatial_domain=None, include_predictions=True)
    vb.set_t_cutoff(START_DAY_NUMBER)
    vb.set_sample_units(None, BETWEEN_SAMPLE_POINTS)  # 2nd argument refers to interval between sample points

    tic = time()
    vb_res = vb.run(1, n_iter=NUM_VALIDATION)
    toc = time()
    print toc - tic
    return vb_res


def run_planar_validation_compare_by_grid(data):
    poly = load_boundary_file()
    sk_planar = hotspot.STLinearSpaceExponentialTime(radius=D_BANDWIDTH_PLANAR, mean_time=T_BANDWIDTH_PLANAR)
    vb_planar = validation.ValidationIntegration(data, sk_planar, spatial_domain=poly, include_predictions=True)
    vb_planar.set_t_cutoff(START_DAY_NUMBER)
    vb_planar.set_sample_units(GRID_LENGTH, N_SAMPLES_PER_GRID)

    tic = time()
    vb_res = vb_planar.run(1, n_iter=NUM_VALIDATION)
    toc = time()
    print toc - tic
    return vb_res


def run_planar_validation_compare_by_grid_network_intersection(data, net, debug=False):
    if debug:
        # set logger to verbose output
        sh = logging.StreamHandler()
        validation.roc.logger.handlers = [sh]
        validation.roc.logger.setLevel(logging.DEBUG)
        validation.roc.logger.debug("Initiated debugging logger for ROC")
    poly = load_boundary_file()
    sk_planar = hotspot.STLinearSpaceExponentialTime(radius=D_BANDWIDTH_PLANAR, mean_time=T_BANDWIDTH_PLANAR)
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

    end_date = START_DATE + datetime.timedelta(days=START_DAY_NUMBER + NUM_VALIDATION + 1)
    data, t0, cid = load_data(start_date=START_DATE, end_date=end_date)
    data_snap, cid_snap = snap_data(data, cid)

    res = {}
    res['network'] = run_network_validation(data_snap)
    with open(os.path.join(OUT_DIR, 'birmingham', 'network_validation_results.dill'), 'w') as f:
        dill.dump(res['network'], f)

    # res['planar_grid_comparison'] = run_planar_validation_compare_by_grid(data)
    # with open(os.path.join(OUT_DIR, 'birmingham', 'planar_grid_validation_results.dill'), 'w') as f:
    #     dill.dump(res['planar_grid_comparison'], f)

    # res['planar_net_comparison'] = run_planar_validation_compare_by_grid_network_intersection(data,
    #                                                                                           data_snap.graph,
    #                                                                                           debug=True)
    # with open(os.path.join(OUT_DIR, 'birmingham', 'planar_net_validation_results.dill'), 'w') as f:
    #     dill.dump(res['planar_net_comparison'], f)

    # with open(os.path.join(OUT_DIR, 'birmingham', 'planar_vs_net.dill'), 'w') as f:
    #     dill.dump(res, f)
