from data.models import DataArray, NetworkSpaceTimeData
import dill
import numpy as np
from validation import validation, hotspot
from network import itn
import datetime
from time import time
import sys
import logging
import os
import fiona
from scripts import OUT_DIR

DATA_DIR = '/media/gabriel/DATAPART1/data/monsuru/network_data_joined_from_toby/'

START_DATE = datetime.date(2011, 9, 29)
START_DAY_NUMBER = 212
D_BANDWIDTH_NETWORK = 550.
T_BANDWIDTH_NETWORK = 70.
D_BANDWIDTH_PLANAR = 610.
T_BANDWIDTH_PLANAR = 55.
NUM_VALIDATION = 100
BETWEEN_SAMPLE_POINTS = 30
GRID_LENGTH = 150
N_SAMPLES_PER_GRID = 15


def load_data(
    crime_type='burglary',
    start_date=START_DATE,
    end_date=None
):
    file_name_map = {
        'burglary': 'burg_D.shp',
        'violence': 'vio_D.shp',
        'shoplifting': 'shop_D.shp'
    }
    fin = file_name_map[crime_type.lower()]
    fn = os.path.join(DATA_DIR, fin)
    xs = []
    ys = []
    dates = []
    with fiona.open(fn, 'r') as sf:
        for r in sf:
            date = datetime.datetime.strptime(r['properties']['Date3'], '%Y-%m-%d').date()
            x, y = r['geometry']['coordinates']
            dates.append(date)
            xs.append(x)
            ys.append(y)

    t0 = min(dates)
    days = [(t - t0).days for t in dates]
    txy = np.vstack((days, xs, ys)).transpose()
    return txy, t0


def load_network():
    fin = 'cm_road_net.shp'
    fn = os.path.join(DATA_DIR, fin)
    return itn.ITNStreetNet.from_shapefile(fn)


def snap_data(data, net):

    # snap space
    snapped_xy, failed = NetworkSpaceTimeData.from_cartesian(net, data, return_failure_idx=True)
    return snapped_xy
    # not_failed = sorted(list(set(range(data.shape[0])) - failed))

    # res = DataArray(data[not_failed, 0]).adddim(snapped_xy, type=NetworkSpaceTimeData)

    # return res


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

    out_dir = os.path.join(OUT_DIR, 'cad_old', 'monsuru_network_prediction')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, 'results.dill')

    end_date = START_DATE + datetime.timedelta(days=START_DAY_NUMBER + NUM_VALIDATION + 1)
    net = load_network()
    res = {}

    for ct in ('burglary', 'violence', 'shoplifting'):

        data, t0 = load_data(start_date=START_DATE, end_date=end_date, crime_type=ct)
        data_snap = snap_data(data, net)
        res[ct] = run_network_validation(data_snap)


    with open(out_file, 'w') as f:
        dill.dump(res, f)

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
