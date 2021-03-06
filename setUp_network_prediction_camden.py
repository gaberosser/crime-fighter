from network.itn import read_gml, ITNStreetNet
from data import models
import os
import pickle
from shapely import geometry
import settings
import numpy as np
from matplotlib import pyplot as plt
from validation import validation, hotspot, roc
import plotting.spatial


def camden_boundary():
    with open(os.path.join(settings.DATA_DIR, 'camden', 'boundary_line.pickle'), 'r') as f:
        xy = pickle.load(f)
        camden = geometry.Polygon(zip(*xy))
    return camden


def network_from_pickle():
    IN_FILE = 'camden_clipped.pickle'
    itn_net_reduced = ITNStreetNet.from_pickle(IN_FILE)
    return itn_net_reduced


def load_network_and_pickle():
    # test dataset is in a directory in the data directory called 'network_data'
    this_dir = os.path.join(settings.DATA_DIR, 'network_data')
    IN_FILE = os.path.join(this_dir, 'mastermap-itn_544003_0_camden_buff2000.gml')
    test_data = read_gml(IN_FILE)
    itn_net = ITNStreetNet.from_data_structure(test_data)

    camden = camden_boundary()

    itn_net_reduced = itn_net.within_boundary(camden.buffer(100))
    itn_net.save('camden.pickle')
    itn_net_reduced.save('camden_clipped.pickle')


def camden_network_plot():

    itn_net_reduced = network_from_pickle()

    itn_net_reduced.plot_network()
    camden = camden_boundary()
    plotting.spatial.plot_shapely_geos(camden, ec='r', fc='none', lw=3)


def load_burglary():
    with open(os.path.join(settings.DATA_DIR, 'cad', 'burglary.pickle'), 'r') as f:
        burglary = pickle.load(f)
    return burglary


def jiggle_points(data):

    snapped = []
    free = []

    for t in data:
        if (t[1] % 25 == 0) and (t[2] % 25 == 0):
            snapped.append(t)
        else:
            free.append(t)

    snapped = np.array(snapped)
    free = np.array(free)

    snapped_jiggled = []
    for t in snapped:
        snapped_jiggled.append(t.copy())
        snapped_jiggled[-1][1:] += np.random.rand(2) * 250 - 125

    return snapped, snapped_jiggled, free


def snap_to_network(itn_net=None, data=None):
    if itn_net is None:
        itn_net = network_from_pickle()
    if data is None:
        data = load_burglary()
    snapped, snapped_jiggled, free = jiggle_points(data)
    free = models.CartesianSpaceTimeData(free)
    snapped_jiggled = models.CartesianSpaceTimeData(snapped_jiggled)

    post_snap = free.time.adddim(
        models.NetworkData.from_cartesian(itn_net, free.space, grid_size=50),
        type=models.NetworkSpaceTimeData
    )

    post_snap_jiggled = snapped_jiggled.time.adddim(
        models.NetworkData.from_cartesian(itn_net, snapped_jiggled.space, grid_size=50),
        type=models.NetworkSpaceTimeData
    )

    return free, post_snap, post_snap_jiggled


if __name__ == "__main__":

    itn_net = network_from_pickle()
    data = load_burglary()
    free, post_snap, post_snap_jiggled = snap_to_network(itn_net, data)
    # combine
    all_data = models.NetworkSpaceTimeData(np.vstack((post_snap_jiggled.data, post_snap.data)))

    psx, psy = all_data.space.to_cartesian().separate


    h = 400 # metres
    t_decay = 30 # days
    sk = hotspot.STNetworkLinearSpaceExponentialTime(radius=h, time_decay=t_decay)
    vb = validation.NetworkValidationMean(all_data, sk, spatial_domain=None, include_predictions=True)
    vb.set_t_cutoff(211)
    vb.set_sample_units(None, 20)  # 2nd argument refers to interval between sample points

    import time
    tic = time.time()
    vb_res = vb.run(1, n_iter=100)
    toc = time.time()
    print toc - tic

    # compare with grid-based method with same parameters
    cb_poly = camden_boundary()
    data_txy = all_data.time.adddim(all_data.space.to_cartesian(), type=models.CartesianSpaceTimeData)
    sk_planar = hotspot.STLinearSpaceExponentialTime(radius=400, mean_time=30.)
    vb_planar = validation.ValidationIntegration(data_txy, sk_planar, spatial_domain=cb_poly, include_predictions=True)
    vb_planar.set_t_cutoff(211)
    vb_planar.set_sample_units(250, 10)

    tic = time.time()
    vb_res_planar = vb_planar.run(1, n_iter=100)
    toc = time.time()
    print toc - tic


    # compare with grid-based method using intersecting network segments to measure sample unit size
    vb_planar_segment = validation.ValidationIntegrationByNetworkSegment(
        data_txy, sk_planar, spatial_domain=cb_poly, graph=itn_net
    )
    vb_planar_segment.set_t_cutoff(211)
    vb_planar_segment.set_sample_units(250, 10)

    tic = time.time()
    vb_res_planar_segment = vb_planar_segment.run(1, n_iter=100)
    toc = time.time()
    print toc - tic