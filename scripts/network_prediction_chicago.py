from network.osm import read_data, OSMStreetNet
from data import models
import os
import pickle
from shapely import geometry
import settings
import numpy as np
from matplotlib import pyplot as plt
from validation import validation, hotspot, roc
import plotting.spatial
from analysis import chicago

INITAL_CUTOFF = 211

def network_from_pickle(domain_name):
    IN_FILE = '%s_clipped.pickle' % domain_name
    net_reduced = OSMStreetNet.from_pickle(IN_FILE)
    return net_reduced


def network_from_db(domain, srid=None, buffer=None):
    from django.db import connection
    from database.models import SRID
    if buffer:
        assert srid is not None, "If buffering is required, an SRID must be specified"


def load_network_and_pickle(domain, domain_name):

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


def network_heatmap(model,
                    points,
                    points_per_edge,
                    t,
                    fmax=0.95,
                    ax=None):
    """
    Plot showing spatial density on a network.
    :param net: StreetNet instance
    :param model: Must have a pdf method that accepts a NetworkData object
    :param t: Time
    :param dr: Distance between network points
    :return:
    """
    from network.plots import colorline
    from network.utils import network_point_coverage
    z = model.predict(t, points)

    norm = plt.Normalize(0., np.nanmax(z[~np.isinf(z)]) * fmax)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    i = 0
    for j, n in enumerate(points_per_edge):
        # get x,y coords
        this_sample_points = points.getrows(range(i, i + n))
        this_sample_points_xy = this_sample_points.to_cartesian()
        this_res = z[range(i, i + n)]

        colorline(this_sample_points_xy.toarray(0),
                  this_sample_points_xy.toarray(1),
                  z=this_res,
                  linewidth=5,
                  cmap=plt.get_cmap('coolwarm'),
                  alpha=0.9,
                  norm=norm
        )
        i += n

    points.graph.plot_network(edge_width=2, edge_inner_col='w')
    # plt.axis([pt[0]-400, pt[0]+400, pt[1]-400, pt[1]+400])

    # ss = ((training.toarray(0) / max(training.toarray(0)))**2 * 200).astype(float)
    # plt.scatter(*training.space.to_cartesian().separate, s=ss, facecolors='none', edgecolors='k', alpha=0.6, zorder=11)

    plt.tight_layout()



if __name__ == "__main__":

    itn_net = network_from_pickle()
    data = load_burglary()
    free, post_snap, post_snap_jiggled = snap_to_network(itn_net, data)
    # combine
    all_data = models.NetworkSpaceTimeData(np.vstack((post_snap_jiggled.data, post_snap.data)))

    psx, psy = all_data.space.to_cartesian().separate

    n_test = 100  # number of testing days
    h = 200 # metres
    t_decay = 30 # days
    grid_length = 250
    n_samples = 20

    sk = hotspot.STNetworkLinearSpaceExponentialTime(radius=h, time_decay=t_decay)
    vb = validation.NetworkValidationMean(all_data, sk, spatial_domain=None, include_predictions=True)
    vb.set_t_cutoff(INITAL_CUTOFF)
    vb.set_sample_units(None, n_samples)  # 2nd argument refers to interval between sample points

    import time
    tic = time.time()
    vb_res = vb.run(1, n_iter=n_test)
    toc = time.time()
    print toc - tic

    # compare with grid-based method with same parameters
    cb_poly = camden_boundary()
    data_txy = all_data.time.adddim(all_data.space.to_cartesian(), type=models.CartesianSpaceTimeData)
    sk_planar = hotspot.STLinearSpaceExponentialTime(radius=h, mean_time=t_decay)
    vb_planar = validation.ValidationIntegration(data_txy, sk_planar, spatial_domain=cb_poly, include_predictions=True)
    vb_planar.set_t_cutoff(INITAL_CUTOFF)
    vb_planar.set_sample_units(grid_length, n_samples)

    tic = time.time()
    vb_res_planar = vb_planar.run(1, n_iter=n_test)
    toc = time.time()
    print toc - tic


    # compare with grid-based method using intersecting network segments to measure sample unit size
    vb_planar_segment = validation.ValidationIntegrationByNetworkSegment(
        data_txy, sk_planar, spatial_domain=cb_poly, graph=itn_net
    )
    vb_planar_segment.set_t_cutoff(INITAL_CUTOFF)
    vb_planar_segment.set_sample_units(grid_length, n_samples)

    tic = time.time()
    vb_res_planar_segment = vb_planar_segment.run(1, n_iter=n_test)
    toc = time.time()
    print toc - tic