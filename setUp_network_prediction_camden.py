__author__ = 'gabriel'
from network.itn import read_gml, ITNStreetNet
from network.streetnet import NetPath, NetPoint, Edge, GridEdgeIndex
from data import models
import os
import pickle
from shapely import geometry
import settings
import numpy as np
from matplotlib import pyplot as plt
from network.utils import network_linkages, network_walker, network_walker_from_net_point
from validation import hotspot, roc
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


def jiggle_points():

    burglary = load_burglary()

    snapped = []
    free = []

    for t in burglary:
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

def plot_before_after_jiggle():
    snapped, snapped_jiggled, free = jiggle_points()
    plt.figure()
    camden_network_plot()
    plt.scatter(snapped[:, 1], snapped[:, 2], c='g')
    plt.scatter(free[:, 1], free[:, 2], c='b')

    plt.figure()
    camden_network_plot()
    plt.scatter(snapped_jiggled[:, 1], snapped_jiggled[:, 2], c='g')
    plt.scatter(free[:, 1], free[:, 2], c='b')


def snap_to_network():
    itn_net_reduced = network_from_pickle()
    snapped, snapped_jiggled, free = jiggle_points()
    free = models.CartesianSpaceTimeData(free)

    post_snap = free.time.adddim(
        models.NetworkData.from_cartesian(itn_net_reduced, free.space, grid_size=50),
        type=models.NetworkSpaceTimeData
    )

    return free, post_snap


def plot_snapping():

    free, free_snapped = snap_to_network()
    itn_net_reduced = network_from_pickle()

    x_pre, y_pre = free.space.separate
    x_post, y_post = free_snapped.space.to_cartesian().separate

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # itn_net_reduced.plot_network(ax=ax, edge_width=7, edge_inner_col='w')
    itn_net_reduced.plot_network(ax=ax)
    ax.scatter(x_pre, y_pre, c='r')
    ax.plot(x_post, y_post, 'ko')
    [ax.plot([x_pre[i], x_post[i]], [y_pre[i], y_post[i]], 'r-') for i in range(free.ndata)]

itn_net = network_from_pickle()
pt = np.array([529760, 182240])
buff = 1500
datum = models.NetPoint.from_cartesian(itn_net, *pt)
pt = np.array(datum.cartesian_coords)
bbox = geometry.Polygon([
    pt - np.array([buff/2., buff/2.]),
    pt + np.array([-buff/2., buff/2.]),
    pt + np.array([buff/2., buff/2.]),
    pt + np.array([buff/2., -buff/2.]),
    pt - np.array([buff/2., buff/2.]),
])
h = 200  # metres

itn_box = itn_net.within_boundary(bbox)
plt.figure()
itn_box.plot_network()
plt.scatter(*datum.cartesian_coords)
plt.axis([pt[0]-400, pt[0]+400, pt[1]-400, pt[1]+400])
circ = geometry.Point(*pt).buffer(h, resolution=128)
plotting.spatial.plot_shapely_geos(circ, fc='r', ec='none', alpha=0.3)
g = network_walker_from_net_point(itn_box, datum, max_distance=h, repeat_edges=False)
paths, dists, edges = zip(*list(g))