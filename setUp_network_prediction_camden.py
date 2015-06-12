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
from network.utils import network_linkages, network_walker, network_walker_from_net_point, network_walker_fixed_distance
from validation import hotspot, roc
import plotting.spatial
from kde import kernels, models as kmodels

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

def plot_before_after_jiggle():
    burglary = load_burglary()
    snapped, snapped_jiggled, free = jiggle_points(burglary)
    plt.figure()
    camden_network_plot()
    plt.scatter(snapped[:, 1], snapped[:, 2], c='g')
    plt.scatter(free[:, 1], free[:, 2], c='b')

    plt.figure()
    camden_network_plot()
    plt.scatter(snapped_jiggled[:, 1], snapped_jiggled[:, 2], c='g')
    plt.scatter(free[:, 1], free[:, 2], c='b')


def snap_to_network(itn_net=None, data=None):
    if itn_net is None:
        itn_net = network_from_pickle()
    if data is None:
        data = load_burglary()
    snapped, snapped_jiggled, free = jiggle_points(data)
    free = models.CartesianSpaceTimeData(free)

    post_snap = free.time.adddim(
        models.NetworkData.from_cartesian(itn_net, free.space, grid_size=50),
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

# recompute datum on the new network
datum = models.NetPoint.from_cartesian(itn_box, *pt)
circ = geometry.Point(*pt).buffer(h, resolution=128)

# max distance points and paths
eps, paths = network_walker_fixed_distance(itn_box, datum, h)
eps = models.NetworkData(eps)

# choose one path
plt.figure()
itn_box.plot_network()
plt.scatter(*datum.cartesian_coords, c='b', s=40, zorder=10)
plt.axis([pt[0]-400, pt[0]+400, pt[1]-400, pt[1]+400])

idx = 1
this_ep = eps.toarray(0)[idx]
dist_along = []
degrees = []

# first partial edge
init_ls = datum.linestring(paths[idx][0])
dist_along.append(init_ls.length)
degrees.append(2)
plt.plot(*init_ls.xy, c='b', lw=4)


# iterate over full edges
for i in range(len(paths[idx]) - 1):
    # get edge
    e = itn_box.g[paths[idx][i+1]][paths[idx][i]].values()[0]  # FIXME: horrible hack, we really need to track FIDs
    dist_along.append(e['linestring'].length)
    degrees.append(itn_box.g.degree(paths[idx][i]))
    plt.plot(*e['linestring'].xy, c='b', lw=4)

# final partial edge
final_ls = this_ep.linestring(paths[idx][-1])
dist_along.append(final_ls.length)
degrees.append(2)
plt.plot(*final_ls.xy, c='b', lw=4)

# plot end point
plt.scatter(*this_ep.cartesian_coords, c='r', s=40, zorder=10)

plt.tight_layout()

# plot the linear kernel along it
plt.figure()
dist_along = np.concatenate(([0], np.cumsum(dist_along)))
degrees = np.array(degrees)
curr_div = 1.
x = []
for i in range(len(dist_along) - 1):
    curr_div *= max(degrees[i] - 1., 1.)
    x = np.linspace(dist_along[i], dist_along[i+1], 500)
    y = 1.0 / (float(h) ** 2) * (h - x) / curr_div
    plt.plot(x, y, 'k-', lw=2)
plt.axis([0, h, 0, 1.02/float(h)])
plt.xlabel("Network distance")
plt.ylabel("Density")

# plt.figure()
# itn_box.plot_network()
# plt.scatter(*datum.cartesian_coords)
# plt.axis([pt[0]-400, pt[0]+400, pt[1]-400, pt[1]+400])
# plotting.spatial.plot_shapely_geos(circ, fc='r', ec='none', alpha=0.3)
# plt.scatter(*eps.to_cartesian().separate, c='r', s=40)
# plt.tight_layout()

## single kernel
# k = kernels.NetworkKernelEqualSplitLinear(datum, h)
# points, n_per_edge = network_point_coverage(itn_box, 10)
# kvals = k.pdf(points)
#
# fmax = 0.9
#
# norm = plt.Normalize(0., np.nanmax(kvals[~np.isinf(kvals)]) * fmax)
# fig = plt.figure()
# i = 0
# for j, n in enumerate(n_per_edge):
#     # get x,y coords
#     this_edge = itn_box.edges()[j]
#     this_sample_points = points.getrows(range(i, i + n))
#     this_sample_points_xy = this_sample_points.to_cartesian()
#     this_res = kvals[range(i, i + n)]
#
#     colorline(this_sample_points_xy.toarray(0),
#               this_sample_points_xy.toarray(1),
#               z=this_res,
#               linewidth=5,
#               cmap=plt.get_cmap('coolwarm'),
#               alpha=0.9,
#               norm=norm
#     )
#     i += n
#
# itn_box.plot_network(edge_width=2, edge_inner_col='w')
# plt.axis([pt[0]-400, pt[0]+400, pt[1]-400, pt[1]+400])
#
# # plotting.spatial.plot_shapely_geos(circ, fc='none', ec='k', alpha=0.7)
# plt.tight_layout()


## multiple sources in KDE
# grab all crimes inside the box
# data = load_burglary()
# data = np.array([t for t in data if geometry.Point(t[1:]).within(bbox)])
# free, post_snap = snap_to_network(itn_box, data)
# psx, psy = post_snap.space.to_cartesian().separate
#
# idx = [i for i in range(psx.size) if geometry.Point((psx[i], psy[i])).within(bbox)]
# training = post_snap.getrows(idx)
#
# kk = kmodels.NetworkFixedBandwidthKde(training, bandwidths=[60, 200], parallel=False)
#
# from network.plotting import network_point_coverage, colorline
# points, n_per_edge = network_point_coverage(itn_box, 10)
# points_st = models.DataArray(np.ones(points.ndata) * (training.toarray(0).max() + 0.1)).adddim(points, type=models.NetworkSpaceTimeData)
# kvals = kk.pdf(points_st)
#
# fmax = 0.9
#
# norm = plt.Normalize(0., np.nanmax(kvals[~np.isinf(kvals)]) * fmax)
# fig = plt.figure()
# i = 0
# for j, n in enumerate(n_per_edge):
#     # get x,y coords
#     this_edge = itn_box.edges()[j]
#     this_sample_points = points.getrows(range(i, i + n))
#     this_sample_points_xy = this_sample_points.to_cartesian()
#     this_res = kvals[range(i, i + n)]
#
#     colorline(this_sample_points_xy.toarray(0),
#               this_sample_points_xy.toarray(1),
#               z=this_res,
#               linewidth=5,
#               cmap=plt.get_cmap('coolwarm'),
#               alpha=0.9,
#               norm=norm
#     )
#     i += n
#
# itn_box.plot_network(edge_width=2, edge_inner_col='w')
# plt.axis([pt[0]-400, pt[0]+400, pt[1]-400, pt[1]+400])
#
# ss = ((training.toarray(0) / max(training.toarray(0)))**2 * 200).astype(float)
# plt.scatter(*training.space.to_cartesian().separate, s=ss, facecolors='none', edgecolors='k', alpha=0.6, zorder=11)
#
# plt.tight_layout()

## validation
data = load_burglary()
data = np.array([t for t in data if geometry.Point(t[1:]).within(bbox)])
free, post_snap = snap_to_network(itn_box, data)
psx, psy = post_snap.space.to_cartesian().separate

idx = [i for i in range(psx.size) if geometry.Point((psx[i], psy[i])).within(bbox)]
net_data = post_snap.getrows(idx)

from validation import validation, hotspot, roc

t_decay = 60 # days
sk = hotspot.STNetworkLinearSpaceExponentialTime(radius=h, time_decay=t_decay)
vb = validation.NetworkValidationMean(net_data, sk, spatial_domain=bbox)
vb.set_t_cutoff(375)
vb.set_sample_units(None, 20)  # 2nd argument refers to interval between sample points

vb_res = vb.run(1, n_iter=1)

small_bbox = geometry.Polygon([
    (pt[0] - 400, pt[1] - 400),
    (pt[0] - 400, pt[1] + 400),
    (pt[0] + 400, pt[1] + 400),
    (pt[0] + 400, pt[1] - 400),
    (pt[0] - 400, pt[1] - 400)
])

in_small_bbox = [i for i in range(vb.roc.n_sample_units) if vb.roc.sample_units[vb.roc.prediction_rank[i]].linestring.intersects(small_bbox)]

vb.roc.plot(show_sample_units=False)
for i in range(10):
    edge = vb.roc.sample_units[vb.roc.prediction_rank[in_small_bbox[i]]]
    cx, cy = edge.centroid_xy
    plt.text(cx, cy, str(i + 1), size=32, color='b')
plt.axis([pt[0]-400, pt[0]+400, pt[1]-400, pt[1]+400])
plt.tight_layout()

plt.rcParams['text.usetex'] = False
with plt.xkcd():
    x = np.linspace(0, 1, 500)
    y1 = x ** 0.5
    y2 = x ** 0.25
    plt.figure()
    plt.plot(x, y1, 'b-')
    plt.plot(x, y2, 'r-')
    plt.plot(x, x, 'k--')
    plt.xlabel('proportion of total network length covered', fontsize=16)
    plt.ylabel('proportion of total crime captured', fontsize=16)
    plt.tight_layout()

plt.rcParams['text.usetex'] = True