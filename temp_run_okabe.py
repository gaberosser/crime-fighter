__author__ = 'gabriel'
from network.itn import read_gml, ITNStreetNet
from network.streetnet import NetPath
from data import models
import os
import unittest
import settings
import numpy as np
from matplotlib import pyplot as plt
from network.utils import network_linkages
from validation import hotspot, roc
import plotting
from kde import okabe

if __name__ == '__main__':

    rng = np.random.RandomState(42)

    this_dir = os.path.dirname(os.path.realpath(__file__))
    IN_FILE = os.path.join('network', 'test_data', 'mastermap-itn_417209_0_brixton_sample.gml')
    test_data = read_gml(IN_FILE)
    itn_net = ITNStreetNet.from_data_structure(test_data)

    # buffered Camden dataset from raw data
    # test dataset is in a directory in the data directory called 'network_data'
    # this_dir = os.path.join(settings.DATA_DIR, 'network_data')
    # IN_FILE = os.path.join(this_dir, 'mastermap-itn_544003_0_camden_buff2000.gml')
    # test_data = read_gml(IN_FILE)
    # itn_net = ITNStreetNet.from_data_structure(test_data)

    # buffered Camden dataset from pickle
    # this_dir = os.path.dirname(os.path.realpath(__file__))
    # IN_FILE = os.path.join(this_dir, 'test_data', 'mastermap-itn_544003_0_camden_buff2000.pickle')
    # itn_net = ITNStreetNet.from_pickle(IN_FILE)

    # get the spatial extent of the network

    xmin, ymin, xmax, ymax = itn_net.extent

    # lay down some random points within that box
    num_pts = 100

    x_pts = rng.rand(num_pts) * (xmax - xmin) + xmin
    y_pts = rng.rand(num_pts) * (ymax - ymin) + ymin

    # now we want to snap them all to the network...
    # method A: do it in two steps...

    # A1: push them into a single data array for easier operation
    xy = models.DataArray.from_args(x_pts, y_pts)
    # A2: use the class method from_cartesian,
    net_point_array_a = models.NetworkData.from_cartesian(itn_net, xy, grid_size=50)  # grid_size defaults to 50

    source = net_point_array_a[0,0]
    targets = net_point_array_a.getrows(range(3,50))

    res = okabe.network_linkage(itn_net, source, targets, 500)

    itn_net.plot_network()
    plt.plot(source.cartesian_coords[0], source.cartesian_coords[1], 'bo')

    for i, t in enumerate(targets.toarray(0)):
        plt.plot(t.cartesian_coords[0], t.cartesian_coords[1], 'go')
        plt.text(t.cartesian_coords[0], t.cartesian_coords[1], str(i), fontsize=18, color='y')

    for n in itn_net.nodes():
        loc = itn_net.g.node[n]['loc']
        plt.plot(loc[0], loc[1], 'kx', markersize=10, linewidth=2.5)

    for k, v in zip(res.keys(), res.values()):
        loc = targets[k, 0].cartesian_coords
        plt.text(loc[0], loc[1], str(k) + ':' + ','.join(['%d' % x for x in sorted([y[1] for y in v])]), fontsize=18)