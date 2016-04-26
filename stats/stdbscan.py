import numpy as np
from data import models
from analysis import chicago, cad
from network import osm, itn
from settings import DATA_DIR
import os
import csv
import datetime
import dill
from network.utils import linkages, network_linkages, linkage_func_separable
import logging


LABEL_UNCLASSIFIED = -2
LABEL_NOISE = -1


# START_DATE = datetime.date(2010, 1, 1)
# END_DATE = datetime.date(2014, 1, 1)
#

domain = cad.get_camden_region(as_shapely=True)
net_file = os.path.join(DATA_DIR, 'camden', 'network', 'mastermap-itn_camden_buff100.net')
net = itn.ITNStreetNet.from_pickle(net_file)

data_fn = os.path.join(DATA_DIR, 'jianan', 'stay_0.csv')
fields = ('Seconds', 'EASTING', 'NORTHING')
with open(data_fn, 'rb') as f:
    c = csv.DictReader(f)
    raw_fields = c.fieldnames
    raw_data = list(c)
    raw_txy = np.array([[r[t] for t in fields] for r in raw_data], dtype=float)

# net = osm.OSMStreetNet.from_pickle(net_file)
# sides = chicago.get_chicago_side_polys(True)
# data, t0, cid = chicago.get_crimes_by_type(domain=sides['South'],
#                                            start_date=START_DATE,
#                                            end_date=END_DATE)

# with open('speedy_load_temp.dill', 'rb') as f:
    # txy = dill.load(f)
    # data = dill.load(f)

data, unsnapped_idx = models.NetworkSpaceTimeData.from_cartesian(net, raw_txy, return_failure_idx=True)
snapped_idx = sorted(set(range(raw_txy.shape[0])) - set(unsnapped_idx))
txy = data.time.adddim(data.space.to_cartesian(), type=models.CartesianSpaceTimeData)


class STDBScan(object):
    __name__ = "STDBScan"

    def __init__(self,
                 data,
                 dd_max,
                 dt_max,
                 min_pts=None):

        # euclidean
        # assert len(data.shape) == 2, "data must be a 2D matrix"

        # euclidean
        # assert data.shape[1] == 3, "data must be 3D"

        # network
        assert data.nd == 2, "data must be 2D (time + net space)"

        self.data = data

        # euclidean
        # self.ndata = data.shape[0]

        # network
        self.ndata = data.ndata

        self.dd_max = dd_max
        self.dt_max = dt_max
        self.min_pts = min_pts or self.initial_estimate_in_pts()
        threshold_fun = linkage_func_separable(dt_max, dd_max)

        # euclidean
        # self.ii, self.jj = linkages(models.CartesianSpaceTimeData(self.data),
        #                             threshold_fun=threshold_fun)
        # network
        self.ii, self.jj, self.dt, self.dd = network_linkages(self.data, threshold_fun)

        self.neighbour_lookup = {}
        for i, j in zip(self.ii, self.jj):
            self.neighbour_lookup.setdefault(i, set())
            self.neighbour_lookup.setdefault(j, set())
            self.neighbour_lookup[i].add(j)
            self.neighbour_lookup[j].add(i)
        self.unclassified = set(range(self.ndata)) - set(self.neighbour_lookup.keys())
        self.remaining = set(self.neighbour_lookup.keys())

        self.logger = logging.getLogger(self.__name__)
        self.logger.handlers = []
        self.logger.addHandler(logging.FileHandler('stdbscan.log', mode='w'))
        self.logger.setLevel(logging.DEBUG)

        self.labels_lookup = self.labels = None


    def initial_estimate_in_pts(self):
        """
        Follow algorithm proposed in the paper
        :return:
        """
        raise NotImplementedError()

    def cluster_population(self):
        if self.labels_lookup is None:
            raise AttributeError("Call run() first")
        return [(lab, len(arr)) for lab, arr in self.labels_lookup.iteritems()]

    def retrieve_neighbours(self, idx):
        """
        Find all the data that are within the neighbourhood of the datum with supplied id.
        Search only the data labelled as 'unclassified' or 'noise'; others have already been assigned.
        :param idx:
        :return:
        """
        return self.neighbour_lookup[idx] & self.remaining

    def run(self):
        cluster_idx = 0
        unclassified = set(range(self.ndata)) - set(self.neighbour_lookup.keys())

        labels = {}
        for i in unclassified:
            labels[i] = LABEL_UNCLASSIFIED
        self.remaining = set(self.neighbour_lookup.keys())

        # ensure that we are iterating over a copy here
        for i in sorted(list(self.remaining)):
            if i not in self.remaining:
                # this datum has been labelled already
                assert i in labels
                self.logger.debug("Point %d is already in cluster %d", i, labels[i])
                continue

            neighbours = self.retrieve_neighbours(i)
            if len(neighbours) < self.min_pts:
                self.logger.debug("Point %d is NOT a core point, labelling as noise for now", i)
                # could relabel this later if it is density connected to another cluster, so leave in remaining
                labels[i] = LABEL_NOISE
            else:
                # new cluster
                self.logger.debug("Starting NEW cluster %d", cluster_idx)
                labels[i] = cluster_idx
                to_remove = {i}
                self.logger.info("Added point %d to cluster %d", i, cluster_idx)
                self.logger.debug("Added neighbours of point %d to stack (n=%d)", i, len(neighbours))

                stack = set(neighbours)
                while len(stack):
                    self.logger.debug("Stack has length %d", len(stack))
                    j = stack.pop()
                    labels[j] = cluster_idx
                    self.logger.info("Added point %d to cluster %d", j, cluster_idx)
                    to_remove.add(j)
                    this_neighbours = self.retrieve_neighbours(j)
                    if len(this_neighbours) < self.min_pts:
                        self.logger.debug("Point %d is a BORDER point", j)
                    else:
                        self.logger.debug("Point %d is a CORE point with %d neighbours", j, len(this_neighbours))
                        for y in this_neighbours:
                            self.logger.debug("Neigbour %d", y)
                            the_label = labels.get(y)
                            if the_label is None:
                                self.logger.debug("Added point %d to cluster %d and the stack", y, cluster_idx)
                                labels[y] = cluster_idx
                                to_remove.add(y)
                                stack.add(y)
                            elif the_label == LABEL_NOISE:
                                self.logger.debug(
                                    "Point %d was previously assigned to noise, now adding it to cluster %d",
                                    y,
                                    cluster_idx)
                                labels[y] = cluster_idx
                                to_remove.add(y)
                            else:
                                self.logger.debug("Point %d is already in cluster %d so we will ignore it", y, the_label)

                cluster_size = (np.array(labels.values()) == cluster_idx).sum()
                self.logger.debug("Finished populating cluster %d. It contains %d members", cluster_idx, cluster_size)
                self.remaining -= to_remove
                self.logger.debug("Now removing %d points from the pool, leaving %d points", len(to_remove), len(self.remaining))

                cluster_idx += 1

        self.labels = labels
        self.labels_lookup = {}

        for k, v in self.labels.items():
            if v == LABEL_UNCLASSIFIED:
                v = LABEL_NOISE
                self.labels[k] = v
            self.labels_lookup.setdefault(v, []).append(k)



dd_max = 125.
dt_max = 900
min_pt = 24

self = STDBScan(data, dd_max, dt_max, min_pts=min_pt)
self.run()

# plotting

from matplotlib import pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111)
net.plot_network(ax=ax)
colour_scale = np.linspace(0, 1, 10)

# noise_idx = self.labels_lookup[LABEL_NOISE] + self.labels_lookup[LABEL_UNCLASSIFIED]
# x = txy[noise_idx, 1]
# y = txy[noise_idx, 2]
# ax.scatter(x, y, c='k', marker='o', s=40, alpha=0.2, label='noise', edgecolor='none')
for i in range(min(len(colour_scale), max(self.labels_lookup.keys()) + 1)):
    this_idx = self.labels_lookup[i]
    c = cm.jet(colour_scale[np.mod(i, len(colour_scale))])
    x = txy[this_idx, 1]
    y = txy[this_idx, 2]
    ax.scatter(x, y, s=40, c=np.ones((x.size, 4)) * c, edgecolor='none', label=str(i), cmap='jet')

ax.set_aspect('equal')

# save to new CSV

out_fn = 'stay_points_labelled.csv'
out_fields = raw_fields + ['easting_post_snap', 'northing_post_snap', 'label',]
outset = set(unsnapped_idx)
idx = 0
missing = []
with open(out_fn, 'wb') as f:
    c = csv.DictWriter(f, out_fields)
    c.writeheader()
    for i, r in enumerate(raw_data):
        if i in outset:
            r['label'] = LABEL_UNCLASSIFIED
            r['easting_post_snap'] = None
            r['northing_post_snap'] = None
        else:
            r['label'] = self.labels[idx]
            r['easting_post_snap'] = txy[idx, 1]
            r['northing_post_snap'] = txy[idx, 2]
            idx += 1
        c.writerow(r)
