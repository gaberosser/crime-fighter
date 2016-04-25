import numpy as np
from data import models
from analysis import chicago
from network import osm
from settings import DATA_DIR
import os
import datetime
import dill
from network.utils import linkages, network_linkages, linkage_func_separable
import logging


LABEL_UNCLASSIFIED = -2
LABEL_NOISE = -1


# START_DATE = datetime.date(2010, 1, 1)
# END_DATE = datetime.date(2014, 1, 1)
#
net_file = os.path.join(DATA_DIR, 'chicago', 'network', 'chicago_south_clipped.net')
net = osm.OSMStreetNet.from_pickle(net_file)
# sides = chicago.get_chicago_side_polys(True)
# data, t0, cid = chicago.get_crimes_by_type(domain=sides['South'],
#                                            start_date=START_DATE,
#                                            end_date=END_DATE)

with open('speedy_load_temp.dill', 'rb') as f:
    # txy = dill.load(f)
    data = dill.load(f)

data = models.NetworkSpaceTimeData.from_cartesian(net, data)
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
                to_remove = {i}
                self.logger.debug("Adding all %d neighbours of point %d to stack", len(neighbours), i)

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

        for k, v in self.labels.iteritems():
            self.labels_lookup.setdefault(v, []).append(k)



dd_max = 100.
dt_max = 21.
self = STDBScan(data, dd_max, dt_max, min_pts=5)
self.run()


from matplotlib import pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111)
# net.plot_network(ax=ax)
colour_scale = np.linspace(0, 1, 5)

noise_idx = self.labels_lookup[LABEL_NOISE] + self.labels_lookup[LABEL_UNCLASSIFIED]
x = txy[noise_idx, 1]
y = txy[noise_idx, 2]
ax.scatter(x, y, c='k', marker='o', s=40, alpha=0.2, label='noise', edgecolor='none')
for i in range(min(5, max(self.labels_lookup.keys()) + 1)):
    this_idx = self.labels_lookup[i]
    c = cm.jet(colour_scale[np.mod(i, len(colour_scale))])
    x = txy[this_idx, 1]
    y = txy[this_idx, 2]
    ax.scatter(x, y, s=40, c=np.ones((x.size, 4)) * c, edgecolor='none', label=str(i), cmap='jet')

ax.set_aspect('equal')