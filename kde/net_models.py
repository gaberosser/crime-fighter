__author__ = 'gabriel'
from network.streetnet import StreetNet, NetPoint
from network.utils import NetworkWalker
import numpy as np
from data.models import NetworkData, NetworkSpaceTimeData
from kernels import LinearKernel1D
import collections
from time import time
from matplotlib import pyplot as plt
import logging

def linear_kernel(x, h):
    return 2 * (h - x) / float(h ** 2)

def exponential_kernel(x, h):
    """ h is the MEAN (not the frequency) """
    return np.exp(-x / float(h)) / float(h)


class KdeBase(object):
    data_class = NetworkData

    def __init__(self, source_data, bandwidths, targets=None, cutoffs=None, verbose=False, **kwargs):
        self.data_checks(source_data)
        self.source = None
        self.set_source(source_data)

        self.bandwidths = None
        self.set_bandwidths(bandwidths)
        self.cutoffs = None
        self.set_cutoffs(cutoffs)

        self.targets = None
        if targets is not None:
            self.set_targets(targets)

        # logging
        self.logger = logging.getLogger(str(self.__class__))
        self.logger.handlers = []  # make sure logger has no handlers to begin with
        if verbose:
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(logging.StreamHandler())
        else:
            self.logger.addHandler(logging.NullHandler())

    def data_checks(self, data):
        pass

    @property
    def n_source(self):
        return self.source.ndata

    @property
    def ndim(self):
        return self.source.nd

    def set_source(self, source_data):
        # TODO: ensure that the source data are never copied here. If they are, the dict hash keys may change?
        self.source = self.data_class(source_data)

    def set_bandwidths(self, bandwidths):
        if bandwidths is None:
            raise NotImplementedError("No automatic bandwidth selection available")

        if self.ndim == 1 and not hasattr(bandwidths, '__iter__'):
            bandwidths = [bandwidths]

        if len(bandwidths) != self.ndim:
            raise AttributeError("Number of supplied bandwidths does not match the dimensionality of the data")

        self.bandwidths = bandwidths

    def set_cutoffs(self, cutoffs):
        if cutoffs is None:
            self.cutoffs = self.bandwidths

        else:

            if self.ndim == 1 and not hasattr(cutoffs, '__iter__'):
                cutoffs = [cutoffs]

            if len(cutoffs) != self.ndim:
                raise AttributeError("Number of supplied cutoffs does not match the dimensionality of the data")

            self.cutoffs = cutoffs

    def set_targets(self, targets):
        self.data_checks(targets)
        self.targets = self.data_class(targets)

    def pdf(self, targets=None, **kwargs):
        raise NotImplementedError



class NetworkKdeBase(KdeBase):
    data_class = NetworkData

    def __init__(self, source_data,
                 bandwidths,
                 targets=None,
                 cutoffs=None,
                 max_split=1e4,
                 **kwargs):
        super(NetworkKdeBase, self).__init__(source_data, bandwidths, targets=targets, cutoffs=cutoffs, **kwargs)
        self.graph_degrees = self.graph.g.degree()
        self.max_split = max_split
        self.walker = None
        self.all_dists = []
        self.all_norms = []
        self.dists = collections.defaultdict(dict)
        self.norm = collections.defaultdict(dict)
        self.dists_r = collections.defaultdict(dict)
        self.norm_r = collections.defaultdict(dict)

    @staticmethod
    def kernel(*args, **kwargs):
        return linear_kernel(*args, **kwargs)

    def data_checks(self, data):
        if data.nd != 1:
            raise AttributeError("Supplied data must be 1D")

    @property
    def graph(self):
        return self.source.graph

    def reset(self):
        self.all_dists = []
        self.all_norms = []
        self.dists = collections.defaultdict(dict)
        self.norm = collections.defaultdict(dict)
        self.dists_r = collections.defaultdict(dict)
        self.norm_r = collections.defaultdict(dict)

    def set_source(self, source_data):
        # TODO: ensure that the source data are never copied here. If they are, the dict hash keys may change?
        self.source = self.data_class(source_data)

    def set_targets(self, targets):
        super(NetworkKdeBase, self).set_targets(targets)
        # set net walker
        self.walker = NetworkWalker(self.graph, targets=self.targets, max_distance=self.cutoffs[0], max_split=self.max_split)
        # reset cache
        self.reset()

    def add_paths(self):
        """ Iterate over sources, adding network paths and distances for all source-target journeys """
        # delete all previous results
        # even if we need some of these data again, they are quick to recompute because the walker caches its journeys
        self.reset()

        for i in range(self.n_source):
            s = self.source.getone(i)
            tic = time()

            source_targets = self.walker.source_to_targets(s)
            if not len(source_targets):
                return

            # ensure dictionary values are accessible
            _ = self.dists[s], self.norm[s]

            for j, p in source_targets.iteritems():
                t = self.targets.getone(j)
                # ensure dictionary values are accessible
                _ = self.norm_r[t], self.dists_r[t]

                i0 = len(self.all_dists)
                i1 = i0 + len(p)
                for u in p:  # for all paths between s and t...
                    self.all_dists.append(u[1])
                    this_norm = 1.
                    for v in u[0]:  # iterate over nodes in this path
                        if self.graph_degrees[v] > 1:
                            this_norm *= (self.graph_degrees[v] - 1)
                    self.all_norms.append(this_norm)

                self.dists_r[t][s] = self.dists[s][t] = self.all_dists[i0:i1]
                self.norm[s][t] = self.norm_r[t][s] = self.all_norms[i0:i1]

            self.logger.info("Computed paths for source %d in time %.3f s", i, time() - tic)

    def pdf(self, targets=None, **kwargs):

        if targets is not None:
            self.set_targets(targets)

        self.add_paths()

        res = np.zeros(self.targets.ndata)
        for i, t in enumerate(self.targets.toarray(0)):
            this_res = 0.
            for s in self.dists_r[t].keys():
                d = np.array(self.dists_r[t][s])
                n = np.array(self.norm_r[t][s])
                this_res += sum(self.kernel(d, self.bandwidths[0]) / n)
            res[i] = this_res
        return res


class NetworkTimeKde(KdeBase):
    data_class = NetworkSpaceTimeData
    net_kde_class = NetworkKdeBase

    def __init__(self, source_data, bandwidths, **kwargs):
        super(NetworkTimeKde, self).__init__(source_data, bandwidths, **kwargs)
        # set spatial KDE object
        self.net_kde = self.net_kde_class(source_data.space,
                                          self.bandwidths[1],
                                          targets=self.targets,
                                          cutoffs=self.cutoffs[1],
                                          **kwargs)

    @staticmethod
    def time_kernel(x, h, *args, **kwargs):
        return exponential_kernel(x, h)

    def data_checks(self, data):
        if data.nd != 2:
            raise AttributeError("Supplied data must be 2D")

    def pdf(self, targets=None, **kwargs):
        if targets is not None:
            self.set_targets(targets)

        t_target = self.targets.time.toarray(0)
        t_source = self.source.time.toarray(0)
        t_target, t_source = np.meshgrid(t_target, t_source)
        dt = (t_target - t_source)
        zt = self.time_kernel(dt, self.bandwidths[0])
        import ipdb; ipdb.set_trace()
        zt[(dt <= 0) | (dt > self.cutoffs[0])] = 0.
        zt = zt.sum(axis=0)

        zd = self.net_kde.pdf(self.targets.space)

        return zt * zd


if __name__ == "__main__":
    from network.tests import load_test_network
    from network import utils
    from data.models import CartesianData, DataArray
    from matplotlib import pyplot as plt

    itn_net = load_test_network()
    nodes = np.array([t['loc'] for t in itn_net.g.node.values()])
    xmin, ymin, xmax, ymax = itn_net.extent
    targets, n_per_edge = utils.network_walker_uniform_sample_points(itn_net, 10)

    # lay down some random points within that box
    num_pts = 100

    x_pts = np.random.rand(num_pts) * (xmax - xmin) + xmin
    y_pts = np.random.rand(num_pts) * (ymax - ymin) + ymin
    xy = CartesianData.from_args(x_pts, y_pts)
    sources = NetworkData.from_cartesian(itn_net, xy, grid_size=50)  # grid_size defaults to 50

    # xy = CartesianData.from_args(np.array([530980]), np.array([174840]))
    # sources = NetworkData.from_cartesian(itn_net, xy)

    kk = NetworkKdeBase(source_data=sources,
                        bandwidths=200,
                        cutoffs=200,
                        verbose=True)
    z = kk.pdf(targets=targets)
    zn = z/max(z)

    plt.figure()
    itn_net.plot_network()
    plt.scatter(nodes[:, 0], nodes[:, 1], marker='x', s=15, c='k')
    plt.scatter(*targets.to_cartesian().separate, s=(zn * 20) ** 2)
    plt.scatter(*sources.to_cartesian().separate, c='r', s=50)

    # add time to sources and targets
    times = DataArray(np.random.rand(num_pts))
    sources_st = times.adddim(sources, type=NetworkSpaceTimeData)
    targets_st = DataArray(np.ones(targets.ndata)).adddim(targets, type=NetworkSpaceTimeData)

    kst = NetworkTimeKde(source_data=sources_st,
                         bandwidths=[0.2, 200],
                         verbose=True)
    zst = kst.pdf(targets_st)
    zst_n = zst / max(zst)

    plt.figure()
    itn_net.plot_network()
    plt.scatter(nodes[:, 0], nodes[:, 1], marker='x', s=15, c='k')
    plt.scatter(*targets.to_cartesian().separate, s=(zst_n * 20) ** 2)
    plt.scatter(*sources.to_cartesian().separate, c=sources_st.time.toarray(0), s=50)