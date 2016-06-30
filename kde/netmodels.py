__author__ = 'gabriel'
from models import KernelCluster, KdeBase
from data.models import NetworkSpaceTimeData
from network.walker import NetworkWalker
from kernels import NetworkTemporalKernelEqualSplit
import operator
import logging

logger_kde = logging.getLogger('NetworkTemporalKde')
fh = logging.FileHandler('netmodels.NetworkTemporalKde.log', mode='w')
fmt = logging.Formatter(fmt='%(levelname)s - %(name)s %(asctime)s %(module)s %(funcName)s [%(process)d %(thread)d] %(message)s')
fh.setFormatter(fmt)
logger_kde.addHandler(fh)

# fh_nw = logging.FileHandler('NetworkWalker.log', mode='w')
# fh_nw.setFormatter(fmt)
logger_nw = logging.getLogger('NetworkWalker')
# logger_nw.addHandler(fh_nw)
logger_nw.addHandler(logging.NullHandler())

class NetworkTemporalKde(KernelCluster):
    data_class = NetworkSpaceTimeData

    def __init__(self, source_data, bandwidths,
                 ktype=NetworkTemporalKernelEqualSplit,
                 targets=None,
                 cutoff_tol=1e-4,
                 max_net_split=1e4,
                 **kwargs):
        self.verbose = kwargs.get('verbose', False)
        self.logger = logger_kde
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.info("NetworkTemporalKde __init__")
        source_data = self.data_class(source_data)
        self.max_net_split = max_net_split
        # TODO: can't call parent class __init__ because it assumes data are in an np array. Switch to using data
        # objects for all classes?
        # super(NetworkTimeKernelCluster, self).__init__(source_data, bandwidths, ktype)
        self.ktype = ktype
        self.kernels = []
        self.data = source_data
        self.bandwidths = None
        self.cutoff_tol = cutoff_tol
        self.cutoffs = None
        self.set_bandwidths_and_cutoffs(bandwidths)
        self.kernels = self.create_kernels()

        self.targets = None
        self.walker = None
        if targets is not None:
            self.set_targets(targets)

    @property
    def ndim(self):
        return self.data.nd

    @property
    def ndata(self):
        return self.data.ndata

    @property
    def norm_constant(self):
        return float(self.ndata)

    def set_bandwidths_and_cutoffs(self, bandwidths):
        """
        Verify that the supplied bandwidths are valid. Compute the cutoffs automatically using the tolerance already
        supplied.
        This makes use of the class method compute_bandwidths that should be available from all kernel classes.
        """
        self.logger.info("set_bandwidths_and_cutoffs with bandwidths=%s", str(bandwidths))
        if not hasattr(bandwidths, '__iter__'):
            bandwidths = [bandwidths] * self.ndim

        if len(bandwidths) != self.ndim:
            raise AttributeError("Number of supplied bandwidths does not match the dimensionality of the data")

        self.bandwidths = bandwidths
        # compute cutoffs directly from the kernel class method
        # if tol is None, only non-arbitrary cutoffs are applied
        self.cutoffs = self.ktype.compute_cutoffs(bandwidths, tol=self.cutoff_tol)
        # if kernels have already been set, update them
        if len(self.kernels):
            for k in self.kernels:
                k.update_bandwidths(self.bandwidths, time_cutoff=self.cutoffs[0])

    def set_targets(self, targets):
        self.logger.info("set_targets")
        self.targets = self.data_class(targets, copy=False)
        self.walker = NetworkWalker(self.data.graph,
                                    targets=self.targets.space,
                                    max_distance=self.cutoffs[1],
                                    max_split=self.max_net_split,
                                    verbose=self.verbose,
                                    logger=logger_nw)
        [k.set_walker(self.walker) for k in self.kernels]

    @property
    def targets_set(self):
        return self.targets is not None

    def create_kernels(self):
        self.logger.info("create_kernels creating %d kernels", self.ndata)
        return [self.ktype(self.data[i], self.bandwidths, time_cutoff=self.cutoffs[0]) for i in range(self.ndata)]

    def update_source_data(self, new_source_data, new_bandwidths=None):
        new_source_data = self.data_class(new_source_data, copy=False)
        self.logger.info("update_source_data with %d sources", new_source_data.ndata)
        if new_bandwidths is None:
            new_bandwidths = self.bandwidths
        self.data = new_source_data
        self.bandwidths = new_bandwidths
        self.kernels = self.create_kernels()
        # set the network walker on the new kernels as they can still use the cache
        [k.set_walker(self.walker) for k in self.kernels]

    def iter_operate(self, funcstr, targets=None, **kwargs):
        """
        Return an iterator that executes the fun named in funcstr to each kernel in turn.
        Unlike the base case, the targets data array is optional: targets are retained between calls for
        caching purposes.
        """
        if targets is not None:
            # reset the network walker
            self.set_targets(targets)
            self.logger.info("iter_operate with new targets set")
        else:
            self.logger.info("iter_operate with no new targets")
        # need to send the target times to the kernel, since net walker only stores the spatial (net) targets
        target_times = self.targets.time
        return (getattr(x, funcstr)(target_times=target_times, **kwargs) for x in self.kernels)

    def additive_operation(self, funcstr, targets=None, **kwargs):
        """ Generic interface to call function named in funcstr on the data, reducing data by summing """
        return reduce(operator.add, self.iter_operate(funcstr, targets=targets, **kwargs))

    def operation(self, funcstr, targets=None, **kwargs):
        return list(self.iter_operate(funcstr, targets=targets, **kwargs))

    def update_target_times(self, target_times):
        """
        Update the times attached to the (spatial) targets.
        :param target_times: Either an iterable (in which case it must be possible to cast to a DataArray) or a scalar
        (in which case all target times are set to this value)
        """
        if hasattr(target_times, '__iter__'):
            target_times = DataArray(target_times, copy=False)
            self.logger.info("update_target_times with an iterable of len %d", target_times.ndata)
            if target_times.ndata != self.targets.ndata:
                raise AttributeError("The number of data points does not match existing data in the supplied array")
            self.targets.time = target_times
        else:
            self.logger.info("update_target_times with a fixed time %f", target_times)
            self.targets.data[:, 0] = target_times

    def pdf(self, targets=None, **kwargs):
        self.logger.info("pdf")
        normed = kwargs.pop('normed', True)
        z = self.additive_operation('pdf', targets=targets)

        if normed:
            z /= self.norm_constant
        return z


if __name__ == '__main__':
    from kernels import NetworkKernelEqualSplitLinear, NetworkTemporalKernelEqualSplit
    from network.tests import load_test_network
    from network import utils
    import numpy as np
    from data.models import CartesianData, DataArray, NetworkData
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
    # not all points necessarily snap successfully
    num_pts = sources.ndata

    radius = 200.

    nw = utils.NetworkWalker(itn_net, targets, max_distance=radius, max_split=1e4)
    k = NetworkKernelEqualSplitLinear(sources.getone(0), 200.)
    k.set_walker(nw)
    z = k.pdf()
    zn = z / max(z)
    # plt.figure()
    # itn_net.plot_network()
    # plt.scatter(nodes[:, 0], nodes[:, 1], marker='x', s=15, c='k')
    # plt.scatter(*targets.to_cartesian().separate, s=(zn * 20) ** 2)
    # plt.scatter(*sources.getone(0).cartesian_coords, c='r', s=50)

    # add time to sources and targets
    times = DataArray(np.random.rand(num_pts))
    sources_st = times.adddim(sources, type=NetworkSpaceTimeData)
    targets_st = DataArray(np.ones(targets.ndata)).adddim(targets, type=NetworkSpaceTimeData)

    kst = NetworkTemporalKernelEqualSplit([sources_st.time.getone(0), sources_st.space.getone(0)], [0.5, 200.])
    kst.set_walker(nw)
    zst = kst.pdf(target_times=targets_st.time)
    zstn = zst / z.max()

    # plt.figure()
    # itn_net.plot_network()
    # plt.scatter(nodes[:, 0], nodes[:, 1], marker='x', s=15, c='k')
    # plt.scatter(*targets.to_cartesian().separate, s=(zstn * 20) ** 2)
    # plt.scatter(*sources.getone(0).cartesian_coords, c=sources_st.time.getone(0), s=50, vmax=1., vmin=0.)


    this_sources_st = sources_st.getrows(range(50))
    kk = NetworkTemporalKde(this_sources_st, bandwidths=[0.5, 200.], targets=targets_st)
    y = kk.pdf()
    yn = y / max(y)

    plt.figure()
    itn_net.plot_network()
    plt.scatter(nodes[:, 0], nodes[:, 1], marker='x', s=15, c='k')
    plt.scatter(*targets.to_cartesian().separate, s=(yn * 20) ** 2)
    plt.scatter(*this_sources_st.space.to_cartesian().separate,
                c=this_sources_st.time.toarray(),
                s=50, vmax=1., vmin=0.5)

    this_sources_st = sources_st.getrows(range(50, num_pts))
    kk.update_source_data(this_sources_st)
    y = kk.pdf()
    yn = y / max(y)
    plt.figure()
    itn_net.plot_network()
    plt.scatter(nodes[:, 0], nodes[:, 1], marker='x', s=15, c='k')
    plt.scatter(*targets.to_cartesian().separate, s=(yn * 20) ** 2)
    plt.scatter(*this_sources_st.space.to_cartesian().separate,
                c=this_sources_st.time.toarray(),
                s=50, vmax=1., vmin=0.5)