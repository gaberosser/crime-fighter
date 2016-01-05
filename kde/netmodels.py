__author__ = 'gabriel'
from models import KernelCluster, KdeBase
from data.models import NetworkSpaceTimeData
from network.utils import NetworkWalker
from kernels import NetworkTemporalKernelEqualSplit
import operator


class NetworkTemporalKde(KernelCluster):
    data_class = NetworkSpaceTimeData

    def __init__(self, source_data, bandwidths,
                 ktype=NetworkTemporalKernelEqualSplit,
                 targets=None,
                 cutoff_tol=1e-4,
                 max_net_split=1e4,
                 **kwargs):
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

    def compute_cutoffs_from_bandwidth(self, bandwidths, tol=1e-4):
        """
        Use this routine to compute the cutoff values automatically from the bandwidths.
        In the case of the spatial component, this is a hard cutoff equal to the bandwidth
        In the case of the temporal component, this depends on the tolerance
        :param bandwidths: Bandwidths, iterable
        :param tol: Specifies the numerical tolerance for cutoff in the time domain
        :return: Cutoffs, iterable, same length as bandwidths
        """
        return self.ktype.compute_cutoffs(bandwidths, tol=tol)

    def set_bandwidths_and_cutoffs(self, bandwidths):
        """
        Verify that the supplied bandwidths are valid. Compute the cutoffs automatically using the tolerance already
        supplied.
        This makes use of the class method compute_bandwidths that should be available from all kernel classes.
        """
        if not hasattr(bandwidths, '__iter__'):
            bandwidths = [bandwidths] * self.ndim

        if len(bandwidths) != self.ndim:
            raise AttributeError("Number of supplied bandwidths does not match the dimensionality of the data")

        self.bandwidths = bandwidths
        self.cutoffs = self.ktype.compute_cutoffs(bandwidths, tol=self.cutoff_tol)
        # if kernels have already been set, update them
        if len(self.kernels):
            for k in self.kernels:
                k.update_bandwidths(self.bandwidths, time_cutoff=self.cutoffs[0])

    def set_targets(self, targets):
        self.targets = self.data_class(targets, copy=False)
        self.walker = NetworkWalker(self.data.graph,
                                    targets=self.targets.space,
                                    max_distance=self.cutoffs[1],
                                    max_split=self.max_net_split)
        [k.set_walker(self.walker) for k in self.kernels]

    @property
    def targets_set(self):
        return self.targets is not None

    def create_kernels(self):
        return [self.ktype(self.data[i], self.bandwidths, time_cutoff=self.cutoffs[0]) for i in range(self.ndata)]

    def update_source_data(self, new_source_data, new_bandwidths=None):
        new_source_data = self.data_class(new_source_data, copy=False)
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
            if target_times.ndata != self.targets.ndata:
                raise AttributeError("The number of data points does not match existing data in the supplied array")
            self.targets.time = target_times
        else:
            self.targets.data[:, 0] = t

    def pdf(self, targets=None, **kwargs):
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