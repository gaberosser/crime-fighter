__author__ = 'gabriel'
import numpy as np
from scipy.stats import gaussian_kde
from scipy import sparse
from kde.kernels import LinearKernel1D
from kde.models import FixedBandwidthKdeScott, VariableBandwidthNnKde, FixedBandwidthLinearSpaceExponentialTimeKde
from kde.netmodels import NetworkTemporalKde
from data.models import DataArray, CartesianSpaceTimeData, SpaceTimeDataArray, NetworkData, NetworkSpaceTimeData, \
    CartesianData
from point_process.utils import linkages, linkage_func_separable
from network.utils import network_linkages

        # x = self.roc.sample_points.toarray(0)
        # y = self.roc.sample_points.toarray(1)
        # ts = np.ones_like(x) * t
        # data_array = self.data_class.from_args(ts, x, y)
        # data_array.original_shape = x.shape
        # return data_array

def generate_st_prediction_dataarray(time, space_arr, dtype=SpaceTimeDataArray):
    """
    Combine a DataArray of spatial sampling points and a scalar time into a ST DataArray by duplicating the time
     and concatenating.
    :param time:
    :param space_arr:
    :return:
    """
    try:
        original_shape = space_arr.original_shape
    except AttributeError:
        original_shape = None
    s_arrs = space_arr.separate
    t_arr = np.ones_like(s_arrs[0]) * time
    res = dtype.from_args(t_arr, *s_arrs)
    res.original_shape = original_shape
    return res


class STBase(object):

    data_class = SpaceTimeDataArray
    space_class = CartesianData

    def __init__(self):
        self.data = None

    def set_data(self, data):
        self.data = self.data_class(data)

    @property
    def ndata(self):
        if self.data is None:
            return 0
        return self.data.ndata

    def train(self, data):
        self.set_data(data)

    def prediction_array(self, time, space_array):
        # combine time with space dimension
        space_array = self.space_class(space_array, copy=False)
        t = DataArray(time * np.ones(space_array.ndata))
        if space_array.original_shape is not None:
            t.original_shape = space_array.original_shape
        return t.adddim(space_array, type=self.data_class)

    def predict(self, time, space_array):
        raise NotImplementedError()


class STKDEBase(STBase):
    kde_class = None

    def __init__(self, **kde_kwargs):
        super(STKDEBase, self).__init__()
        self.kde = None
        self.kde_kwargs = kde_kwargs

    def set_kde(self):
        self.kde = self.kde_class(self.data, **self.kde_kwargs)

    def train(self, data):
        super(STKDEBase, self).train(data)
        self.set_kde()

    def predict(self, time, space_array):
        target_data = self.prediction_array(time, space_array)
        return self.kde.pdf(target_data)


class STGaussianNn(STBase):
    data_class = CartesianSpaceTimeData
    kde_class = VariableBandwidthNnKde


class STLinearSpaceExponentialTime(STKDEBase):

    data_class = CartesianSpaceTimeData
    kde_class = FixedBandwidthLinearSpaceExponentialTimeKde

    def __init__(self, radius, mean_time):
        # TODO: have disabled tol cutoff for now because it was inconsistent with the KDE framework.
        # Add support for this? But only worth doing with a proper filtering mechanism (e.g. see below)
        kde_kwargs = {'bandwidths': (float(mean_time), float(radius))}
        super(STLinearSpaceExponentialTime, self).__init__(**kde_kwargs)

    # TODO: consider whether this kind of approach might help us in the fixed bandwidth KDE situation more generally
    # I have disabled it for now because it essentially repeats all the same calls as KDE.

    # def get_linkages(self, target_data):
    #     def linkage_fun(dt, dd):
    #         return dd <= self.radius
    #
    #     link_i, link_j = linkages(self.data, linkage_fun, data_target=target_data)
    #     return link_i, link_j

    # def predict(self, time, space_array):
    #     data_array = self.prediction_array(time, space_array)
    #     link_i, link_j = self.get_linkages(data_array)
    #     if not len(link_i):
    #         return np.zeros(space_array.ndata)
    #     dt = (data_array.time.getrows(link_j) - self.data.time.getrows(link_i))
    #     dt = dt.toarray()
    #     dd = data_array.space.getrows(link_j).distance(self.data.space.getrows(link_i))
    #     dd = dd.toarray()
    #     a = np.exp(-dt / self.mean_time) / self.mean_time
    #     b = (self.radius - dd) / self.radius ** 2
    #     # import ipdb; ipdb.set_trace()
    #     m = sparse.lil_matrix((self.data.ndata, data_array.ndata))
    #
    #     m[link_i, link_j] = a * b
    #
    #     res = np.array(m.sum(axis=0).flat)
    #
    #     # reshape if necessary
    #     if data_array.original_shape is not None:
    #         res = res.reshape(data_array.original_shape)
    #     return res


class STKernelBowers(STBase):
    data_class = CartesianSpaceTimeData

    def __init__(self, a, b, min_frac=1e-6):
        """
        :param min_frac: The cutoff point at which to stop considering triggering effect. Defined as the fraction of
        the initial triggering value.
        """
        self.a = a
        self.b = b
        self.min_frac = min_frac
        super(STKernelBowers, self).__init__()

    def get_linkages(self, target_data):
        def linkage_fun(dt, dd):
            return dd <= 400  # units metres

        link_i, link_j = linkages(self.data, linkage_fun, data_target=target_data)

        dt = 1 / ((target_data.time.getrows(link_j) - self.data.time.getrows(link_i)) * self.a + 1.).toarray(0)
        dd = 1 / (target_data.space.getrows(link_j).distance(self.data.space.getrows(link_i)) * self.b + 1.).toarray(0)
        return link_i, link_j, dt, dd

    def predict(self, time, space_array):
        ## FIXME: the function used here is basically WRONG
        #  The original paper advocates only including crimes within a 400m radius then summing the kernel:
        #  Sum(1/(delta_dist) * 1/(delta_time))
        #  In practice, going to use (1+delta_dist) in the denominator to avoid the singularity!
        #  Though this should be mitigated by the remove_coincident_points problem, but that won't sort out delta_t=0
        data_array = self.prediction_array(time, space_array)
        link_i, link_j, dt, dd = self.get_linkages(data_array)

        c1 = 1 / (dt * self.a + 1.)
        c2 = 1 / (dd * self.b + 1.)
        # c1 = 1 / (dt + 1.)
        # c2 = 1 / (dd + 1.)

        if np.any(dt < 0):
            raise ValueError("Negative dt value encountered.")

        m = sparse.lil_matrix((self.data.ndata, data_array.ndata))

        m[link_i, link_j] = c1 * c2
        res = np.array(m.sum(axis=0).flat)

        # reshape if necessary
        if data_array.original_shape is not None:
            res = res.reshape(data_array.original_shape)
        return res


class SKernelBase(STBase):
    """
    Spatial kernel.  Aggregates data over time, with the time window defined in the input parameter dt.
    This is the base class, from which different KDE variants inherit.
    """

    def __init__(self, dt=None):
        self.dt = dt
        self.kde = None
        super(SKernelBase, self).__init__()

    def set_data(self, data):
        tf = max(data.time)
        if self.dt:
            self.data = data.space.getrows(data.time.toarray(0) >= (tf - self.dt))  # only spatial component
        else:
            self.data = data.space

    def set_kde(self):
        raise NotImplementedError

    def train(self, data):
        self.set_data(SpaceTimeDataArray(data))
        self.set_kde()

    def predict(self, time, space_array):
        space_array = self.data_class(space_array, copy=False)
        return self.kde.pdf(space_array)


class SKernelHistoric(SKernelBase):

    def __init__(self, dt=None, bdwidth=None):
        super(SKernelHistoric, self).__init__(dt=dt)
        self.bdwidth = bdwidth

    def set_kde(self):
        self.kde = FixedBandwidthKdeScott(self.data)


class SKernelHistoricVariableBandwidthNn(SKernelBase):

    def __init__(self, dt=None, nn=None):
        super(SKernelHistoricVariableBandwidthNn, self).__init__(dt=dt)
        self.nn = nn

    def set_kde(self):
        self.kde = VariableBandwidthNnKde(self.data, number_nn=self.nn)


class SNetworkKernelBase(SKernelBase):
    data_class = NetworkData

    def __init__(self, dt, max_distance):
        super(SNetworkKernelBase, self).__init__(dt)
        self.max_distance = max_distance

    def set_kde(self):
        # instantiate a space-only network KDE
        pass


class STNetworkKernelBase(STBase):
    data_class = NetworkSpaceTimeData
    space_class = NetworkData

    @property
    def spatial_bandwidth(self):
        raise NotImplementedError()


class STNetworkFixedRadius(STNetworkKernelBase):
    def __init__(self, radius, a=1):
        self.radius = radius
        self.t_decay = a
        super(STNetworkFixedRadius, self).__init__()

    def get_linkages(self, target_data):
        def linkage_fun(dt, dd):
            return dd <= self.radius
        return network_linkages(
            self.data,
            linkage_fun,
            data_target_net=target_data
        )

    @property
    def spatial_bandwidth(self):
        return self.radius

    def predict(self, time, space_array):
        data_array = self.prediction_array(time, space_array)
        link_i, link_j, dt, dd = self.get_linkages(data_array)

        # space_part = np.ones_like(dd)
        time_part = np.exp(-self.t_decay * dt)

        m = sparse.lil_matrix((self.data.ndata, data_array.ndata))

        m[link_i, link_j] = time_part
        res = np.array(m.sum(axis=0).flat)

        # reshape if necessary
        if data_array.original_shape is not None:
            res = res.reshape(data_array.original_shape)
        return res



class STNetworkBowers(STNetworkKernelBase, STKernelBowers):

    def get_linkages(self, target_data):
        def linkage_fun(dt, dd):
            return dd <= 400
            # aa = 1. / (dt * self.a + 1.)
            # bb = 1. / (dd * self.b + 1.)
            # return aa * bb >= self.min_frac
        return network_linkages(
            self.data,
            linkage_fun,
            data_target_net=target_data
        )

    @property
    def spatial_bandwidth(self):
    # FIXME: shouldn't hard-code this parameter
        return 400.


class STNetworkLinearSpaceExponentialTime(STNetworkKernelBase):
    """ Linear kernel in network space, vanishing at radius. Exponentially decaying time component """
    def __init__(self, radius, time_decay):
        self.radius = radius  # length unit
        self.time_decay = time_decay  # time unit
        self.kde = None
        super(STNetworkLinearSpaceExponentialTime, self).__init__()

    def reset_kde(self):
        """ Redefines the KDE, resetting any cached results """
        self.kde = NetworkTemporalKde(self.data,
                                      bandwidths=[self.time_decay, self.radius])

    def train(self, data):
        super(STNetworkLinearSpaceExponentialTime, self).train(data)
        # If the KDE has not previously been set, define now...
        if self.kde is None:
            self.reset_kde()
        else:
            # Otherwise just update the existing KDE sources
            self.kde.update_source_data(self.data, new_bandwidths=[self.time_decay, self.radius])

    def predict(self, time, space_array, force_update=False):
        """
        Generate a prediction from the trained model.
        Supply single time and spatial sample points in separate arrays
        If force_update=False then the spatial component is assumed unchanged, so only the time is altered.
        This is important: updating the spatial sample points loses all cached net paths.
        """
        if self.kde.targets_set and not force_update:
            self.kde.update_target_times(time)
            return self.kde.pdf()
        else:
            space_array = self.space_class(space_array, copy=False)
            time_array = DataArray(np.ones(space_array.ndata) * time)
            targets_array = time_array.adddim(space_array, type=self.data_class)
            return self.kde.pdf(targets=targets_array)
