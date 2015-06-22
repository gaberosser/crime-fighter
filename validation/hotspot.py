__author__ = 'gabriel'
import numpy as np
from scipy.stats import gaussian_kde
from scipy import sparse
from kde.models import FixedBandwidthKdeScott, VariableBandwidthNnKde
from kde.netmodels import NetworkTemporalKde
from data.models import DataArray, CartesianSpaceTimeData, SpaceTimeDataArray, NetworkData, NetworkSpaceTimeData
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


class STKernelBase(object):

    data_class = DataArray

    def __init__(self):
        self.data = None

    @property
    def ndata(self):
        if self.data is None:
            return 0
        return self.data.ndata

    def train(self, data):
        self.data = self.data_class(data)

    def predict(self, data_array):
        raise NotImplementedError()


class STKernelBowers(STKernelBase):
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
            # aa = 1. / (dt * self.a + 1.)
            # bb = 1. / (dd * self.b + 1.)
            # return aa * bb >= self.min_frac
        link_i, link_j = linkages(self.data, linkage_fun, data_target=target_data)

        dt = 1 / ((target_data.time.getrows(link_j) - self.data.time.getrows(link_i)) * self.a + 1.).toarray(0)
        dd = 1 / (target_data.space.getrows(link_j).distance(self.data.space.getrows(link_i)) * self.b + 1.).toarray(0)
        return link_i, link_j, dt, dd

    def predict(self, data_array):
        ## FIXME: the function used here is basically WRONG
        #  The original paper advocates only including crimes within a 400m radius then summing the kernel:
        #  Sum(1/(delta_dist) * 1/(delta_time))
        #  In practice, going to use (1+delta_dist) in the denominator to avoid the singularity!
        #  Though this should be mitigated by the remove_coincident_points problem, but that won't sort out delta_t=0
        data_array = self.data_class(data_array)

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


class SKernelBase(STKernelBase):
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

    # def predict(self, data_array):
    #     data_array = DataArray(data_array)
    #     assert data_array.nd == 2, 'Predict requires a 2D DataArray'
    #     return self.kde.pdf(data_array)

    def predict(self, data_array):
        data_array = SpaceTimeDataArray(data_array)
        return self.kde.pdf(data_array.space)

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

    def train(self, data):
        self.set_data(SpaceTimeDataArray(data))
        self.set_prediction_method()

    def set_prediction_method(self):
        pass


class STNetworkKernelBase(STKernelBase):
    data_class = NetworkSpaceTimeData

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

    def predict(self, data_array):
        data_array = self.data_class(data_array)
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

    @property
    def spatial_upper_limit(self):
        return self.radius

    @property
    def temporal_upper_limit(self):
        return self.time_decay * 7.

    def reset_kde(self):
        """ Redefines the KDE, resetting any cached results """
        self.kde = NetworkTemporalKde(self.data,
                                      cutoffs=[self.temporal_upper_limit, self.spatial_upper_limit],
                                      bandwidths=[self.time_decay, self.radius])

    def train(self, data):
        super(STNetworkLinearSpaceExponentialTime, self).train(data)
        # If the KDE has not previously been set, define now...
        if self.kde is None:
            self.reset_kde()
        else:
            # Otherwise just update the existing KDE sources
            self.kde.update_source_data(self.data, new_bandwidths=[self.time_decay, self.radius])

    def predict(self, time, spatial_data_array):
        """ Generate a prediction from the trained model.
        """
        if self.kde.targets_set:
            return self.kde.pdf(time)
        else:
            data_array = self.data_class(spatial_data_array)
            return self.kde.pdf(time, net_targets=spatial_data_array)
