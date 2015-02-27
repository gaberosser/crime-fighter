__author__ = 'gabriel'
import numpy as np
from scipy.stats import gaussian_kde
from scipy import sparse
from kde.models import FixedBandwidthKdeScott, VariableBandwidthNnKde
from data.models import DataArray, CartesianSpaceTimeData, SpaceTimeDataArray
from point_process.utils import linkages


class Hotspot(object):

    def __init__(self, stkernel, data=None):
        self.stkernel = stkernel
        if data is not None:
            self.stkernel.train(data)

    @property
    def ndata(self):
        return self.stkernel.ndata

    def train(self, data, **kwargs):
        self.stkernel.train(data)

    def predict(self, data_array):
        """
        :param data_array: data array object that inherits from data.models.Data
        :return: prediction at the datapoints in data_array
        """
        return self.stkernel.predict(data_array)


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

    def __init__(self, a, b):
        self.a = a
        self.b = b
        super(STKernelBowers, self).__init__()

    def predict(self, data_array):
        data_array = self.data_class(data_array)

        # construct linkage indices
        # e = 0.005
        e = 1e-6
        max_delta_t = (1 - e) / (self.a * e)
        max_delta_d = (1 - e) / (self.b * e)

        link_i, link_j = linkages(self.data, max_delta_t, max_delta_d, data_target=data_array)

        m = sparse.lil_matrix((self.data.ndata, data_array.ndata))
        tt = 1 / ((data_array.time.getrows(link_j) - self.data.time.getrows(link_i)) * self.a + 1.).toarray(0)
        dd = 1 / (data_array.space.getrows(link_j).distance(self.data.space.getrows(link_i)) * self.b + 1.).toarray(0)

        m[link_i, link_j] = tt * dd
        m[tt < 0] = 0.

        return np.array(m.sum(axis=0).flat)


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
            self.data = data.space.getrows(data.time.toarray(0) >= (tf - self.dt)) # only spatial component
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
