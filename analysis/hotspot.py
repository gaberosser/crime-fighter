__author__ = 'gabriel'
import numpy as np
from scipy.stats import gaussian_kde
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
        e = 0.005
        max_delta_t = (1 - e) / (self.a * e)
        max_delta_d = (1 - e) / (self.b * e)

        link_i, link_j = linkages(self.data, max_delta_t, max_delta_d, data_target=data_array)


        # construct time and space difference matrices
        # FIXME: this will fail with large numbers of sources
        # easy fix - define cutoff based on a and b
        t2, t1 = np.meshgrid(data_array.time, self.data.time, copy=False)
        dt = t2 - t1
        x2, x1 = np.meshgrid(data_array.toarray(1), self.data.toarray(1), copy=False)
        dx = x2 - x1
        y2, y1 = np.meshgrid(data_array.toarray(2), self.data.toarray(2), copy=False)
        dy = y2 - y1
        dist = np.sqrt(dx ** 2 + dy ** 2)

        res =  sum(1.0 / (
            (1 + self.a * np.abs(dt)) *
            (1 + self.b * data_array.space.distance(self.data.space))
        )
        )


class SKernelHistoric(STKernelBase):

    def __init__(self, dt=None, bdwidth=None):
        self.dt = dt
        self.bdwidth = bdwidth
        self.kde = None
        super(SKernelHistoric, self).__init__()

    def set_data(self, data):
        tf = max(data.time)
        if self.dt:
            self.data = data.space[data.time >= (tf - self.dt)] # only spatial component
        else:
            self.data = data.space

    def set_kde(self):
        self.kde = FixedBandwidthKdeScott(self.data)
        # self.kde = gaussian_kde(self.data.data.transpose(), bw_method=self.bdwidth or 'scott')

    def train(self, data):
        self.set_data(SpaceTimeDataArray(data))
        self.set_kde()

    def predict(self, data_array):
        data_array = SpaceTimeDataArray(data_array)
        return self.kde.pdf(data_array.space)
        # return self.kde(data_array.data.transpose()).reshape(data_array.original_shape, order='F')


class SKernelHistoricVariableBandwidthNn(STKernelBase):

    def __init__(self, dt=None, nn=None):
        self.dt = dt
        self.nn = nn
        self.kde = None
        super(SKernelHistoricVariableBandwidthNn, self).__init__()

    def set_data(self, data):
        tf = max(data.time)
        if self.dt:
            self.data = data.space.getrows(np.where(data.time >= (tf - self.dt))[0])  # only spatial component
        else:
            self.data = data.space

    def set_kde(self):
        self.kde = VariableBandwidthNnKde(self.data, number_nn=self.nn)

    def train(self, data):
        self.set_data(SpaceTimeDataArray(data))
        self.set_kde()

    def predict(self, data_array):
        data_array = DataArray(data_array)
        assert data_array.nd == 2, 'Predict requires a 2D DataArray'
        return self.kde.pdf(data_array)