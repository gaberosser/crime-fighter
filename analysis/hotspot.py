__author__ = 'gabriel'
import numpy as np
from scipy.stats import gaussian_kde
from kde.models import VariableBandwidthNnKde

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

    def predict(self, t, x, y):
        return self.stkernel.predict(t, x, y)


class STKernelBase(object):

    def __init__(self):
        self.data = np.array([])

    @property
    def ndata(self):
        return self.data.shape[0]

    def train(self, data):
        self.data = data

    def _evaluate(self, t, x, y):
        raise NotImplementedError()

    def predict(self, t, x, y):
        it = np.nditer([t, x, y] + [None,])
        for i in it:
            i[-1][...] = self._evaluate(i[0], i[1], i[2])
        return it.operands[-1]


class STKernelBowers(STKernelBase):

    def __init__(self, a, b):
        self.a = a
        self.b = b
        super(STKernelBowers, self).__init__()

    def _evaluate(self, t, x, y):
        td = self.data[:, 0]
        xd = self.data[:, 1]
        yd = self.data[:, 2]
        return sum(1.0 / ((1 + self.a * np.abs(t - td)) * (1 + self.b * np.sqrt((x - xd)**2 + (y - yd)**2))))


class SKernelHistoric(STKernelBase):

    def __init__(self, dt=None, bdwidth=None):
        self.dt = dt
        self.bdwidth = bdwidth
        self.kde = None
        super(SKernelHistoric, self).__init__()

    def set_data(self, data):
        # assume last data point is most recent
        tf = data[-1, 0]
        if self.dt:
            self.data = data[data[:, 0] >= (tf - self.dt), 1:] # only spatial component
        else:
            self.data = data[:, 1:]

    def set_kde(self):
        self.kde = gaussian_kde(self.data.transpose(), bw_method=self.bdwidth or 'silverman')

    def train(self, data):
        self.set_data(data)
        self.set_kde()

    def _evaluate(self, t, x, y):
        if not isinstance(x, np.ndarray):
            return self.kde([x, y])
        shp = x.shape
        return self.kde([x.flatten(), y.flatten()]).reshape(shp)


class SKernelHistoricVariableBandwidthNn(STKernelBase):

    def __init__(self, dt=None, nn=None):
        self.dt = dt
        self.nn = nn
        self.kde = None
        super(SKernelHistoricVariableBandwidthNn, self).__init__()

    def set_data(self, data):
        # assume last data point is most recent
        tf = data[-1, 0]
        if self.dt:
            self.data = data[data[:, 0] >= (tf - self.dt), 1:] # only spatial component
        else:
            self.data = data[:, 1:]

    def set_kde(self):
        self.kde = VariableBandwidthNnKde(self.data, nn=self.nn)

    def train(self, data):
        self.set_data(data)
        self.set_kde()

    def _evaluate(self, t, x, y):
        return self.kde.pdf(x, y)

    def predict(self, t, x, y):
        return self._evaluate(t, x, y)