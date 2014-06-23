__author__ = 'gabriel'
import numpy as np


class Hotspot(object):

    def __init__(self, data, stkernel):
        self.stkernel = stkernel
        self.stkernel.train(data)

    @property
    def ndata(self):
        return self.data.shape[0]

    def train(self, data):
        self.stkernel.train(data)

    def predict(self, t, x, y):
        return self.stkernel.predict(t, x, y)


class STKernelBowers(object):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def _evaluate(self, t, x, y):
        td = self.data[:, 0]
        xd = self.data[:, 1]
        yd = self.data[:, 2]
        return sum(1.0 / ((1 + self.a * np.abs(t - td)) * (1 + self.b * np.sqrt((x - xd)**2 + (y - yd)**2))))

    def train(self, data):
        self.data = data

    def predict(self, t, x, y):
        it = np.nditer([t, x, y] + [None,])
        for i in it:
            i[-1][...] = self._evaluate(i[0], i[1], i[2])
        return it.operands[-1]
