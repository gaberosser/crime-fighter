__author__ = 'gabriel'
import numpy as np
from network.geos import NetworkPoint


class Data(object):

    @property
    def nd(self):
        raise NotImplementedError()

# class TimeData(object):
#
#     def __init__(self, data):
#         self.data = np.array(data)
#         if self.data.ndim != 1:
#             raise AttributeError("Time data must be one-dimensional")
#
#     @property
#     def ndata(self):
#         return self.data.size
#
#
# class SpaceData(object):
#
#     def __init__(self, *args):
#         if not len(args):
#             raise AttributeError("Must provide some initial data")
#         ndim = len(args)
#         ndata = len(args[0])
#         if ndim == 1:
#             self.data = np.array(args).reshape((ndata, 1))
#         else:
#             self.data = np.array(args).transpose()
#
#     @property
#     def ndata(self):
#         return self.data.shape[0]
#
#     @property
#     def ndim(self):
#         return self.data.shape[1]
#
#     def distance(self, other):
#         # distance between self and other
#         if not isinstance(other, SpaceData):
#             raise AttributeError("Cannot find distance between type %s and type %s" % (self.__class__, other.__class))
#         if other.ndim != self.ndim:
#             raise AttributeError("Incompatible dimensions")
#         if other.ndata != self.ndata:
#             raise AttributeError("Incompatible number of datapoints")
#         return np.sqrt(np.sum((self.data - other.data) ** 2, axis=1))


class NetworkData(Data):

    def __init__(self, graph, network_points):
        self.graph = graph
        self.data = [x if isinstance(x, NetworkPoint) else NetworkPoint(self.graph, **x) for x in network_points]

    @property
    def ndata(self):
        return len(self.data)

    def distance(self, other):
        # distance between self and other
        if not isinstance(other, NetworkData):
            raise AttributeError("Cannot find distance between type %s and type %s" % (self.__class__, other.__class))
        return [x.network_distance(y) for (x, y) in zip(self.data, other.data)]


class DataArray(Data, np.ndarray):

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type

        obj = np.asarray(input_array).view(cls)

        # check dimensions
        if obj.ndim == 0:
            raise AttributeError("Input array has no data")

        if obj.ndim == 1:
            obj = obj.reshape((obj.size, 1))

        elif obj.ndim > 2:
            # separate by last dimension, flattening all other dimensions
            # NB: the order of unravelling here is 'F', which is different to numpy's default 'C' method
            # that is used by the 'flat' iterator
            nd = obj.shape[-1]
            ndata = obj[..., 0].size
            obj = obj.reshape((ndata, nd), order='F')

        # Finally, we must return the newly created object:
        return obj

    def getdim(self, dim):
        return DataArray(self[:, dim])

    @property
    def nd(self):
        return self.shape[1]

    @property
    def ndata(self):
        return self.shape[0]


class SpaceTimeDataArray(DataArray):
    """
    DataArray in which the first dimension is assumed to represent time and the remainder space
    """
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type

        obj = super(SpaceTimeDataArray, cls).__new__(cls, input_array)

        if obj.nd < 2:
            raise AttributeError("Must have >= 2 dimensions for ST data")

        return obj

    @property
    def time(self):
        # time component of datapoints
        return DataArray(self[:, 0])

    @property
    def space(self):
        # space component of datapoints
        return DataArray(self[:, 1:])


class CartesianData(DataArray):

    def distance(self, other):
        # distance between self and other
        if not isinstance(other, self.__class__):
            raise AttributeError("Cannot find distance between type %s and type %s" % (self.__class__, other.__class))
        if other.nd != self.nd:
            raise AttributeError("Incompatible dimensions")
        if other.ndata != self.ndata:
            raise AttributeError("Incompatible number of datapoints")

        # write out exact formula for common cases: nd == 1, 2, 3
        if self.nd == 1:
            return np.sqrt((self - other) ** 2)

        if self.nd == 2:
            return np.sqrt((self[:, 0] - other[:, 0])**2 + (self[:, 1] - other[:, 1])**2)

        if self.nd == 3:
            return np.sqrt(
                (self[:, 0] - other[:, 0])**2
                + (self[:, 1] - other[:, 1])**2
                + (self[:, 2] - other[:, 2])**2
            )

        # otherwise use (slower) generic algorithm
        return np.sqrt(np.sum((self - other) ** 2, axis=1))


class CartesianSpaceTimeData(SpaceTimeDataArray, CartesianData):
    """
    SpaceTime data, where the distance function is defined in the same way as for Cartesian data
    As for SpaceTimeDataArray, the first dimension refers to time
    """
    @property
    def space(self):
        # space component of datapoints
        return CartesianData(self[:, 1:])


