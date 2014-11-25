__author__ = 'gabriel'
import numpy as np
from network.geos import NetworkPoint
from warnings import warn
import ipdb


def negative_time_dimension(data_array):
    """
    Return a copy of the input array with the time dimension taking negative values
    :param data_array:
    :return:
    """
    # copy
    new_data = data_array[:]
    # reverse time dimension
    new_data[:, 0] *= -1.0
    return new_data


def exp(data_array):
    res = data_array.__class__(np.exp(data_array.data))
    res.original_shape = data_array.original_shape
    return res


class Data(object):

    @property
    def nd(self):
        raise NotImplementedError()


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


class DataArray(Data):

    def __init__(self, obj):
        self.original_shape = None

        if isinstance(obj, DataArray):
            self.data = obj.data.copy()
            self.original_shape = obj.original_shape
            return


        if not isinstance(obj, np.ndarray):
            obj = np.array(obj, dtype=float)
        else:
            obj = obj.astype(float)

        # check dimensions
        if obj.ndim == 0:
            # input is a single value
            self.data = obj.reshape(1, 1)

        elif obj.ndim == 1:
            self.data = obj.reshape((obj.size, 1))

        elif obj.ndim == 2:
            self.data = obj

        elif obj.ndim > 2:
            # separate by last dimension, flattening all other dimensions
            # NB: the order of unravelling here is 'F', which is different to numpy's default 'C' method
            # that is used by the 'flat' iterator
            nd = obj.shape[-1]
            ndata = obj[..., 0].size
            # record original shape for later rebuilding
            self.original_shape = obj[..., 0].shape
            self.data = obj.reshape((ndata, nd), order='F')

    def copy(self):
        return self.__class__(self)

    @classmethod
    def from_meshgrid(cls, *args):
        # create an instance from the output of meshgrid
        data = np.concatenate([t[..., np.newaxis] for t in args], axis=len(args))
        return cls(data)

    @classmethod
    def from_args(cls, *args):
        # create an instance from the input args, each of which represents a dimension
        # this loses the original shape
        data = np.vstack([t.flat for t in args]).transpose()
        return cls(data)

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, self.data.__str__())

    def __builtin_combine__(self, other, func):
        if isinstance(other, DataArray):
            res = self.__class__(func(other.data))
        else:
            res = self.__class__(func(other))
        res.original_shape = self.original_shape
        return res

    def __builtin_unary__(self, func, *args, **kwargs):
        res = self.__class__(func(*args, **kwargs))
        res.original_shape = self.original_shape
        return res

    def __eq__(self, other):
        return np.all(self.__builtin_combine__(other, self.data.__eq__))

    def __add__(self, other):
        return self.__builtin_combine__(other, self.data.__add__)

    def __sub__(self, other):
        return self.__builtin_combine__(other, self.data.__sub__)

    def __div__(self, other):
        return self.__builtin_combine__(other, self.data.__div__)

    def __mul__(self, other):
        return self.__builtin_combine__(other, self.data.__mul__)

    def __gt__(self, other):
        return self.__builtin_combine__(other, self.data.__gt__)

    def __lt__(self, other):
        return self.__builtin_combine__(other, self.data.__lt__)

    def __ge__(self, other):
        return self.__builtin_combine__(other, self.data.__ge__)

    def __le__(self, other):
        return self.__builtin_combine__(other, self.data.__le__)

    def __neg__(self):
        return self.__builtin_unary__(self.data.__neg__)

    def __pow__(self, power, modulo=None):
        return self.__builtin_unary__(self.data.__pow__, power, modulo)

    def __len__(self):
        return self.ndata

    def __getitem__(self, item):
        ## TODO: ideally we should be returning a similar type here, IF that makes sense
        ## obj[0], obj[[0, 4, 7]] : return similar type
        ## obj[:, 0] : don't?
        return self.data.__getitem__(item)

    def __setitem__(self, i, value):
        self.data.__setitem__(i, value)

    def sumdim(self):
        # sums over dimensions, returning a class type with a single dimension and same original_shape
        res = self.__class__(self.data.sum(axis=1))
        res.original_shape = self.original_shape
        return res

    def getdim(self, dim):
        # extract required dimension
        obj = DataArray(self.data[:, dim])
        # set original shape manually
        obj.original_shape = self.original_shape
        return obj

    def adddim(self, obj, strict=True):
        obj = DataArray(obj)
        if obj.ndata != self.ndata:
            raise AttributeError("Cannot add dimension because ndata does not match")
        if strict:
            # check shape
            if obj.original_shape is None and self.original_shape:
                warn("Adding data with no original shape - it will be coerced into the existing shape")
            if obj.original_shape != self.original_shape and self.original_shape is not None:
                raise AttributeError("Attempting to add data with incompatible original shape.  Set strict=False to bypass this check.")
        new_obj = DataArray(np.hstack((self.data, obj)))
        new_obj.original_shape = self.original_shape
        return new_obj

    def getrows(self, idx):
        if hasattr(idx, '__iter__'):
            new_data = self.data[idx]
        else:
            new_data = self.data[np.newaxis, idx]
        res = self.__class__(new_data)
        # NB cannot restore original shape here, so leave as None
        return res

    @property
    def nd(self):
        return self.data.shape[1]

    @property
    def ndata(self):
        return self.data.shape[0]

    @property
    def separate(self):
        return tuple(self.toarray(i) for i in range(self.nd))

    def toarray(self, dim):
        # return an np.ndarray object with the same shape as the original
        # dim is a non-optional input argument detailing which dimension is required
        # if all dimensions are required, use separate instead
        if dim > (self.nd - 1):
            raise AttributeError("Requested dim %d but this array has nd %d" % (dim, self.nd))
        if self.original_shape:
            return self.data[:, dim].reshape(self.original_shape, order='F')
        else:
            return self.data[:, dim].squeeze()


class SpaceTimeDataArray(DataArray):
    """
    DataArray in which the first dimension is assumed to represent time and the remainder space
    """
    time_class = DataArray
    space_class = DataArray

    def __init__(self, obj):
        super(SpaceTimeDataArray, self).__init__(obj)
        # if self.nd < 2:
        #     raise AttributeError("Must have >= 2 dimensions for ST data")

    @property
    def time(self):
        # time component of datapoints
        res = self.time_class(self[:, 0])
        res.original_shape = self.original_shape
        return res

    @time.setter
    def time(self, time):
        self.data[:, 0:1] = DataArray(time).data

    @property
    def space(self):
        # space component of datapoints
        res = self.space_class(self[:, 1:])
        res.original_shape = self.original_shape
        return res

    @space.setter
    def space(self, space):
        self.data[:, 1:] = DataArray(space).data


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
            res = np.sqrt((self - other) ** 2)

        if self.nd == 2:
            res = np.sqrt((self[:, 0] - other[:, 0])**2 + (self[:, 1] - other[:, 1])**2)

        if self.nd == 3:
            res = np.sqrt(
                (self[:, 0] - other[:, 0])**2
                + (self[:, 1] - other[:, 1])**2
                + (self[:, 2] - other[:, 2])**2
            )

        # otherwise use (slower) generic algorithm
        res = np.sqrt(np.sum((self - other) ** 2, axis=1))

        return DataArray(res)


class CartesianSpaceTimeData(SpaceTimeDataArray, CartesianData):
    """
    SpaceTime data, where the distance function is defined in the same way as for Cartesian data
    As for SpaceTimeDataArray, the first dimension refers to time
    """
    space_class = CartesianData



