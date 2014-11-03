__author__ = 'gabriel'
import numpy as np
from network.geos import NetworkPoint
from warnings import warn


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


class DataArray(Data, np.ndarray):

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type

        obj = np.asarray(input_array).view(cls)
        obj.original_shape = None

        # check dimensions
        if obj.ndim == 0:
            # input is either empty or a single value
            try:
                obj = obj.reshape(1, 1)
            except ValueError:
                raise AttributeError("Input array has no data")

        elif obj.ndim == 1:
            obj = obj.reshape((obj.size, 1))

        elif obj.ndim > 2:
            # separate by last dimension, flattening all other dimensions
            # NB: the order of unravelling here is 'F', which is different to numpy's default 'C' method
            # that is used by the 'flat' iterator
            nd = obj.shape[-1]
            ndata = obj[..., 0].size
            # record original shape for later rebuilding
            obj.original_shape = obj[..., 0].shape
            obj = obj.reshape((ndata, nd), order='F')

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None:
            return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.original_shape = getattr(obj, 'original_shape', None)
        # We do not need to return anything

    def __reduce__(self):
        # required for pickling.
        # call parent reduce method first
        recon, initargs, state = super(DataArray, self).__reduce__()
        # add any extra attributes
        state += (self.__dict__,)
        return recon, initargs, state

    def __setstate__(self, state):
        # called upon unpickling
        self.__dict__ = state[-1]
        # call parent setstate
        super(DataArray, self).__setstate__(state[:-1])

    def getdim(self, dim):
        # extract required dimension
        obj = DataArray(self[:, dim])
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
        new_obj = DataArray(np.hstack((self, obj)))
        new_obj.original_shape = self.original_shape
        return new_obj

    @property
    def nd(self):
        return self.shape[1]

    @property
    def ndata(self):
        return self.shape[0]

    @property
    def separate(self):
        return tuple(self.toarray(i) for i in range(self.nd))

    def toarray(self, dim):
        # return an np.ndarray object with the same shape as the original
        # dim is a non-optional input argument detailing which dimension is required, unless self.nd=1 in which case
        # dim=0 is assumed.  If all dimensions are required, use separate instead
        if dim > (self.nd - 1):
            raise AttributeError("Requested dim %d but this array has nd %d" % (dim, self.nd))
        if self.original_shape:
            return self.getdim(dim).reshape(self.original_shape, order='F').view(np.ndarray)
        else:
            return self.getdim(dim).squeeze().view(np.ndarray)



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
        res = DataArray(self[:, 0])
        res.original_shape = self.original_shape
        return res

    @time.setter
    def time(self, time):
        self[:, 0:1] = time

    @property
    def space(self):
        # space component of datapoints
        res = DataArray(self[:, 1:])
        res.original_shape = self.original_shape
        return res

    @space.setter
    def space(self, space):
        self[:, 1:] = DataArray(space)


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


