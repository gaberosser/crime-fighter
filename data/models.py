__author__ = 'gabriel'
import numpy as np
from network.streetnet import NetPoint, StreetNet
from warnings import warn
from copy import copy as cpy


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


class DataArray(object):

    datatype = float
    combination_output_class = None

    def __init__(self, obj, copy=True, **kwargs):
        self.original_shape = None

        # if a dtype kwarg has been supplied, use that
        dtype = kwargs.get('dtype', self.datatype)

        if isinstance(obj, self.__class__):
            if copy:
                self.data = obj.data.copy()
            else:
                self.data = obj.data
            self.original_shape = cpy(obj.original_shape)
            return

        if not isinstance(obj, np.ndarray):
            obj = np.array(obj, dtype=dtype, copy=copy)
        else:
            obj = obj.astype(dtype, copy=copy)

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
            # NB: the order of unravelling here is numpy's default 'C' method
            # the same ordering is used by the 'flat' iterator
            nd = obj.shape[-1]
            ndata = obj[..., 0].size
            # record original shape for later rebuilding
            self.original_shape = obj[..., 0].shape
            dim_arrs = []
            for i in range(nd):
                dim_arrs.append(obj[..., i].flatten())
            self.data = np.vstack(dim_arrs).transpose()

    def copy(self):
        return self.__class__(self, copy=True)

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

    def __builtin_combine__(self, other, func, dtype=None):
        cls = self.combination_output_class or self.__class__

        if isinstance(other, DataArray):
            res = cls(func(other.data), dtype=dtype)
        else:
            res = cls(func(other), dtype=dtype)
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

    def __radd__(self, other):
        return self.__builtin_combine__(other, self.data.__radd__)

    def __sub__(self, other):
        return self.__builtin_combine__(other, self.data.__sub__)

    def __rsub__(self, other):
        return self.__builtin_combine__(other, self.data.__rsub__)

    def __div__(self, other):
        return self.__builtin_combine__(other, self.data.__div__)

    def __rdiv__(self, other):
        return self.__builtin_combine__(other, self.data.__rdiv__)

    def __mul__(self, other):
        return self.__builtin_combine__(other, self.data.__mul__)

    def __and__(self, other):
        return self.__builtin_combine__(other, self.data.__and__, dtype=bool)

    def __gt__(self, other):
        return self.__builtin_combine__(other, self.data.__gt__, dtype=bool)

    def __lt__(self, other):
        return self.__builtin_combine__(other, self.data.__lt__, dtype=bool)

    def __ge__(self, other):
        return self.__builtin_combine__(other, self.data.__ge__, dtype=bool)

    def __le__(self, other):
        return self.__builtin_combine__(other, self.data.__le__, dtype=bool)

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

    ## TODO: check this doesn't break things
    def __iter__(self):
        return self.data.__iter__()

    def sumdim(self):
        # sums over dimensions, returning a class type with a single dimension and same original_shape
        res = self.__class__(self.data.sum(axis=1))
        res.original_shape = self.original_shape
        return res

    def getdim(self, dim):
        # extract required dimension
        obj = DataArray(self.data[:, dim], copy=False)
        # set original shape manually
        obj.original_shape = self.original_shape
        return obj

    def adddim(self, obj, strict=True, type=None):
        dest_cls = type or DataArray
        obj = DataArray(obj)
        if obj.ndata != self.ndata:
            raise AttributeError("Cannot add dimension because ndata does not match")
        if strict:
            # check shape
            if obj.original_shape is None and self.original_shape:
                warn("Adding data with no original shape - it will be coerced into the existing shape")
            if obj.original_shape != self.original_shape and self.original_shape is not None:
                raise AttributeError("Attempting to add data with incompatible original shape.  Set strict=False to bypass this check.")
        new_obj = dest_cls(np.hstack((self.data, obj.data)))
        new_obj.original_shape = self.original_shape
        return new_obj

    def getone(self, idx):
        if hasattr(idx, '__iter__'):
            raise AttributeError("Input must be an integer index")
        if self.nd == 1:
            return self.data[idx, 0]
        else:
            return self.getrows(idx)

    def getrows(self, idx):
        if hasattr(idx, '__iter__'):
            new_data = self.data[idx]
        else:
            new_data = self.data[np.newaxis, idx]
        res = self.__class__(new_data, copy=False)
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

    def toarray(self, dim=None):
        # return an np.ndarray object with the same shape as the original
        # dim is a non-optional input argument detailing which dimension is required
        # if all dimensions are required, use separate instead
        if dim is None:
            if self.nd == 1:
                dim = 0
            else:
                raise AttributeError("Must specify the dimension required when ndim > 1.")
        if dim > (self.nd - 1):
            raise AttributeError("Requested dim %d but this array has nd %d" % (dim, self.nd))
        res = self.data[:, dim]
        if self.original_shape:
            res = res.reshape(self.original_shape)
        return res


class SpaceTimeDataArray(DataArray):
    """
    DataArray in which the first dimension is assumed to represent time and the remainder space
    """
    time_class = DataArray
    space_class = DataArray

    # def __init__(self, obj):
    #     super(SpaceTimeDataArray, self).__init__(obj)
        # if self.nd < 2:
        #     raise AttributeError("Must have >= 2 dimensions for ST data")

    @property
    def time(self):
        # time component of datapoints
        res = self.time_class(self[:, 0].astype(float, copy=False))
        res.original_shape = self.original_shape
        return res

    @time.setter
    def time(self, time):
        self.data[:, 0:1] = self.time_class(time).data

    @property
    def space(self):
        # space component of datapoints
        res = self.space_class(self[:, 1:], copy=False)
        res.original_shape = self.original_shape
        return res

    @space.setter
    def space(self, space):
        self.data[:, 1:] = self.space_class(space).data


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

        elif self.nd == 2:
            res = np.sqrt((self[:, 0] - other[:, 0])**2 + (self[:, 1] - other[:, 1])**2)

        elif self.nd == 3:
            res = np.sqrt(
                (self[:, 0] - other[:, 0])**2
                + (self[:, 1] - other[:, 1])**2
                + (self[:, 2] - other[:, 2])**2
            )

        # otherwise use (slower) generic algorithm
        else:
            res = np.sqrt(((self - other) ** 2).sumdim().toarray(0))

        res = DataArray(res)
        if self.original_shape is not None:
            res.original_shape = self.original_shape
        return res

    def angle(self, other):
        """
        Return the angle in radians, anti-clockwise relative to the horizontal, of the vector that connects
        THIS POINT TO ANOTHER POINT
        """
        if other.nd != self.nd:
            raise AttributeError("Incompatible dimensions")
        if other.ndata != self.ndata:
            raise AttributeError("Incompatible number of datapoints")

        if self.nd == 1:
            raise AttributeError("Cannot compute angle difference between 1D points")
        elif self.nd == 2:
            res = np.arctan2(other[:, 1] - self[:, 1], other[:, 0] - self[:, 0])
        else:
            raise NotImplementedError("Have not implemented angles in >2D yet.")
        res = DataArray(res)
        if self.original_shape is not None:
            res.original_shape = self.original_shape
        return res

class CartesianSpaceTimeData(SpaceTimeDataArray, CartesianData):
    """
    SpaceTime data, where the distance function is defined in the same way as for Cartesian data
    As for SpaceTimeDataArray, the first dimension refers to time
    """
    ## FIXME: the distance and angle methods have no meaning here?
    space_class = CartesianData


class NetworkData(DataArray):

    datatype = object
    combination_output_class = DataArray

    def __init__(self, network_points, copy=True, **kwargs):
        """
        Create a 1D NetworkData array of network points.
        :param network_points: iterable containing instances of NetPoint
        :param kwargs: May contain the optional 'strict' attribute (default True): If True, all points are checked upon
        instantiation to ensure they have the same network object. This should be very fast.
        :return:
        """
        super(NetworkData, self).__init__(network_points, copy=copy, **kwargs)
        if self.nd != 1:
            raise AttributeError("NetworkData must be one-dimensional.")
        if kwargs.pop('strict', True):
            for x in self.data.flat:
                if x.graph is not self.graph:
                    raise AttributeError("All network points must be defined on the same graph")

    @property
    def graph(self):
        if self.ndata:
            return self.data[0, 0].graph
        else:
            return None

    @classmethod
    def from_cartesian(cls, net, data, grid_size=50, radius=None, return_failure_idx=False):
        """
        Generate a NetworkData object for the (x, y) coordinates in data.
        :param net: The StreetNet object that will be used to snap network points.
        :param data: Either a 2D DataArray object or data that can be used to instantiate one.
        :param grid_size: The size of the grid used to index the network. This is used to speed up snapping.
        :param radius: Optionally supply a maximum snapping distance.
        :param return_failure_idx: If True, output includes the index of points that did not snap. Otherwise, failed
        snaps are removed silently from the array.
        :return: NetworkData object
        """
        data = DataArray(data)
        if data.nd != 2:
            raise AttributeError("Input data must be 2D")
        grid_edge_index = net.build_grid_edge_index(grid_size) if grid_size else None
        net_points = []
        fail_idx = []
        for i, (x, y) in enumerate(data):
            t = NetPoint.from_cartesian(net, x, y,
                                        grid_edge_index=grid_edge_index,
                                        radius=radius)
            if t is None:
                if return_failure_idx:
                    fail_idx.append(i)
            else:
                net_points.append(t)
        obj = cls(net_points)
        if return_failure_idx:
            return obj, fail_idx
        else:
            return obj

    def to_cartesian(self):
        """
        Convert all network points into Cartesian coordinates using linear interpolation of the edge LineStrings
        :return: CartesianData
        """
        res = CartesianData([t.cartesian_coords for t in self.data.flat])
        res.original_shape = self.original_shape
        return res

    @property
    def ndata(self):
        return len(self.data)

    def distance_function(self, x, y):
        return x.distance(y)

    def distance(self, other, directed=False):
        # distance between self and other
        if not self.ndata == other.ndata:
            raise AttributeError("Lengths of the two data arrays are incompatible")
        return DataArray([x.distance(y) for (x, y) in zip(self.data.flat, other.data.flat)])

    def euclidean_distance(self, other):
        """ Euclidean distance between the data """
        return DataArray([x.euclidean_distance(y) for (x, y) in zip(self.data.flat, other.data.flat)])


class NetworkSpaceTimeData(SpaceTimeDataArray):

    datatype = object
    space_class = NetworkData

    @property
    def graph(self):
        return self.space.graph

    @classmethod
    def from_cartesian(cls, net, data, return_failure_idx=False, **kwargs):
        """
        Generate a network-time data array from the input data, which should be castable to SpaceTimeDataArray
        Parameters are exactly as for NetworkData.from_cartesian
        :return: NetworkSpaceTimeData array where data have been snapped spatially to the network. If no snap was
        found, these points are skipped.
        If return_failure_idx is True, an index listing failed snaps is returned too.
        """
        data = SpaceTimeDataArray(data)
        snapped, fail_idx = cls.space_class.from_cartesian(net, data.space, return_failure_idx=True, **kwargs)
        success_idx = np.array(list(set(range(data.ndata)) - set(fail_idx)))
        success_idx.sort()
        time = data.time.getrows(success_idx)
        new_array = time.adddim(snapped, type=cls)
        if return_failure_idx:
            return new_array, fail_idx
        else:
            return new_array