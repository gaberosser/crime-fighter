__author__ = 'gabriel'

from django.contrib.gis import geos
import numpy as np
import math
from spatial import create_spatial_grid, random_points_within_poly
from data.models import DataArray
import collections
import warnings


class RocSpatialGrid(object):

    def __init__(self, data=None, poly=None, data_index=None):
        self.poly = poly
        self.side_length = None
        self._intersect_grid = None
        self._extent_grid = None
        self._full_grid_square = None
        self._grid_labels = None
        self.centroids = None
        self.sample_points = None
        self.a = None
        self.prediction_values = None
        self._data = None
        self.index = None
        self.set_data(data, index=data_index)

    @property
    def data(self):
        if self._data is not None:
            return self._data
        raise AttributeError("Data have not been set yet, call set_data")

    @property
    def ndata(self):
        return self.data.ndata
        # return len(self.data)

    def set_data(self, data, index=None):
        if data is not None:
            data = DataArray(data)
            if data.nd != 2:
                raise AttributeError("Data must be a 2D array")
        self._data = data
        if index is not None:
            if len(index) != data.ndata:
                raise AttributeError("Length of index vector must match number of input data")
        self.index = index

    def generate_bounding_poly(self):
        # called when no polygon is provided, computes the bounding rectangle for the data
        xmin, ymin = np.min(self.data, axis=0)
        xmax, ymax = np.max(self.data, axis=0)
        return geos.Polygon([
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
        ])

    def set_grid(self, length_or_arr, *args, **kwargs):
        '''
        Set the ROC grid.
        :param length_or_arr: Either a scalar, interpreted as the side length of the grid square, OR an array of
          geos.Polygon or geos.MultiPolygon objects
        :param args: Passed to set_sample_points
        :param kwargs: Passed to set_sample_points
        :return: None
        '''
        # reset prediction values
        self.prediction_values = None
        if not self.poly:
            # find minimal bounding rectangle
            self.poly = self.generate_bounding_poly()

        if hasattr(length_or_arr, '__iter__'):
            # list of polygons supplied
            self.side_length = None
            self._intersect_grid = length_or_arr
            self._extent_grid = [x.extent for x in length_or_arr]
            # assume none of these are full
            ## FIXME: improve this by checking whether it's a square?
            self._full_grid_square = [False] * self.ngrid
        else:
            self.side_length = length_or_arr
            self._intersect_grid, self._extent_grid, self._full_grid_square = create_spatial_grid(self.poly, self.side_length)
        self.centroids = np.array([t.centroid.coords for t in self._intersect_grid])
        self.a = np.array([t.area for t in self._intersect_grid])
        self.set_sample_points(*args, **kwargs)

    def copy_grid(self, roc):
        # reset prediction values
        self.prediction_values = None
        self._intersect_grid = list(roc.igrid)
        self._extent_grid = list(roc.egrid)
        self._full_grid_square = list(roc.full_grid_square)
        self.centroids = np.array(roc.centroids)
        self.sample_points = np.array(roc.sample_points)
        self.a = np.array(roc.a)

    def set_sample_points(self, *args, **kwargs):
        # sample points here are just the centroids
        self.sample_points = DataArray(self.centroids)
        self.sample_points.original_shape = (1, self.ngrid)

    @property
    def igrid(self):
        if self._intersect_grid is not None:
            return self._intersect_grid
        raise AttributeError("Grid has not been computed, run set_grid with grid length")

    @property
    def egrid(self):
        if self._extent_grid is not None:
            return self._extent_grid
        raise AttributeError("Grid has not been computed, run set_grid with grid length")

    @property
    def ngrid(self):
        if self._intersect_grid is not None:
            return len(self._intersect_grid)
        raise AttributeError("Grid has not been computed, run set_grid with grid length")

    @property
    def full_grid_square(self):
        if self._full_grid_square is not None:
            return self._full_grid_square
        raise AttributeError("Grid has not been computed, run set_grid with grid length")

    def set_prediction(self, prediction):
        if prediction.shape[1] != self.ngrid:
            raise AttributeError("Dim 1 of supplied prediction does not match grid")
        self.prediction_values = np.mean(prediction, axis=0)

    @property
    def prediction_rank(self):
        if self.prediction_values is None:
            raise AttributeError("No prediction supplied, run set_prediction")
        return np.argsort(self.prediction_values)[::-1]

    @property
    def true_count(self):

        n = []
        for xmin, ymin, xmax, ymax in self.egrid:
            n.append(sum(
                (self.data[:, 0] >= xmin)
                & (self.data[:, 0] < xmax)
                & (self.data[:, 1] >= ymin)
                & (self.data[:, 1] < ymax)
            ))
        return np.array(n)

    @property
    def true_grid_index(self):
        """
        Return the crime indices in each grid square.
        The index is self.index if it exists, otherwise just a plain lookup index.
        """
        indices = []
        for xmin, ymin, xmax, ymax in self.egrid:
            this_idx = (
                (self.data[:, 0] >= xmin)
                & (self.data[:, 0] < xmax)
                & (self.data[:, 1] >= ymin)
                & (self.data[:, 1] < ymax)
            )
            if not np.any(this_idx):
                indices.append(None)
            elif self.index is not None:
                indices.append(self.index[this_idx])
            else:
                indices.append(np.where(this_idx)[0])
        return indices

    def evaluate(self):

        # count actual crimes in testing dataset on same grid
        true_grid_ind = np.array(self.true_grid_index)[self.prediction_rank]
        true_counts = np.array([(t.size if t is not None else 0) for t in true_grid_ind])
        # true_counts = self.true_count[self.prediction_rank]
        true_counts_sorted = np.sort(self.true_count)[::-1]
        pred_values = self.prediction_values[self.prediction_rank]
        area = self.a[self.prediction_rank]
        total_area = sum(area)

        N = sum(true_counts)
        n = np.cumsum(true_counts)
        carea = np.cumsum(area) / total_area
        cfrac = n / float(N)
        cfrac_max = np.cumsum(true_counts_sorted) / float(N)
        pai = cfrac * (total_area / np.cumsum(area))

        res = {
            'prediction_rank': self.prediction_rank,
            'prediction_values': pred_values,
            'cumulative_area': carea,
            'cumulative_crime': cfrac,
            'cumulative_crime_count': n,
            'cumulative_crime_max': cfrac_max,
            'pai': pai,
            'ranked_crime_id': true_grid_ind,
        }

        return res


class RocSpatialGridMonteCarloIntegration(RocSpatialGrid):

    def set_sample_points(self, n_sample_per_grid, respect_boundary=True, *args, **kwargs):
        """ Generate n_sample_per_grid sample points per grid unit
         Return n_sample_per_grid x self.ndata x 2 array, final dim is x, y """

        if self.side_length is None:
            # grid was supplied as an array
            # slow version: need to iterate over the polygons
            xres = np.empty((n_sample_per_grid, self.ngrid), dtype=float)
            yres = np.empty((n_sample_per_grid, self.ngrid), dtype=float)
            for i, p in enumerate(self.igrid):
                xres[:, i], yres[:, i] = random_points_within_poly(p, n_sample_per_grid)

        else:
            xmins = np.array([x[0] for x in self.egrid])
            ymins = np.array([x[1] for x in self.egrid])
            xres = np.random.rand(n_sample_per_grid, self.ngrid) * self.side_length + xmins
            yres = np.random.rand(n_sample_per_grid, self.ngrid) * self.side_length + ymins

        if respect_boundary:
            # loop over grid squares that are incomplete
            for i in np.where(np.array(self.full_grid_square) == False)[0]:
                inside_idx = np.array([geos.Point(x, y).within(self.poly) for x, y in zip(xres[:, i], yres[:, i])])
                # pad empty parts with repeats of the centroid location
                num_empty = n_sample_per_grid - sum(inside_idx)
                if num_empty:
                    cx, cy = self.centroids[i]
                    rem_x = np.concatenate((xres[inside_idx, i], cx * np.ones(num_empty)))
                    rem_y = np.concatenate((yres[inside_idx, i], cy * np.ones(num_empty)))
                    xres[:, i] = rem_x
                    yres[:, i] = rem_y

        self.sample_points = DataArray.from_meshgrid(xres, yres)


class WeightedRocSpatialGrid(RocSpatialGrid):

    def __init__(self, data=None, poly=None, half_life=1.):
        super(WeightedRocSpatialGrid, self).__init__(data=data, poly=poly)
        self.decay_constant = None
        self.set_decay_constant(half_life)

    def set_decay_constant(self, half_life):
        self.decay_constant = np.log(2.) / half_life

    def set_data(self, data):
        # Data is a 2D array, cols: t, x, y
        # Time = 0 is the present, by convention
        if data is not None:
            if data.ndim != 2:
                raise AttributeError("Data must be a 2D array")
            if data.shape[1] != 3:
                raise AttributeError("Second dimension must be of length 3 (spatiotemporal data)")
        # self._data = [geos.Point(list(t)) for t in data]
        self._data = data

    @property
    def weights(self):
        return np.exp(-self.decay_constant * self._data[:, 0])

    @property
    def sum_weights(self):
        return np.sum(self.weights)

    @property
    def true_count(self):

        n = []
        for xmin, ymin, xmax, ymax in self.egrid:
            idx = (self.data[:, 1] >= xmin) \
                & (self.data[:, 1] < xmax) \
                & (self.data[:, 2] >= ymin) \
                & (self.data[:, 2] < ymax)
            n.append(sum(self.weights[idx]))

        return np.array(n)

    def evaluate(self):

        # count actual crimes in testing dataset on same grid
        true_counts = self.true_count[self.prediction_rank]
        true_counts_sorted = np.sort(self.true_count)[::-1]
        pred_values = self.prediction_values[self.prediction_rank]
        area = self.a[self.prediction_rank]
        total_area = sum(area)

        carea = np.cumsum(area) / total_area
        cfrac = np.cumsum(true_counts) / self.sum_weights
        cfrac_max = np.cumsum(true_counts_sorted) / self.sum_weights
        pai = cfrac * (total_area / np.cumsum(area))

        res = {
            'prediction_rank': self.prediction_rank,
            'prediction_values': pred_values,
            'cumulative_area': carea,
            'cumulative_crime': cfrac,
            'cumulative_crime_max': cfrac_max,
            'pai': pai,
        }

        return res