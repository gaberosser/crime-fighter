__author__ = 'gabriel'

from django.contrib.gis import geos
import numpy as np
import pp
import mcint
import math
from spatial import create_spatial_grid
import collections


class RocSpatial(object):

    def __init__(self, data=None, poly=None):
        self.poly = poly
        self._intersect_grid = None
        self._extent_grid = None
        self.centroids = None
        self.a = None
        self.prediction_values = None
        self._data = None
        self.set_data(data)

    @property
    def data(self):
        if self._data is not None:
            return self._data
        raise AttributeError("Data have not been set yet, call set_data")

    @property
    def ndata(self):
        return self.data.shape[0]
        # return len(self.data)

    def set_data(self, data):
        if data is not None:
            if data.ndim != 2:
                raise AttributeError("Data must be a 2D array")
            if data.shape[1] != 2:
                raise AttributeError("Second dimension must be of length 2 (spatial data)")
        # self._data = [geos.Point(list(t)) for t in data]
        self._data = data

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

    def set_grid(self, side_length):
        if not self.poly:
            # find minimal bounding rectangle
            self.poly = self.generate_bounding_poly()
        self._intersect_grid, self._extent_grid = create_spatial_grid(self.poly, side_length)
        self.centroids = np.array([t.centroid.coords for t in self._intersect_grid])
        self.a = np.array([t.area for t in self._intersect_grid])

    def copy_grid(self, roc):
        self._intersect_grid = roc._intersect_grid
        self._extent_grid = roc._extent_grid
        self.centroids = np.array(roc.centroids)
        self.a = np.array(roc.a)

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

    def set_prediction(self, prediction):
        if len(prediction) != len(self.igrid):
            raise AttributeError("Length of supplied prediction does not match grid")
        self.prediction_values = prediction

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

    def evaluate(self):

        # count actual crimes in testing dataset on same grid
        true_counts = self.true_count[self.prediction_rank]
        true_counts_sorted = np.sort(self.true_count)[::-1]
        pred_values = self.prediction_values[self.prediction_rank]
        area = self.a[self.prediction_rank]
        total_area = sum(area)

        N = sum(true_counts)
        carea = np.cumsum(area) / total_area
        cfrac = np.cumsum(true_counts) / float(N)
        cfrac_max = np.cumsum(true_counts_sorted) / float(N)
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


class WeightedRocSpatial(RocSpatial):

    def __init__(self, data=None, poly=None, half_life=1.):
        super(WeightedRocSpatial, self).__init__(data=data, poly=poly)
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