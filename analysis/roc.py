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

    def set_grid(self, side_length):
        if not self.poly:
            # find minimal bounding rectangle
            xmin, ymin = np.min(self.data, axis=0)
            xmax, ymax = np.max(self.data, axis=0)
            self.poly = geos.Polygon([
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
            ])
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
