__author__ = 'gabriel'
from copy import copy
import numpy as np
import tools
from analysis.spatial import (create_spatial_grid,
                              random_points_within_poly,
                              shapely_rectangle_from_vertices,
                              geodjango_rectangle_from_vertices,
                              HAS_GEODJANGO)
from data.models import DataArray, NetworkSpaceTimeData, NetworkData, NetPoint, CartesianData
from shapely.geometry import Point, Polygon
from network.utils import network_walker_uniform_sample_points
from time import time
import logging
logger = logging.getLogger(__name__)

# optional failure upon importing plotting routines for headless operation
try:
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib import cm
    from plotting.spatial import plot_shapely_geos
    from plotting.utils import colour_mapper
    MPL = True
except ImportError:
    MPL = False

# optional failure when importing rtree
try:
    from rtree import index
    RTREE = True
except ImportError:
    RTREE = False    


if HAS_GEODJANGO:
    from django.contrib.gis import geos


class SpatialRoc(object):
    """
    Base spatial ROC class. This contains common functionality for assessing predictive accuracy for spatially-
    dependent predictions.
    """
    data_class = DataArray

    def __init__(self,
                 data=None,
                 poly=None,
                 data_index=None,
                 **kwargs):
        self.poly = poly

        self.centroids = None
        self.sample_units = None
        self.sample_points = None
        self.n_sample_point_per_unit = None
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

    def check_data(self, data):
        """ Can be used to test that data have the correct properties before assignment.
        :param data: Instance of self.data_class
        """
        pass

    def set_data(self, data, index=None):
        """
        Set the data, and the index if supplied. The purpose of the index is to make it possible to determine
        retrospectively which specific crimes were correctly predicted. If index is missing, just use the array
        indices
        :param data: Array or DataArray of data. Will be converted to DataArray upon loading.
        :param index: Optional array of crime indices, corresponding to the crimes in data.
        :return:
        """
        if data is not None:
            data = self.data_class(data)
            logger.info("Length of supplied data: %d", data.ndata)
            self.check_data(data)
            self._data = data
            if index is not None:
                if len(index) != data.ndata:
                    raise AttributeError("Length of index vector must match number of input data")
            else:
                index = np.arange(data.ndata)
            self.index = index
            logger.info("self.ndata: %d", self.ndata)

    def plot(self, show_sample_units=True,
             show_prediction=True,
             fmax=0.9,
             cmap='Reds',
             **kwargs
    ):
        """
        Plot the prediction and underlying sample units
        :param show_sample_units: If True, plot a representation of the sample units
        :param show_prediction: If True, plot a representation of the prediction
        :param fmax: The cumulant cutoff for plotting predictions, e.g. 0.9 indicates that the colour axis runs from the
        min value to the 90th percentile. All values above this are saturated.
        :return:
        """
        raise NotImplementedError()

    @property
    def sample_unit_size(self):
        """ Return the size of each sample unit, e.g. length or area. """
        raise NotImplementedError()

    def set_sample_units(self, *args, **kwargs):
        '''
        Set the following attributes: sample units, centroids, sample_points, n_sample_points_per_unit
        :param args: Passed to set_sample_points
        :param kwargs: Passed to set_sample_points
        :return: None
        '''
        # reset prediction values
        self.prediction_values = None
        raise NotImplementedError()

    def copy_sample_units(self, roc):
        # reset prediction values
        self.prediction_values = None
        self.sample_units = copy(roc.sample_units)
        self.sample_points = copy(roc.sample_points)
        self.n_sample_point_per_unit = copy(roc.n_sample_point_per_unit)

    def set_sample_points(self, *args, **kwargs):
        """ Set the sampling locations within each sample unit """
        raise NotImplementedError()

    @property
    def n_sample_units(self):
        if self.sample_units is None:
            raise AttributeError("Sample units have not been computed, run set_sample_units")
        return len(self.sample_units)

    def set_prediction(self, prediction):
        logger.info("self.ndata: %d", self.ndata)
        if prediction.size != self.sample_points.ndata:
            raise AttributeError("Length of supplied prediction does not match sample points")
        # compute the mean predicted value per sample unit
        self.prediction_values = []
        tally = 0
        for n in self.n_sample_point_per_unit:
            self.prediction_values.append(np.mean(prediction[tally:tally + n]))
            tally += n
        self.prediction_values = np.array(self.prediction_values)

    @property
    def prediction_rank(self):
        if self.prediction_values is None:
            raise AttributeError("No prediction supplied, run set_prediction")
        return tools.numpy_most_compact_int_dtype(np.argsort(self.prediction_values)[::-1])

    # network-specific
    def in_sample_unit(self, sample_unit):
        """ Return bool array, one entry per data point. True indicates that the datum is within the sample unit. """
        raise NotImplementedError()

    @property
    def true_count(self):
        """ Return an array in which each element gives the number of crimes corresponding to a sample unit."""
        n = []
        for t in self.sample_units:
            n.append(sum(self.in_sample_unit(t)))
        return np.array(n)

    @property
    def true_index(self):
        """
        Return the crime indices in each sample unit.
        """
        indices = []
        for t in self.sample_units:
            this_idx = self.in_sample_unit(t)
            indices.append(self.index[this_idx])
        return np.array(indices, dtype=object)

    def evaluate(self, include_predictions=False):
        # check that there are some testing data, and return reduced results if not
        if self.data.ndata == 0:
            return {
                'prediction_rank': self.prediction_rank
            }

        # count actual crimes in testing dataset on same grid
        true_grid_ind = self.true_index[self.prediction_rank]
        true_counts = np.array([t.size for t in true_grid_ind])
        # true_counts = self.true_count[self.prediction_rank]
        true_counts_sorted = np.sort(self.true_count)[::-1]
        # disabling due to memory consumption
        # pred_values = self.prediction_values[self.prediction_rank]
        sample_unit_sizes = self.sample_unit_size[self.prediction_rank]
        total_sample_unit = sum(sample_unit_sizes)

        N = sum(true_counts)
        print N
        n = np.cumsum(true_counts)
        carea = np.cumsum(sample_unit_sizes) / total_sample_unit
        cfrac = n / float(N)
        cfrac_max = np.cumsum(true_counts_sorted) / float(N)
        pai = cfrac * (total_sample_unit / np.cumsum(sample_unit_sizes))

        res = {
            'prediction_rank': self.prediction_rank,
            'cumulative_area': carea,
            'cumulative_crime': cfrac,
            'cumulative_crime_count': n,
            'cumulative_crime_max': cfrac_max,
            'pai': pai,
            'ranked_crime_id': true_grid_ind,
        }

        if include_predictions:
            res['prediction_values'] = np.array(self.prediction_values)

        return res


class RocGrid(SpatialRoc):

    def __init__(self,
                 data=None,
                 poly=None,
                 data_index=None,
                 **kwargs):
        super(RocGrid, self).__init__(data=data,
                                      poly=poly,
                                      data_index=data_index,
                                      **kwargs)
        self.grid_polys = None
        self.full_grid_square = None
        self.side_length = None

    def check_data(self, data):
        if data.nd != 2:
            raise AttributeError("Data must be a 2D array")

    @staticmethod
    def generate_bounding_poly(data):
        """ Compute the bounding rectangle for the data """
        xmin, ymin = np.min(data, axis=0)
        xmax, ymax = np.max(data, axis=0)
        return Polygon([
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
        ])

    def set_centroids(self):
        """
        Set the centroids attribute. Requires poly and grid_polys to be defined.
        """
        centroid_coords = lambda x: (x.x, x.y)
        self.centroids = self.data_class([centroid_coords(t.centroid) for t in self.grid_polys])

    def set_sample_units_predefined(self, grid, *args, **kwargs):
        """
        Set the ROC grid from an existing result.
        :param grid: Array of (xmin, ymin, xmax, ymax) giving grid square bounds
        :param args:
        :param kwargs:
        :return:
        """
        self.prediction_values = None
        self.side_length = None
        if not self.poly:
            # find minimal bounding rectangle
            self.poly = self.generate_bounding_poly(self.data)
        if isinstance(self.poly, Polygon):
            # shapely
            creation_func = shapely_rectangle_from_vertices
        elif HAS_GEODJANGO:
            # geodjango
            creation_func = geodjango_rectangle_from_vertices
        else:
            raise TypeError("Geodjango is not supported and the polygon is NOT shapely format.")
        self.sample_units = grid
        self.grid_polys = []
        self.full_grid_square = []
        for g in self.sample_units:
            t = creation_func(*g)
            if t.within(self.poly):
                self.full_grid_square.append(True)
            else:
                self.full_grid_square.append(False)
            self.grid_polys.append(self.poly.intersection(t))
        self.set_centroids()
        self.set_sample_points(*args, **kwargs)

    def set_sample_units(self, side_length, *args, **kwargs):
        '''
        Set the ROC grid.
        :param length_or_arr: Either a scalar, interpreted as the side length of the grid square, OR an array of
          shapely Polygon or shapely MultiPolygon objects
        :param args: Passed to set_sample_points
        :param kwargs: Passed to set_sample_points
        :return: None
        '''
        # reset prediction values
        self.prediction_values = None
        if not self.poly:
            # find minimal bounding rectangle
            self.poly = self.generate_bounding_poly(self.data)
        self.side_length = side_length
        self.grid_polys, self.sample_units, self.full_grid_square = create_spatial_grid(self.poly, self.side_length)
        self.set_centroids()
        self.set_sample_points(*args, **kwargs)

    @property
    def sample_unit_size(self):
        return np.array([t.area for t in self.grid_polys])

    def copy_sample_units(self, roc):
        super(RocGrid, self).copy_sample_units(roc)
        self.side_length = roc.side_length
        self.grid_polys = copy(roc.grid_polys)
        self.full_grid_square = copy(roc.full_grid_square)

    def set_sample_points(self, *args, **kwargs):
        # sample points here are just the centroids
        self.sample_points = self.data_class(self.centroids)
        self.n_sample_point_per_unit = np.ones(self.n_sample_units)

    def in_sample_unit(self, sample_unit):
        xmin, ymin, xmax, ymax = sample_unit
        return (
            (self.data.toarray(0) >= xmin)
            & (self.data.toarray(0) < xmax)
            & (self.data.toarray(1) >= ymin)
            & (self.data.toarray(1) < ymax)
        )

    def plot(self,
             show_sample_units=True,
             show_sample_points=False,
             show_prediction=True,
             fmax=0.9,
             cmap='Reds',
             ax=None,
             **kwargs
    ):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
        plot_shapely_geos(self.poly, facecolor='none', edgecolor='k', ax=ax)

        if show_prediction:
            # create dictionary of segment colours for plotting
            # this requires creating a norm instance and using that to index a colourmap
            cmapper = colour_mapper(self.prediction_values, fmax=fmax, vmin=0)
            for pv, grid in zip(self.prediction_values, self.sample_units):
                sq = shapely_rectangle_from_vertices(*grid)
                plot_shapely_geos(sq, ax=ax, facecolor=cmapper.to_rgba(pv), edgecolor='none',
                                  alpha=kwargs.pop('alpha', 0.4))

        if show_sample_units:
            plot_shapely_geos([shapely_rectangle_from_vertices(*grid) for grid in self.sample_units],
                              ax=ax,
                              facecolor='none')

        if show_sample_points:
            plt.scatter(*self.sample_points.separate, s=10, c='k', marker='o')

        # remove x and y ticks as these rarely add anything
        ax.set_xticks([])
        ax.set_yticks([])
        plt.draw()


class RocGridMean(RocGrid):

    def set_sample_points(self, n_sample_per_grid, respect_boundary=True, *args, **kwargs):
        """ Generate n_sample_per_grid sample points per grid unit
         Return n_sample_per_grid x self.ndata x 2 array, final dim is x, y """

        if HAS_GEODJANGO and isinstance(self.poly, geos.GEOSGeometry):
            point_class = geos.Point
        else:
            point_class = Point

        if self.side_length is None:
            # grid was supplied as an array
            # slow version: need to iterate over the polygons
            xres = np.empty((n_sample_per_grid, self.n_sample_units), dtype=float)
            yres = np.empty((n_sample_per_grid, self.n_sample_units), dtype=float)
            for i, p in enumerate(self.grid_polys):
                xres[:, i], yres[:, i] = random_points_within_poly(p, n_sample_per_grid)

        else:
            xmins = np.array([x[0] for x in self.sample_units])
            ymins = np.array([x[1] for x in self.sample_units])
            xres = np.random.rand(n_sample_per_grid, self.n_sample_units) * self.side_length + xmins
            yres = np.random.rand(n_sample_per_grid, self.n_sample_units) * self.side_length + ymins

        if respect_boundary:
            # loop over grid squares that are incomplete
            for i in np.where(np.array(self.full_grid_square) == False)[0]:
                inside_idx = np.array([point_class(x, y).within(self.poly) for x, y in zip(xres[:, i], yres[:, i])])
                # pad empty parts with repeats of the centroid location
                num_empty = n_sample_per_grid - sum(inside_idx)
                if num_empty:
                    cx, cy = self.centroids[i]
                    rem_x = np.concatenate((xres[inside_idx, i], cx * np.ones(num_empty)))
                    rem_y = np.concatenate((yres[inside_idx, i], cy * np.ones(num_empty)))
                    xres[:, i] = rem_x
                    yres[:, i] = rem_y

        xres = xres.flatten(order='F')
        yres = yres.flatten(order='F')
        self.sample_points = DataArray.from_args(xres, yres)

        self.n_sample_point_per_unit = np.ones(self.n_sample_units) * n_sample_per_grid


class RocGridTimeWeighted(RocGrid):
    """
    UNFINISHED class. Key idea: rather than selecting future events in a binary fashion based on their time of
    occurrence, we could take ALL future events, but weight the more recent ones more highly.
    This sort of avoids the need to pick an advance time window (though of course you then need a decay constant
    instead.
    """
    def __init__(self, data=None, poly=None, half_life=1.):
        super(RocGridTimeWeighted, self).__init__(data=data, poly=poly)
        self.decay_constant = None
        self.set_decay_constant(half_life)

    def set_decay_constant(self, half_life):
        self.decay_constant = np.log(2.) / half_life

    def set_data(self, data, index=None):
        # Data is a 3D array, cols: t, x, y
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
        for t in self.sample_units:
            idx = self.in_sample_unit(t)
            n.append(sum(self.weights[idx]))
        return np.array(n)

    def evaluate(self):

        # count actual crimes in testing dataset on same grid
        true_counts = self.true_count[self.prediction_rank]
        true_counts_sorted = np.sort(self.true_count)[::-1]
        pred_values = self.prediction_values[self.prediction_rank]
        area = self.sample_unit_size[self.prediction_rank]
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


class NetworkRocSegments(SpatialRoc):
    """
    Network-based spatial ROC class. The sample unit is a street segment, equivalent to an edge in the network.
    Sample points are simply the path centroid of each segment.
    """
    data_class = NetworkData

    def __init__(self,
                 data=None,
                 graph=None,
                 poly=None,
                 data_index=None,
                 **kwargs):
        self.graph = graph
        super(NetworkRocSegments, self).__init__(
            data=data,
            poly=poly,
            data_index=data_index,
            **kwargs
        )

    def set_data(self, data, index=None):
        super(NetworkRocSegments, self).set_data(data, index=index)
        if self.graph is None and data is not None:
            self.graph = self.data.graph

    @property
    def data(self):
        if self._data is not None:
            return self._data
        raise AttributeError("Data have not been set yet, call set_data")

    @property
    def cartesian_data(self):
        return self.data.to_cartesian()

    def plot(self,
             show_sample_units=False,
             show_prediction=True,
             fmax=0.9,
             cmap='Reds',
             **kwargs
    ):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')

        if show_prediction:
            # create dictionary of segment colours for plotting
            # this requires creating a norm instance and using that to index a colourmap
            vmax = sorted(self.prediction_values)[int(self.n_sample_units * fmax)]
            cmap = cm.get_cmap(cmap)
            norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
            colour_mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            edge_inner_col = {}
            for pv, edge in zip(self.prediction_values, self.sample_units):
                edge_inner_col[edge['fid']] = colour_mapper.to_rgba(pv)
            self.graph.plot_network(ax=ax, edge_width=7, edge_inner_col=edge_inner_col)
        else:
            # plot standard network edge outlines without colour
            self.graph.plot_network()


        if show_sample_units:
            # TODO: this might not be the best way, but going to illustrate sampling with crosses
            plt.plot(self.sample_points.to_cartesian().toarray(0),
                     self.sample_points.to_cartesian().toarray(1),
                     'kx',
                     markersize=10,
                     markeredgewidth=3)

        # remove x and y ticks as these rarely add anything
        ax.set_xticks([])
        ax.set_yticks([])

    @staticmethod
    def generate_bounding_poly(data):
        # called when no polygon is provided, computes the bounding polygon for the data based on the network
        # TODO: add this method to the StreetNet class?
        raise NotImplementedError()

    @property
    def sample_unit_size(self):
        """ Return the size of each sample unit, e.g. length or area. """
        return np.array([t['length'] for t in self.sample_units])

    # segment-specific
    def set_sample_units(self, length_or_arr, *args, **kwargs):
        '''
        Set the ROC grid.
        :param length_or_arr: ignored here(?)
        :param args: Passed to set_sample_points
        :param kwargs: Passed to set_sample_points
        :return: None
        '''
        # reset prediction values
        self.prediction_values = None

        self.sample_units = self.graph.edges()
        self.centroids = self.data_class([t.centroid for t in self.sample_units])  # array
        self.set_sample_points(*args, **kwargs)

    def copy_sample_units(self, roc):
        # reset prediction values
        self.prediction_values = None
        self.sample_units = copy(roc.sample_units)
        self.sample_points = copy(roc.sample_points)
        self.n_sample_point_per_unit = copy(roc.n_sample_point_per_unit)

    def set_sample_points(self, *args, **kwargs):
        # sample points here are just the centroids
        self.sample_points = self.data_class(self.centroids)
        self.n_sample_point_per_unit = np.ones(self.n_sample_units)

    def in_sample_unit(self, sample_unit):
        """ Return bool array, one entry per data point. True indicates that the datum is within the sample unit. """
        return np.array([t.edge == sample_unit for t in self.data.toarray(0)])


class RocGridByNetworkLengthMean(RocGridMean):
    """
    Planar ROC class used for comparison with network predictions. This class generates a grid AND requires a reference
    to a network object. Predictions are carried out just as in the standard RocGrid fashion, BUT the cumulative_area
    becomes cumulative length of network segments within the grid squares.
    """
    data_class = CartesianData

    def __init__(self,
                 data=None,
                 poly=None,
                 data_index=None,
                 graph=None,
                 **kwargs):
        if graph is None:
            raise AttributeError("Must specify a graph object.")
        self.graph = graph
        self.network_length_in_grid = None
        super(RocGrid, self).__init__(data=data,
                                      poly=poly,
                                      data_index=data_index,
                                      **kwargs)
        self.grid_polys = None
        self.full_grid_square = None
        self.side_length = None

    def set_sample_units(self, side_length, *args, **kwargs):
        """
        Set the ROC grid.
        :param side_length: side length of grid squares
        :param args: Passed to set_sample_points
        :param kwargs: Passed to set_sample_points.
        :return: None
        """
        # define the grid as usual
        super(RocGridByNetworkLengthMean, self).set_sample_units(side_length, *args, **kwargs)

        # compute intersection with net
        self.set_network_grid_intersection_lengths()

    def set_network_grid_intersection_lengths(self):        
        logger.debug("set_network_grid_intersection_lengths called")
        if RTREE:
            # segment the network into edges intersecting grid squares
            lines = list(self.graph.lines_iter())
            # create an index ahead of time for fast lookups
            geo_idx = index.Index()
            logger.debug("Creating index")
            for i, l in enumerate(lines):
                geo_idx.insert(i, l.bounds)
            self.network_length_in_grid = []
            logger.debug("Starting iteration over %d sample units", self.n_sample_units)
            tic = time()
            for i, t in enumerate(self.sample_units):
                sq = shapely_rectangle_from_vertices(*t)
                length_tally = 0.
                for j in geo_idx.intersection(sq.bounds):
                    length_tally += sq.intersection(lines[j]).length
                self.network_length_in_grid.append(length_tally)
            logger.debug("Complete in %f s", time() - tic)
            self.network_length_in_grid = np.array(self.network_length_in_grid)
        else:
            # this older version is SLOW. Avoid if at all possible
            # segment the network into edges intersecting grid squares
            lines = list(self.graph.lines_iter())
            self.network_length_in_grid = []
            for t in self.sample_units:
                sq = shapely_rectangle_from_vertices(*t)
                this_int = [sq.intersection(t) for t in lines if sq.intersects(t)]
                this_length = sum([t.length for t in this_int])
                self.network_length_in_grid.append(this_length)
            self.network_length_in_grid = np.array(self.network_length_in_grid)        

    @property
    def sample_unit_size(self):
        return self.network_length_in_grid


class NetworkRocSegmentsMean(NetworkRocSegments):
    """ As for NetworkRocSegments, but sampling at multiple locations per segment to get a better estimate """

    def set_sample_points(self, dl, *args, **kwargs):
        """
        Set the sample points in each segment.
        :param dl: Distance between sample points. If the unit length is less than this, it is sampled at the centroid.
        The sample points aren't guaranteed to be this far apart.
        :param args:
        :param kwargs:
        :return:
        """
        sp = []
        n = []
        for edge in self.sample_units:
            # generate some distance along values and convert into NetPoints
            da = np.linspace(0., edge['length'], np.ceil(edge['length'] / dl) + 2)[1:-1]
            this_sp = []
            for d in da:
                node_dist = {
                    edge.orientation_neg: d,
                    edge.orientation_pos: edge['length'] - d,
                }
                this_sp.append(NetPoint(self.graph, edge, node_dist))
            sp.extend(this_sp)
            n.append(len(this_sp))

        self.sample_points = self.data_class(sp)
        self.n_sample_point_per_unit = np.array(n)



class NetworkRocUniformSamplingGrid(NetworkRocSegments):
    """
    Place sample points uniformly over the network, then divide into sample units by imposing a regular grid
    """
    @staticmethod
    def generate_bounding_poly(data):
        return RocGrid.generate_bounding_poly(data.to_cartesian())

    def set_sample_units(self, side_length, interval, *args, **kwargs):
        """
        Set the ROC grid.
        :param side_length: side length of grid squares
        :param interval: The length interval between sample points
        :param args: Passed to set_sample_points
        :param kwargs: Passed to set_sample_points.
        :return: None
        """
        # reset prediction values
        self.prediction_values = None

        if not self.poly:
            # find minimal bounding rectangle
            self.poly = self.generate_bounding_poly(self.data)

        # set sample grid
        self.side_length = side_length
        self.grid_polys, self.sample_units, _ = create_spatial_grid(self.poly, self.side_length)

        # set network sampling points
        self.set_sample_points(interval, *args, **kwargs)

    def set_sample_points(self, interval, *args, **kwargs):
        # get sample points
        # these are initially in the same order as the network edges
        sample_points, _ = network_walker_uniform_sample_points(self.graph, interval)

        # divide into sample units and reorder the sample points
        # will also need to redefine the sample units and grid polys at the end
        sample_units = []
        grid_polys = []
        reordered_sample_points = []
        self.n_sample_point_per_unit = []
        x, y = sample_points.to_cartesian().separate

        for i in range(len(self.sample_units)):
            xmin, ymin, xmax, ymax = self.sample_units[i]
            in_su = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
            n = sum(in_su)
            if n:
                reordered_sample_points.extend(sample_points.getrows(in_su).toarray(0))
                self.n_sample_point_per_unit.append(n)
                sample_units.append((xmin, ymin, xmax, ymax))
                grid_polys.append(self.grid_polys[i])

        self.n_sample_point_per_unit = np.array(self.n_sample_point_per_unit)
        self.sample_points = NetworkData(reordered_sample_points)
        self.sample_units = sample_units
        self.grid_polys = grid_polys

    def plot(self,
             show_sample_units=True,
             show_prediction=True,
             fmax=0.9,
             cmap='Reds',
             **kwargs):
        from matplotlib import patches

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')

        if show_prediction:
            # create dictionary of segment colours for plotting
            # this requires creating a norm instance and using that to index a colourmap
            vmax = sorted(self.prediction_values)[int(self.n_sample_units * fmax)]
            cmap = cm.get_cmap(cmap)
            norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
            colour_mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            edge_inner_col = {}
            for pv, edge in zip(self.prediction_values, self.sample_units):
                edge_inner_col[edge['fid']] = colour_mapper.to_rgba(pv)
            self.graph.plot_network(ax=ax, edge_width=7, edge_inner_col=edge_inner_col)
        else:
            # plot standard network edge outlines without colour
            self.graph.plot_network(edge_width=10, edge_inner_col='w')

        if show_sample_units:
            # alternating grey - black grid squares / crosses
            xsp, ysp = self.sample_points.to_cartesian().separate
            mins = np.array(self.sample_units).min(axis=0)
            maxs = np.array(self.sample_units).max(axis=0)
            xmin_group = np.arange(mins[0], maxs[2], 2 * self.side_length)
            ymin_group = np.arange(mins[1], maxs[3], 2 * self.side_length)

            count = 0
            for gp, su, n in zip(self.grid_polys, self.sample_units, self.n_sample_point_per_unit):
                a = np.any(np.abs(xmin_group - su[0]) < 1e-3) ^ np.any(np.abs(ymin_group - su[1]) < 1e-3)
                fc = np.ones(3) * (0.5 if a else 0.8)
                mc = np.ones(3) * 0.5 if a else 'k'
                plot_shapely_geos(gp, facecolor=fc, alpha=0.4)
                plt.plot(
                    xsp[count:count + n],
                    ysp[count:count + n],
                    'o',
                    color=mc,
                    markersize=5,
                )
                count += n

        # remove x and y ticks as these rarely add anything
        ax.set_xticks([])
        ax.set_yticks([])
