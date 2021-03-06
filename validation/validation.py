
__author__ = 'gabriel'
from shapely.geometry import Point
import numpy as np
import math
import hotspot
import roc
import collections
from data.models import CartesianSpaceTimeData, NetworkSpaceTimeData, DataArray
import logging


logger = logging.getLogger('validation.validation')
# default: output all logs to console
ch = logging.StreamHandler()
logger.setLevel(logging.DEBUG)
logger.addHandler(ch)

def mc_sampler(poly):
    # poly is a shapely polygon
    x_min, y_min, x_max, y_max = poly.bounds
    while True:
        x = np.random.random() * (x_max - x_min) + x_min
        y = np.random.random() * (y_max - y_min) + y_min
        if poly.intersects(Point(x, y)):
            yield (x, y)


class ValidationBase(object):

    roc_class = roc.RocGrid
    data_class = CartesianSpaceTimeData

    def __init__(self, data, model,
                 data_index=None,
                 spatial_domain=None,
                 sample_unit_size=None,
                 cutoff_t=None,
                 include_predictions=False):

        self.include_predictions = include_predictions

        self.data = None
        self.data_index = None
        self.set_data(data, data_index=data_index)
        self.model = model

        # set initial time cut point
        self.cutoff_t = cutoff_t or self.t[int(self.ndata / 2)]

        # set roc
        self.bounding_poly = spatial_domain
        self.roc = None
        self.set_roc()

        # setup grid if sample_unit_size supplied
        if sample_unit_size:
            self.set_sample_units(sample_unit_size)

    def set_data(self, data, data_index=None):
        # sort data in increasing time
        self.data = self.data_class(data)
        sort_idx = np.argsort(self.data.time.toarray(0))
        self.data = self.data.getrows(sort_idx)
        self.data_index = data_index
        if data_index is not None:
            self.data_index = np.array(data_index)
            self.data_index = self.data_index[sort_idx]

    def set_roc(self):

        # set roc
        self.roc = self.roc_class(poly=self.bounding_poly)
        # set roc with ALL data initially so it can compute the boundary if no poly has been provided
        self.roc.set_data(self.data[:, 1:], index=self.data_index)


    def set_t_cutoff(self, cutoff_t, b_train=True, **kwargs):
        """
        Set cutoff time that divides dataset into training and testing portions.
        :param cutoff_t: New value for cutoff time.
        :param b_train: Boolean indicating whether the model should be (re)trained after setting the new cutoff
        :param kwargs: kwargs to pass to training function.
        """
        self.cutoff_t = cutoff_t
        if b_train:
            self.train_model(**kwargs)

    def set_sample_units(self, length_or_roc, *args, **kwargs):
        """
        Set the domain grid for computing SER etc.
        :param grid: Either a scalar giving the grid square length or an instance of RocSpatialGrid from which the grid
        will be copied
        """
        if isinstance(length_or_roc, self.roc_class):
            self.roc.copy_sample_units(length_or_roc)
        else:
            self.roc.set_sample_units(length_or_roc, *args, **kwargs)

    @property
    def spatial_domain(self):
        return self.roc.poly

    @property
    def sample_points(self):
        return self.roc.sample_points

    @property
    def ndata(self):
        return self.data.ndata

    @property
    def t(self):
        return self.data.toarray(0)

    @property
    def training(self):
        # return self.data.getrows(self.t <= self.cutoff_t)
        return self.data.getrows(self.t < self.cutoff_t)

    def testing_index(self, dt_plus=None, dt_minus=0.):
        """
        Return the array indices corresponding to the testing data based on the supplied parameters.
        :param dt_plus: Number of time units ahead of cutoff_t to take as maximum testing data.  If None, take ALL data.
        :param dt_minus: Number of time units ahead of cutoff_t to take as minimum testing data.  Defaults to 0.
        :return: Indices corresponding to the testing data, so that testing data can be extracted as
                 self.data[indices]
        """
        assert dt_minus >= 0., "dt_minus must be positive"
        bottom = self.cutoff_t + dt_minus
        if dt_plus:
            assert dt_plus >= 0., "dt_plus must be positive"
            # ind = (self.t > bottom) & (self.t <= (self.cutoff_t + dt_plus))
            ind = (self.t >= bottom) & (self.t < (self.cutoff_t + dt_plus))
        else:
            # ind = self.t > bottom
            ind = self.t >= bottom

        return np.where(ind)[0]

    def testing(self, dt_plus=None, dt_minus=0., as_point=False):
        """
        :param dt_plus: Number of time units ahead of cutoff_t to take as maximum testing data.  If None, take ALL data.
        :param dt_minus: Number of time units ahead of cutoff_t to take as minimum testing data.  Defaults to 0.
        :param as_point: If True, return N length list of (time, Point) tuples, else return N x 3 matrix
        :return: Testing data for comparison with predictions, based on value of cutoff_t.
        """
        ind = self.testing_index(dt_plus, dt_minus)
        d = self.data.getrows(ind)
        if as_point:
            return [(t[0], Point(list(t[1:]))) for t in d]
        else:
            return d

    def testing_data_index(self, dt_plus=None, dt_minus=0., as_point=False):
        """
        Parameters as per testing_index.
        :return: The data indices attached to the testing data.  If self.data_index has not been set, return the lookup
        indices instead.
        """
        ind = self.testing_index(dt_plus, dt_minus)
        if self.data_index is not None:
            return self.data_index[ind]
        else:
            return ind

    def train_model(self, *args, **kwargs):
        """ Train the predictor on training data """
        self.model.train(self.training, *args, **kwargs)

    def prediction_array(self, t):
        n = self.roc.sample_points.ndata
        ts = np.ones(n) * t
        data_array = DataArray.from_args(ts).adddim(self.roc.sample_points, type=self.data_class)
        return data_array

    def predict(self, t, **kwargs):
        # estimate total propensity in each grid poly
        return self.model.predict(t, self.sample_points)

    def run_roc_prediction(self, t, testing_data, testing_ind, **kwargs):
        # run prediction
        # output should be M x ndata matrix, where M is the number of sample points per grid square
        prediction = self.predict(t, **kwargs)
        self.roc.set_data(testing_data.space, index=testing_ind)
        self.roc.set_prediction(prediction)
        return self.roc.evaluate(include_predictions=self.include_predictions)

    def _iterate_run(self, pred_dt_plus, true_dt_plus, true_dt_minus, **kwargs):
        true_dt_plus = true_dt_plus or pred_dt_plus

        testing_data = self.testing(dt_plus=true_dt_plus, dt_minus=true_dt_minus)
        testing_ind = self.testing_data_index(dt_plus=true_dt_plus, dt_minus=true_dt_minus)
        prediction_t = self.cutoff_t + pred_dt_plus
        print "Cutoff time %d, I found %d datapoints" % (self.cutoff_t, len(testing_data))

        res = {}

        # ROC
        roc_evaluation = self.run_roc_prediction(prediction_t, testing_data, testing_ind, **kwargs)
        res.update(roc_evaluation)

        # conditional likelihood
        # compute the intensity at the test points
        # skip if no data
        if testing_data.ndata == 0:
            res['conditional_likelihood'] = np.nan
        else:
            # adding a small eps here to avoid log(0), but is this justifiable?
            intensity_at_test_points = self.model.predict(prediction_t, testing_data.space) + 1e-12
            res['conditional_likelihood'] = sum(np.log(intensity_at_test_points))

        return res

    def run(self, time_step,
            pred_dt_plus=None,
            true_dt_plus=None,
            true_dt_minus=0,
            t_upper=None,
            n_iter=None,
            pred_kwargs=None,
            train_kwargs=None, **kwargs):
        """
        Run the required train / predict / assess sequence
        Take the mean of the metrics returned
        :param time_step: Time step to use in each successive train-predict-assess cycle
        :param pred_dt_plus: Time units ahead of cutoff_t to use when computing the prediction
        :param true_dt_plus: Time units ahead of cutoff_t to take as the maximum test data, defaults to same value as
        pred_dt_plus
        :param true_dt_minus: Time units ahead of cutoff_t to take as the minimum test data, defaults to 0
        :param t_upper: Maximum time to use in data.  Upon reaching this, the run ceases.
        :param pred_kwargs: kwargs passed to prediction function
        :param train_kwargs: kwargs passed to train function
        :param kwargs: kwargs for the run function itself
        :return: results dictionary.
        """
        assert self.sample_points is not None, "ROC sample units not set. Run set_sample_units first."

        if bool(t_upper) and bool(n_iter):
            logger.exception("Both t_upper AND n_iter were supplied, but only one is supported")
            raise AttributeError("Both t_upper AND n_iter were supplied, but only one is supported")

        verbose = kwargs.pop('verbose', True)
        pred_kwargs = pred_kwargs or {}
        train_kwargs = train_kwargs or {}
        pred_dt_plus = pred_dt_plus or time_step
        true_dt_plus = true_dt_plus or pred_dt_plus

        if t_upper:
            n_iter = math.ceil((t_upper - self.cutoff_t) / time_step)
        elif n_iter:
            # check that this number of iterations is possible
            if (self.cutoff_t + (n_iter - 1) * time_step + true_dt_minus) > self.data[-1, 0]:
                logger.exception("The requested number of iterations is too great for the supplied data.")
                raise AttributeError("The requested number of iterations is too great for the supplied data.")
        else:
            n_iter = math.ceil((self.data[-1, 0] - self.cutoff_t) / time_step)

        res = collections.defaultdict(list)

        # store some attributes that will make repeating easier
        res['pred_dt_plus'] = pred_dt_plus
        res['true_dt_plus'] = true_dt_plus
        res['true_dt_minus'] = true_dt_minus


        if verbose:
            logger.info("Running %d validation iterations..." % n_iter)

        count = 0

        try:
            while count < n_iter:
                if count == 0:
                    # initial training
                    self._initial_setup(**train_kwargs)
                else:
                    # update model as required and update the cutoff time
                    self._update(time_step, **train_kwargs)

                if verbose:
                    logger.info("Running validation with cutoff time %s (iteration %d / %d)" % (str(self.cutoff_t),
                                                                                                count + 1,
                                                                                                n_iter))

                # predict and assess iteration
                this_res = self._iterate_run(pred_dt_plus, true_dt_plus, true_dt_minus, **pred_kwargs)
                for k, v in this_res.items():
                    res[k].append(v)

                # add record of cutoff time to help later repeats
                res['cutoff_t'].append(self.cutoff_t)

                count += 1

        except KeyboardInterrupt:
            # this breaks the loop, now return res as it stands
            pass

        self.post_process(res)
        return res

    def post_process(self, res):
        """
        Called at the very end of the run method with the output.  This is the place to make any final changes if
        required
        :param res: output from run method just before this function gets called
        :return: None - modify in place
        """
        # some lists are better converted to numpy arrays
        convert_to_arr = (
            'prediction_rank',
            'cumulative_area',
            'prediction_values',
            'cumulative_crime',
            'cumulative_crime_count',
            'cumulative_crime_max',
            'pai'
        )
        for k in convert_to_arr:
            if k in res:
                # this allows for optional components such as prediction values
                res[k] = np.array(res[k])

    def _update(self, time_step, **train_kwargs):
        """
        This function gets called after a successful train-predict-assess cycle.
        It is used to setup and TRAIN the model for the next cycle.
        It should ALWAYS update the cutoff time
        """
        # training is implied here
        self.set_t_cutoff(self.cutoff_t + time_step, **train_kwargs)

    def _initial_setup(self, **train_kwargs):
        """
        Called instead of _update on the first iteration of a run.
        Since this is called *before* the predict step, we don't advance the time on the first iteration
        :param train_kwargs:
        :return:
        """
        self._update(time_step=0., **train_kwargs)

    def repeat_run(self, run_res, pred_dt_plus=None, true_dt_plus=None, true_dt_minus=None,
                   pred_kwargs=None, **kwargs):
        """
        Repeat a validation run, but rather than training the model use supplied output of a previous run to save time.
        Parameters are as for run(), but some are unnecessary.
        Cutoff times and trained models are extracted from res.
        """
        verbose = kwargs.pop('verbose', True)
        pred_kwargs = pred_kwargs or {}
        pred_dt_plus = pred_dt_plus or run_res['pred_dt_plus']
        true_dt_plus = true_dt_plus or run_res['true_dt_plus']
        true_dt_minus = true_dt_minus or run_res['true_dt_minus']

        # number of iterations
        n_iter = len(run_res['cutoff_t'])

        res = collections.defaultdict(list)
        res['pred_dt_plus'] = pred_dt_plus
        res['true_dt_plus'] = true_dt_plus
        res['true_dt_minus'] = true_dt_minus

        if verbose:
            logger.info("Running %d repeat validation iterations..." % n_iter)

        count = 0

        try:
            while count < n_iter:

                this_cutoff_t = run_res['cutoff_t'][count]
                self.set_t_cutoff(this_cutoff_t, b_train=False)  # no need to train

                if verbose:
                    logger.info("Running repeat validation with cutoff time %s (iteration %d / %d)" % (str(self.cutoff_t),
                                                                                                       count + 1,
                                                                                                       n_iter))

                # copy model
                self.model = run_res['model'][count]

                # predict and assess iteration
                this_res = self._iterate_run(pred_dt_plus, true_dt_plus, true_dt_minus, **pred_kwargs)
                for k, v in this_res.items():
                    res[k].append(v)

                # add record of cutoff time to help later repeats
                res['cutoff_t'].append(self.cutoff_t)

                count += 1

        except KeyboardInterrupt:
            # this breaks the loop, now return res as it stands
            pass

        self.post_process(res)
        return res


class WeightedValidation(ValidationBase):
    ## TODO: implement WeightedRocSpatialGrid - should be a fairly trivial extension
    # -> include ALL test data
    # -> BUT modify the time column to ensure that self.cutoff_t corresponds to 0
    # -> only real change is in def testing()?
    pass



class ValidationIntegration(ValidationBase):

    roc_class = roc.RocGridMean


class NetworkValidationBase(ValidationBase):

    roc_class = roc.NetworkRocSegments
    data_class = NetworkSpaceTimeData

    def __init__(self, *args, **kwargs):
        # declare extra attributes that are specific to the network class
        self.graph = kwargs.pop('graph', None)
        super(NetworkValidationBase, self).__init__(*args, **kwargs)

    def set_data(self, data, data_index=None):
        super(NetworkValidationBase, self).set_data(data, data_index=data_index)
        # add the graph attribute
        self.graph = data.graph

    def _initial_setup(self, **train_kwargs):
        """
        On first run, set the NetworkWalker that will be used for all subsequent network path finding?
		Or is it going in the KDE class?
        """
        super(NetworkValidationBase, self)._initial_setup(**train_kwargs)


class ValidationIntegrationByNetworkSegment(NetworkValidationBase):

    roc_class = roc.RocGridByNetworkLengthMean
    data_class = CartesianSpaceTimeData

    def __init__(self, data, model,
                 data_index=None,
                 graph=None,
                 *args, **kwargs):
        if graph is None:
            raise AttributeError('Must specify the kwarg graph at instantiation.')
        super(ValidationIntegrationByNetworkSegment, self).__init__(data,
                                                                    model,
                                                                    data_index=data_index,
                                                                    graph=graph,
                                                                    *args, **kwargs)
        # self.graph = graph

    def set_data(self, data, data_index=None):
        ValidationBase.set_data(self, data, data_index=data_index)

    def set_roc(self):
        """
        The graph instance must be supplied at instantiation to ROC
        """
        # set roc
        self.roc = self.roc_class(poly=self.bounding_poly, graph=self.graph)
        # set roc with ALL data initially
        self.roc.set_data(self.data[:, 1:], index=self.data_index)



class NetworkValidationMean(NetworkValidationBase):

    roc_class = roc.NetworkRocSegmentsMean


if __name__ == "__main__":
    from database import models
    from plotting import spatial
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    camden = models.Division.objects.get(name='Camden')
    xm = 526500
    ym = 186000
    nd = 1000
    # nice normal data
    # data = np.hstack((np.random.normal(loc=5, scale=5, size=(nd, 1)),
    #                   multivariate_normal.rvs(mean=[xm, ym], cov=np.diag([1e5, 1e5]), size=(nd, 1))))

    # moving cluster
    data = np.hstack((
        np.linspace(0, 10, nd).reshape((nd, 1)),
        xm + np.linspace(0, 5000, nd).reshape((nd, 1)) + np.random.normal(loc=0, scale=100, size=(nd, 1)),
        ym + np.linspace(0, -4000, nd).reshape((nd, 1)) + np.random.normal(loc=0, scale=100, size=(nd, 1)),
    ))

    # use Bowers kernel
    # stk = hotspot.STKernelBowers(1, 1e-4)
    # vb = ValidationBase(data, hotspot.Hotspot, camden.mpoly, model_args=(stk,))
    # vb.set_sample_units(200)
    # vb.set_t_cutoff(4.0)

    # use basic historic data spatial hotspot
    sk = hotspot.SKernelHistoric(2) # use heatmap from final 2 days data
    # vb = ValidationBase(data, hotspot.Hotspot, camden.mpoly, model_args=(sk,))
    # vb.set_sample_units(200)
    vb = ValidationIntegration(data, sk, spatial_domain=camden.mpoly, include_predictions=True)
    vb.set_sample_units(200, n_sample_per_grid=10)
    vb.set_t_cutoff(4.0)

    res = vb.run(time_step=1, n_iter=5) # predict one day ahead
    pred_values = res['prediction_values'][-1]
    polys_pred_rank_order = [vb.roc.grid_polys[i] for i in res['prediction_rank'][-1]]

    norm = mpl.colors.Normalize(min(pred_values), max(pred_values))
    cmap = mpl.cm.jet
    sm = mpl.cm.ScalarMappable(norm, cmap)

    # Figure: surface showing prediction values by grid square
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (p, r) in zip(polys_pred_rank_order, sorted(pred_values, reverse=True)):
        # spatial.plot_geodjango_shapes(shapes=(p,), ax=ax, facecolor=sm.to_rgba(r), set_axes=False)
        try:
            spatial.plot_shapely_geos(shapes=p, ax=ax, facecolor=sm.to_rgba(r))
        except Exception:
            pass
    spatial.plot_geodjango_shapes((vb.spatial_domain,), ax=ax, facecolor='none')

    # Figure: surface showing true values by grid square

    norm = mpl.colors.Normalize(min(vb.roc.true_count), max(vb.roc.true_count))
    sm = mpl.cm.ScalarMappable(norm, cmap)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (p, r) in zip(vb.roc.grid_polys, vb.roc.true_count):
        # spatial.plot_geodjango_shapes(shapes=(p,), ax=ax, facecolor=sm.to_rgba(r), set_axes=False)
        try:
            spatial.plot_shapely_geos(shapes=p, ax=ax, facecolor=sm.to_rgba(r))
        except Exception:
            pass
    spatial.plot_geodjango_shapes((vb.spatial_domain,), ax=ax, facecolor='none')

