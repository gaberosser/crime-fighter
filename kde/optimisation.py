__author__ = 'gabriel'
import models
import netmodels
import numpy as np
import multiprocessing as mp
import datetime
from functools import partial
import contextlib
import logging
from time import time

def _log_likelihood_fixed_wrapper(args):
    return _compute_log_likelihood_fixed(*args)


def _compute_log_likelihood_fixed(training, testing, bandwidths):
    k = models.FixedBandwidthKde(training, bandwidths=bandwidths, parallel=False)
    z = k.pdf(testing)
    return np.sum(np.log(z))


def _log_likelihood_variable_wrapper(args):
    return _compute_log_likelihood_variable(*args)


def _compute_log_likelihood_variable(training, testing, nn):
    k = models.SpaceTimeVariableBandwidthNnTimeReflected(training, number_nn=nn, parallel=False, strict=False)
    z = k.pdf(testing)
    return np.sum(np.log(z))


def _log_likelihood_fixed_network_wrapper(args):
    return _compute_log_likelihood_fixed_network(*args)


def _compute_log_likelihood_fixed_network(training, testing, bandwidths):
    k = netmodels.NetworkTemporalKde(training,
                                     bandwidths=bandwidths,
                                     cutoffs=[bandwidths[0] * 7., bandwidths[1]])
    z = k.pdf(testing)
    return np.sum(np.log(z))


def plot_total_likelihood_surface(ss, tt, ll,
                                  ax=None,
                                  fmin=None,
                                  fmax=None,
                                  **kwargs):
    """
    Plot a surface showing the total log likelihood, evaluated over multiple windows
    :param ss: Spatial bandwidths
    :param tt: Temporal bandwidths
    :param ll: Log likelihoods
    :return:
    """
    from matplotlib import pyplot as plt
    fmin = fmin if fmin is not None else 0.25
    fmax = fmax if fmax is not None else 0.98
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ll_total = ll.sum(axis=0)
    # get caxis limits
    v = ll_total.flatten()
    v.sort()
    vmin = v[int(len(v) * fmin)]
    vmax = v[int(len(v) * fmax)]
    h = ax.pcolor(ss, tt, ll_total,
                  vmin=vmin, vmax=vmax, **kwargs)
    return h


def compute_log_likelihood_surface_fixed_bandwidth(data,
                                                   start_day,
                                                   niter,
                                                   min_d=10.,
                                                   max_d=500,
                                                   min_t=1.,
                                                   max_t=180.,
                                                   npts=20):

    # res = np.zeros((npts, npts + 1, niter))
    res = []
    tt, ss = np.meshgrid(
        np.linspace(min_t, max_t, npts),
        np.linspace(min_d, max_d, npts + 1),
    )

    try:
        for i in range(niter):
            training = data[data[:, 0] < (start_day + i)]
            testing_idx = (data[:, 0] >= (start_day + i)) & (data[:, 0] < (start_day + i + 1))
            testing = data[testing_idx]

            pool = mp.Pool()
            q = pool.map_async(
                _log_likelihood_fixed_wrapper,
                ((training, testing, (tt.flat[j], ss.flat[j], ss.flat[j])) for j in range(ss.size))
            )
            pool.close()
            this_res = q.get(1e100)
            res.append(np.array(this_res).reshape((npts + 1, npts)))
            # for j in range(ss.size):
                # res[:, :, i].flat[j] = this_res[j]
            print "Completed iteration %d / %d" % (i + 1, niter)

    except (Exception, KeyboardInterrupt) as exc:
        print "Terminating early due to Exception:"
        print repr(exc)

    res = np.array(res)
    return ss, tt, res


def compute_log_likelihood_surface_variable_bandwidth(data,
                                                      start_day,
                                                      niter,
                                                      min_nn=5,
                                                      max_nn=150,
                                                      npts=20):

    res = np.zeros((npts, niter))
    nn = np.linspace(min_nn, max_nn, npts)

    try:
        for i in range(niter):
            this_res = []
            training = data[data[:, 0] < (start_day + i)]
            testing_idx = (data[:, 0] >= (start_day + i)) & (data[:, 0] < (start_day + i + 1))
            testing = data[testing_idx]

            pool = mp.Pool()
            q = pool.map_async(
                _log_likelihood_variable_wrapper,
                ((training, testing, n) for n in nn)
            )
            pool.close()
            this_res = q.get(1e100)
            res[:, i] = np.array(this_res)

    except Exception as exc:
        print "Terminating early due to Exception:"
        print repr(exc)

    return nn, res


def compute_log_likelihood_surface_network_fixed_bandwidth(data,
                                                           start_day,
                                                           niter,
                                                           min_d=10.,
                                                           max_d=500,
                                                           min_t=1.,
                                                           max_t=180.,
                                                           npts=20):

    res = np.zeros((npts, npts, niter))
    ss, tt = np.meshgrid(
        np.linspace(min_d, max_d, npts),
        np.linspace(min_t, max_t, npts),
    )

    try:
        for i in range(niter):
            training = data.getrows(data.time.toarray() < (start_day + i))
            testing_idx = (data.time.toarray() >= (start_day + i)) & (data.time.toarray() < (start_day + i + 1))
            testing = data.getrows(testing_idx)

            pool = mp.Pool()
            q = pool.map_async(
                _log_likelihood_fixed_network_wrapper,
                ((training, testing, (t, s)) for (s, t) in zip(ss.flat, tt.flat))
            )
            pool.close()
            this_res = q.get(1e100)
            res[:, :, i] = np.array(this_res).reshape((npts, npts))
            print "Completed iteration %d / %d" % (i + 1, niter)

    except (Exception, KeyboardInterrupt) as exc:
        print "Terminating early due to Exception:"
        print repr(exc)

    return ss, tt, res


from data import iterator


class ForwardChainingValidationBase(object):

    ll_func = None
    MIN_N_PARAM = None
    MAX_N_PARAM = None

    def __init__(self,
                 data,
                 data_index=None,
                 initial_cutoff=None,
                 data_class=None,
                 parallel=True):

        self.roller = iterator.RollingOrigin(data,
                                             data_index=data_index,
                                             initial_cutoff_t=initial_cutoff,
                                             data_class=data_class)
        self.data = self.roller.data
        self.data_index = self.roller.data_index
        self.grid = None
        self.res_arr = None
        self.parallel = parallel
        self.nparam = None
        self.logger = None

    def set_logger(self, logger=None, verbose=True):
        if logger is None:
            if self.logger:
                self.logger = logging.getLogger(__name__)
                self.logger.handlers = []
            else:
                self.logger = logging.getLogger(__name__)
                self.logger.handlers = []
                if verbose:
                    self.logger.setLevel(logging.DEBUG)
                    ch = logging.StreamHandler()
                    ch.setLevel(logging.INFO)
                    self.logger.addHandler(ch)
        else:
            self.logger = logger


    def set_parameter_grid(self, npt=10, *args):
        """
        :param npt: The number of points to include in each dimension
        :param args: (xmin, xmax, ymin, ymax, ...)
        The parameter grid has (ndim + 1) dimensions, with the first axis being of length ndim.
        """
        assert len(args) % 2 == 0, "Number of args must be even (xmin, xmax, ymin, ymax, ...)"
        ndim = len(args) / 2
        if self.MAX_N_PARAM is not None and ndim > self.MAX_N_PARAM:
            raise AttributeError("Maximum number of permissible parameters is %d, but %d were supplied" % (
                self.MAX_N_PARAM, ndim
            ))
        if self.MIN_N_PARAM is not None and ndim < self.MIN_N_PARAM:
            raise AttributeError("Minimum number of permissible parameters is %d, but %d were supplied" % (
                self.MIN_N_PARAM, ndim
            ))
        x = []
        slices = []
        for i in range(ndim):
            x.append(np.linspace(args[2 * i], args[2 * i + 1], npt))
            slices.append(slice(0, npt))
        grid_idx = np.mgrid[slices]
        self.grid = np.array([x[i][grid_idx[i]] for i in range(ndim)])
        self.nparam = npt ** ndim if ndim else 0

    def initial_setup(self):
        """
        If any of the models need to be initialised / trained before the main run, do so here.
        :return:
        """
        self.res_arr = []

    def args_kwargs_generator(self, *args, **kwargs):
        """
        Prepare the *args and **kwargs that will be passed to the method along with training and testing data.
        :return: generator of tuples (args (iterable), kwargs (dict)), one per parameter combination
        """
        raise NotImplementedError()

    def run_one_timestep(self, training, testing, *args, **kwargs):
        raise NotImplementedError()

    def run(self, niter=None):
        self.initial_setup()
        count = 1
        try:
            for obj in self.roller.iterator(niter=niter):
                self.logger.info("Iteration %d", count)
                tic = time()
                self.res_arr.append(self.run_one_timestep(obj.training, obj.testing))
                self.logger.info("Completed in %f s", time() - tic)
                count += 1
        except KeyboardInterrupt:
            # stop wherever we got to
            count -= 1
            self.logger.info("Interrupted. Quitting gracefully.")
        self.logger.info("Stopping after %d iterations", count)


def kde_wrapper(training, testing, kde_class, args_kwargs):
    k = kde_class(training, *args_kwargs[0], **args_kwargs[1])
    z = k.pdf(testing)
    return np.sum(np.log(z))


class PlanarFixedBandwidth(ForwardChainingValidationBase):

    ll_func = kde_wrapper

    def args_kwargs_generator(self, *args, **kwargs):
        """
        Prepare the *args and **kwargs that will be passed to the method along with training and testing data.
        :return: generator of tuples (args (iterable), kwargs (dict)), one per parameter combination
        """
        ndim = self.grid.ndim - 1
        data_getter = ([self.grid[i].flat[j] for i in range(ndim)] for j in range(self.nparam))
        for x in data_getter:
            yield ((), {'bandwidths': x, 'parallel': False})

    def run_one_timestep(self, training, testing, *args, **kwargs):
        shape = self.grid.shape[1:]
        the_func = partial(kde_wrapper, training, testing, models.FixedBandwidthKde)
        param_gen = self.args_kwargs_generator()
        if self.parallel:
            with contextlib.closing(mp.Pool()) as pool:
                z = np.array(pool.map(the_func, param_gen))
        else:
            z = np.array(map(the_func, param_gen))
        return z.reshape(shape)


class PlanarFixedBandwidthSpatialSymm(PlanarFixedBandwidth):
    """
    Supply only TWO dims when setting the grid and the spatial bandwidth is automatically copied
    """
    MAX_N_PARAM = 2
    MIN_N_PARAM = 2

    def args_kwargs_generator(self, *args, **kwargs):
        """
        Prepare the *args and **kwargs that will be passed to the method along with training and testing data.
        :return: generator of tuples (args (iterable), kwargs (dict)), one per parameter combination
        """
        ndim = self.grid.ndim - 1
        assert ndim == 2
        data_getter = ([self.grid[0].flat[j],
                        self.grid[1].flat[j],
                        self.grid[1].flat[j]] for j in range(self.nparam))
        for x in data_getter:
            yield ((), {'bandwidths': x, 'parallel': False})


def kde_persistent_wrapper(kde_obj, bandwidths):
    """
    Wrapper to call KDE for use when persistence is required (i.e. in the case of a netKDE).
    This means the KDE object is NOT reinstantiated each time. Instead the bandwidths are changed in place.
    Training and testing data are therefore not needed, because they remain in the KDE object between calls.
    :param kde_obj: The persistent object
    :param bandwidths: The new bandwidths for reassignment
    :return:
    """
    MIN_LOG = -50
    kde_obj.set_bandwidths_and_cutoffs(bandwidths)
    z = kde_obj.pdf()  # no target data as it is already present
    logz = np.log(z)
    # replace zeroes with an arbitrary minimum
    logz[logz < MIN_LOG] = MIN_LOG
    return np.sum(logz)


class NetworkFixedBandwidth(ForwardChainingValidationBase):

    ll_func = kde_persistent_wrapper
    MAX_N_PARAM = 2
    MIN_N_PARAM = 2
    kde_class = netmodels.NetworkTemporalKde

    def __init__(self, *args, **kwargs):
        super(NetworkFixedBandwidth, self).__init__(*args, **kwargs)
        ## TODO: parallelising would help but it needs to work differently
        # instead of doing parallel loop over parameters, we need parallel loop over successive days.
        # Disabling for now.
        self.parallel = False
        self.kde = None

    def args_kwargs_generator(self, *args, **kwargs):
        """
        Prepare the *args and **kwargs that will be passed to the method along with training and testing data.
        :return: generator of tuples (args (iterable), kwargs (dict)), one per parameter combination
        """
        ndim = self.grid.ndim - 1
        data_getter = ([self.grid[i].flat[j] for i in range(ndim)] for j in range(self.nparam))
        for x in data_getter:
            yield x

    def run_one_timestep(self, training, testing, *args, **kwargs):
        shape = self.grid.shape[1:]
        max_bandwidths = []
        for i in range(self.grid.shape[0]):
            max_bandwidths.append(np.max(self.grid[i, :]))
        the_kde = self.kde_class(training, max_bandwidths, targets=testing, verbose=False)
        # precompute network paths by calling pdf now.
        # this will ensure they are computed with a maximum cutoff value, so that all other bandwidths will result in
        # the original paths being used.
        # it will be fast to recompute this when we actually need to in the loop below.
        the_kde.pdf()
        # the_func = partial(kde_persistent_wrapper, the_kde)
        param_gen = self.args_kwargs_generator()
        # if self.parallel:
        #     with contextlib.closing(mp.Pool()) as pool:
        #         z = np.array(pool.map(the_func, param_gen))
        # else:
        full_results = []
        for bd in param_gen:
            the_kde.set_bandwidths_and_cutoffs(bd)
            full_results.append(the_kde.pdf())
        return full_results
        # z = np.array(map(the_func, param_gen))
        # return z.reshape(shape)


if __name__ == '__main__':
    # load some Chicago South data for testing purposes

    from analysis import chicago
    npt = 10  # number of points PER DIM to use when optimising bandwidths
    num_validation = 20  # number of prediction time windows
    param_extent = (1., 60., 50., 2000.)
    start_date = datetime.datetime(2011, 3, 1)  # first date for which data are required
    start_day_number = 50  # number of days (after start date) on which first prediction is made
    end_date = start_date + datetime.timedelta(days=start_day_number + num_validation)

    all_domains = chicago.get_chicago_side_polys(as_shapely=True)
    data, t0, cid = chicago.get_crimes_by_type(start_date=start_date,
                                               end_date=end_date,
                                               crime_type='burglary',
                                               domain=all_domains['South'])

    # load Chicago South network
    from database.chicago.loader import load_network
    from data.models import NetworkSpaceTimeData
    net = load_network('s')

    # snap crimes to network to obtain array
    snapped_data, snap_fail_idx = NetworkSpaceTimeData.from_cartesian(net, data=data, return_failure_idx=True)

    opt = NetworkFixedBandwidth(snapped_data, parallel=False, initial_cutoff=start_day_number)
    opt.set_logger(verbose=True)
    opt.set_parameter_grid(npt, *param_extent)
    opt.run(niter=num_validation)

    # debugging code

    # roller_one = opt.roller.iterator(1).next()
    # training = roller_one.training
    # testing = roller_one.testing
    # res_arr = opt.run_one_timestep(training=training, testing=testing)

    # compute one timestep manually so the KDE is available for testing
    # max_bandwidths = []
    # for i in range(opt.grid.shape[0]):
    #     max_bandwidths.append(np.max(opt.grid[i, :]))
    # the_kde = opt.kde_class(training, max_bandwidths, targets=testing, verbose=False)
    # the_kde.pdf()
    # param_gen = opt.args_kwargs_generator()
    # full_results = []
    # for bd in param_gen:
    #     the_kde.set_bandwidths_and_cutoffs(bd)
    #     full_results.append(the_kde.pdf())