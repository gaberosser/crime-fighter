__author__ = 'gabriel'
import models
import netmodels
import numpy as np
import multiprocessing as mp
import datetime


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


if __name__ == '__main__':
    # load some Chicago data for testing purposes

    from analysis import chicago
    num_validation = 100  # number of predict - assess cycles
    start_date = datetime.datetime(2011, 3, 1)  # first date for which data are required
    start_day_number = 366  # number of days (after start date) on which first prediction is made
    end_date = start_date + datetime.timedelta(days=start_day_number + num_validation)

    all_domains = chicago.get_chicago_side_polys(as_shapely=True)
    data, t0, cid = chicago.get_crimes_by_type(start_date=start_date,
                                               end_date=end_date,
                                               crime_type='burglary',
                                               domain=all_domains['South'])
