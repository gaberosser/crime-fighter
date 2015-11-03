__author__ = 'gabriel'
import models
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


def compute_log_likelihood_surface_fixed_bandwidth(data,
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
            this_res = []
            training = data[data[:, 0] < (start_day + i)]
            testing_idx = (data[:, 0] >= (start_day + i)) & (data[:, 0] < (start_day + i + 1))
            testing = data[testing_idx]

            pool = mp.Pool()
            q = pool.map_async(
                _log_likelihood_fixed_wrapper,
                ((training, testing, (s, s, t)) for (s, t) in zip(ss.flat, tt.flat))
            )
            pool.close()
            this_res = q.get(1e100)
            res[:, :, i] = np.array(this_res).reshape((npts, npts))

    except Exception as exc:
        print "Terminating early due to Exception:"
        print repr(exc)

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
