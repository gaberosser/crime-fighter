__author__ = 'gabriel'

import estimation
import ipdb
from kde.methods import pure_python as pp_kde
import numpy as np
from time import time
import warnings

class PointProcess(object):
    def __init__(self, p=None, max_trigger_d=None, max_trigger_t=None, min_bandwidth=None, dtype=np.float64,
                 estimator=None, num_nn=None):
        self.dtype = dtype
        self.data = np.array([], dtype=dtype)
        self.min_bandwidth = min_bandwidth
        self.num_nn = num_nn or [None] * 3
        if not hasattr(self.num_nn, '__iter__'):
            self.num_nn = [self.num_nn] * 3
            warnings.warn("Received single fixed number of NNs to use for KDE (%d).  Using this for all dimensions.",
                          self.num_nn)
        self.max_trigger_d = max_trigger_d
        self.max_trigger_t = max_trigger_t

        # initialise matrix p or use one provided
        if p is not None:
            self.p = p
            self.pset = True
        else:
            self.p = np.zeros((self.ndata, self.ndata))
            self.pset = False
            # look for an estimator function
            self.estimator = estimator or estimation.initial_guess_educated

        # init storage containers
        self.linkage = []
        self.interpoint_distance_data = []
        self.run_times = []
        self.num_bg = []
        self.num_trig = []
        self.l2_differences = []
        self.bg_t_kde = None
        self.bg_xy_kde = None
        self.trigger_kde = None

    def reset(self):
        # reset storage containers
        self.run_times = []
        self.num_bg = []
        self.num_trig = []
        self.l2_differences = []
        self.bg_t_kde = None
        self.bg_xy_kde = None
        self.trigger_kde = None

    def set_data(self, data):
        self.data = np.array(data, dtype=self.dtype)
        # sort data by time
        self.data = self.data[self.data[:, 0].argsort()]
        # set threshold distance and time if not provided
        self.max_trigger_t = self.max_trigger_t or np.ptp(self.data[:, 0]) / 10.
        self.max_trigger_d = self.max_trigger_d or np.sqrt(np.ptp(self.data[:, 1])**2 + np.ptp(self.data[:, 2])**2) / 20.

    @property
    def ndata(self):
        return self.data.shape[0]

    @property
    def niter(self):
        return len(self.l2_differences)

    def set_linkages(self):
        pdiff = estimation.pairwise_differences(self.data, dtype=self.dtype)
        distances = np.sqrt(pdiff[:, :, 1] ** 2 + pdiff[:, :, 2] ** 2)
        self.linkage = np.where((distances < self.max_trigger_d) & (pdiff[:, :, 0] > 0) & (pdiff[:, :, 0] < self.max_trigger_t))
        self.interpoint_distance_data = self.data[self.linkage[1], :] - self.data[self.linkage[0], :]

    def background_density(self, t, x, y):
        """
        Return the (unnormalised) density due to background events
        """
        return self.bg_t_kde.pdf(t, normed=False) * self.bg_xy_kde.pdf(x, y, normed=False) / self.ndata

    def trigger_density(self, t, x, y):
        """
        Return the (unnormalised) trigger density
        """
        return self.trigger_kde.pdf(t, x, y, normed=False) / self.ndata

    def evaluate_conditional_intensity(self, t, x, y, data=None):
        """
        Evaluate the conditional intensity, lambda, at point (t, x, y) or at points specified in 1D arrays t, x, y.
        Optionally provide data matrix to incorporate new history, otherwise run with training data.
        """
        data = data or self.data
        bg = self.background_density(t, x, y)
        # bg = self.bg_t_kde.pdf(t) * self.bg_xy_kde.pdf(x, y)

        if not isinstance(t, np.ndarray):
            t = np.array(t)
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        ttarget, tdata = np.meshgrid(t, data[:, 0])
        xtarget, xdata = np.meshgrid(x, data[:, 1])
        ytarget, ydata = np.meshgrid(y, data[:, 2])
        trigger = np.sum(self.trigger_density(ttarget - tdata, xtarget - xdata, ytarget - ydata), axis=0)

        # may need to reshape if t, x, y had shape before
        shp = t.shape
        trigger = trigger.reshape(shp)

        return bg + trigger

    def predict(self, t, x, y):
        """
        Required for plugin to validation code
        """
        return self.evaluate_conditional_intensity(t, x, y)

    def _iterate(self):
            colsum = np.sum(self.p, axis=0)
            if np.any((colsum < (1 - 1e-12)) | (colsum > (1 + 1e-12))):
                raise AttributeError("Matrix P failed requirement that columns sum to 1 within tolerance.")
            if np.any(np.tril(self.p, k=-1) != 0.):
                raise AttributeError("Matrix P failed requirement that lower diagonal is zero.")

            bg, interpoint, cause_effect = estimation.sample_bg_and_interpoint(self.data, self.p)
            self.num_bg.append(bg.shape[0])
            self.num_trig.append(interpoint.shape[0])

            # compute KDEs
            try:
                self.bg_t_kde = pp_kde.VariableBandwidthNnKde(bg[:, 0], nn=self.num_nn[0])
                self.bg_xy_kde = pp_kde.VariableBandwidthNnKde(bg[:, 1:], nn=self.num_nn[1])
                self.trigger_kde = pp_kde.VariableBandwidthNnKde(interpoint,
                                                                 min_bandwidth=self.min_bandwidth,
                                                                 nn=self.num_nn[2])
            except AttributeError as exc:
                print "Error.  Num BG: %d, num trigger %d" % (bg.shape[0], interpoint.shape[0])
                raise exc

            # evaluate BG at data points
            m = self.background_density(self.data[:, 0], self.data[:, 1], self.data[:, 2])

            # evaluate trigger KDE at all interpoint distances
            g = np.zeros((self.ndata, self.ndata))
            g[self.linkage] = self.trigger_density(self.interpoint_distance_data[:, 0],
                                                   self.interpoint_distance_data[:, 1],
                                                   self.interpoint_distance_data[:, 2])

            # sanity check
            if np.any(np.diagonal(g) != 0):
                raise AttributeError("Non-zero diagonal values found in g.")

            # recompute P
            l = np.sum(g, axis=0) + m
            new_p = (m / l) * np.eye(self.ndata) + (g / l)

            # compute difference
            q = new_p - self.p
            err_denom = float(self.ndata * (self.ndata + 1)) / 2.
            self.l2_differences.append(np.sqrt(np.sum(q**2)) / err_denom)

            # update p
            self.p = new_p


    def train(self, data, niter=30, verbose=True, tol_p=1e-7):

        self.set_data(data)
        # compute linkage indices
        self.set_linkages()

        # reset all other storage containers
        self.reset()

        # initial estimate for p if required
        if not self.pset:
            self.p = self.estimator(self.data)

        ps = []

        for i in range(niter):
            # ipdb.set_trace()
            ps.append(self.p)
            tic = time()
            try:
                self._iterate()
            except Exception as exc:
                print repr(exc)
                warnings.warn("Stopping training algorithm prematurely due to error on iteration %d." % i+1)
                break

            # record time taken
            self.run_times.append(time() - tic)

            if verbose:
                print "Completed %d / %d iterations in %f s.  L2 norm = %e" % (i+1, niter, self.run_times[-1], self.l2_differences[-1])

            if tol_p != 0. and self.l2_differences[-1] < tol_p:
                if verbose:
                    print "Training terminated in %d iterations as tolerance has been met." % (i+1)
                break
        return ps


class PointProcessDeterministic(PointProcess):

    def set_kdes(self):
        p_bg = np.diag(self.p)
        self.bg_t_kde = pp_kde.WeightedVariableBandwidthNnKde(self.data[:, 0],
                                                              weights=p_bg,
                                                              nn=self.num_nn[0])
        self.bg_xy_kde = pp_kde.WeightedVariableBandwidthNnKde(self.data[:, 1:],
                                                               weights=p_bg,
                                                               nn=self.num_nn[1])
        self.trigger_kde = pp_kde.WeightedVariableBandwidthNnKde(self.interpoint_distance_data,
                                                                 weights=self.p[self.linkage],
                                                                 min_bandwidth=self.min_bandwidth,
                                                                 nn=self.num_nn[2])

    def _iterate(self):
        colsum = np.sum(self.p, axis=0)
        if np.any((colsum < (1 - 1e-12)) | (colsum > (1 + 1e-12))):
            raise AttributeError("Matrix P failed requirement that columns sum to 1 within tolerance.")
        if np.any(np.tril(self.p, k=-1) != 0.):
            raise AttributeError("Matrix P failed requirement that lower diagonal is zero.")

        p_bg = np.diag(self.p)
        effective_num_bg = sum(p_bg)
        self.num_bg.append(effective_num_bg)
        self.num_trig.append(self.ndata - effective_num_bg)

        # reset KDE weights
        self.bg_t_kde.weights = p_bg
        self.bg_xy_kde.weights = p_bg
        self.trigger_kde.weights = self.p[self.linkage]

        # evaluate BG at data points
        m = self.background_density(self.data[:, 0], self.data[:, 1], self.data[:, 2])

        # evaluate trigger KDE at all interpoint distances

        g = np.zeros((self.ndata, self.ndata))
        g[self.linkage] = self.trigger_density(self.interpoint_distance_data[:, 0],
                                               self.interpoint_distance_data[:, 1],
                                               self.interpoint_distance_data[:, 2])

        # sanity check
        if np.any(np.diagonal(g) != 0):
            raise AttributeError("Non-zero diagonal values found in g.")

        # recompute P
        l = np.sum(g, axis=0) + m
        new_p = (m / l) * np.eye(self.ndata) + (g / l)

        # compute difference
        q = new_p - self.p
        err_denom = float(self.ndata * (self.ndata + 1)) / 2.
        self.l2_differences.append(np.sqrt(np.sum(q**2)) / err_denom)

        # update p
        self.p = new_p

    def train(self, data, niter=30, verbose=True, tol_p=1e-7):

        self.set_data(data)
        # compute linkage indices
        self.set_linkages()

        # reset all other storage containers
        self.reset()

        # initial estimate for p if required
        if not self.pset:
            self.p = self.estimator(self.data)

        # set KDEs once now - weights will change but not bandwidths

        try:
            self.set_kdes()
        except AttributeError as exc:
            print repr(exc)
            raise exc

        ps = []

        for i in range(niter):
            # ipdb.set_trace()
            ps.append(self.p)
            tic = time()
            try:
                self._iterate()
            except Exception as exc:
                print repr(exc)
                warnings.warn("Stopping training algorithm prematurely due to error on iteration %d." % i+1)
                break

            # record time taken
            self.run_times.append(time() - tic)

            if verbose:
                print "Completed %d / %d iterations in %f s.  L2 norm = %e" % (i+1, niter, self.run_times[-1], self.l2_differences[-1])

            if tol_p != 0. and self.l2_differences[-1] < tol_p:
                if verbose:
                    print "Training terminated in %d iterations as tolerance has been met." % (i+1)
                break
        return ps


class PointProcessDeterministicFixedBandwidth(PointProcessDeterministic):

    def set_kdes(self):
        ## FIXME: using the min_bandwidth argument for now - not cool.
        p_bg = np.diag(self.p)
        self.bg_t_kde = pp_kde.WeightedFixedBandwidthKde(self.data[:, 0],
                                                         weights=p_bg,
                                                         bandwidths=self.min_bandwidth[0])
        self.bg_xy_kde = pp_kde.WeightedFixedBandwidthKde(self.data[:, 1:],
                                                          weights=p_bg,
                                                          bandwidths=self.min_bandwidth[1:])
        self.trigger_kde = pp_kde.WeightedFixedBandwidthKde(self.interpoint_distance_data,
                                                            weights=self.p[self.linkage],
                                                            bandwidths=self.min_bandwidth)
