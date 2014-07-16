__author__ = 'gabriel'

import estimation
import ipdb
from kde.methods import pure_python as pp_kde
import numpy as np
from time import time
import warnings
from scipy import sparse
import operator
import psutil

class PointProcess(object):
    def __init__(self, p=None, max_trigger_d=None, max_trigger_t=None, min_bandwidth=None, dtype=np.float64,
                 estimator=None, num_nn=None):
        self.dtype = dtype
        self.data = np.array([], dtype=dtype)
        self.T = 0.
        self.min_bandwidth = min_bandwidth
        self.num_nn = num_nn or [None] * 3
        if not hasattr(self.num_nn, '__iter__'):
            self.num_nn = [self.num_nn] * 3
            warnings.warn("Received single fixed number of NNs to use for KDE (%d).  Using this for all dimensions." %
                          self.num_nn[0])
        self.max_trigger_d = max_trigger_d
        self.max_trigger_t = max_trigger_t

        # initialise matrix p or use one provided
        if p is not None:
            self.p = p
            self.pset = True
        else:
            self.p = sparse.lil_matrix((self.ndata, self.ndata))
            self.pset = False
            # look for an estimator function
            self.estimator = estimator or estimation.estimator_bowers

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
        self.T = np.ptp(self.data[:, 0]) # time window
        # set threshold distance and time if not provided
        self.max_trigger_t = self.max_trigger_t or np.ptp(self.data[:, 0]) / 10.
        self.max_trigger_d = self.max_trigger_d or np.sqrt(np.ptp(self.data[:, 1])**2 + np.ptp(self.data[:, 2])**2) / 20.

    @property
    def ndata(self):
        return self.data.shape[0]

    @property
    def niter(self):
        return len(self.l2_differences)


    def _set_linkages_meshed(self):
        """ Lightweight implementation for setting parent-offspring couplings, but consumes too much memory
            on larger datasets """

        td = reduce(operator.sub, np.meshgrid(self.data[:, 0], self.data[:, 0], copy=False))
        xd = reduce(operator.sub, np.meshgrid(self.data[:, 1], self.data[:, 1], copy=False))
        yd = reduce(operator.sub, np.meshgrid(self.data[:, 2], self.data[:, 2], copy=False))
        distances = np.sqrt(xd ** 2 + yd ** 2)

        self.linkage = np.where((distances < self.max_trigger_d) & (td > 0) & (td < self.max_trigger_t))

    def _set_linkages_iterated(self, chunksize=2**16):
        """ Iteration-based approach to computing parent-offspring couplings, required when memory is limited """

        chunksize = min(chunksize, self.ndata * (self.ndata - 1) / 2)
        idx_i, idx_j = estimation.pairwise_differences_indices(self.ndata)
        link_i = []
        link_j = []

        for k in range(0, len(idx_i), chunksize):
            i = idx_i[k:(k + chunksize)]
            j = idx_j[k:(k + chunksize)]
            t = self.data[j, 0] - self.data[i, 0]
            d = np.sqrt((self.data[j, 1] - self.data[i, 1])**2 + (self.data[j, 2] - self.data[i, 2])**2)
            mask = (t <= self.max_trigger_t) & (d <= self.max_trigger_d)
            link_i.extend(i[mask])
            link_j.extend(j[mask])

        self.linkage = (np.array(link_i), np.array(link_j))


    def set_linkages(self):
        """
        Set the allowed parent-offspring couplings, based on the maximum permitted time and distance values.
        Meshgrid is a convenient but memory-heavy approach that fails on larger datasets
        """
        sysmem = psutil.virtual_memory().total
        N = self.ndata ** 2 * 8  # estimated nbytes for a square matrix
        if N / float(sysmem) > 0.05:
            self._set_linkages_iterated()
        else:
            self._set_linkages_meshed()

        self.interpoint_distance_data = self.data[self.linkage[1], :] - self.data[self.linkage[0], :]

    # def delete_overlaps(self):
    #     """ Prevent lineage links with exactly overlapping entries """
    #     n = np.arange(self.ndata)
    #     for i in n:
    #         rpt_idx = np.where(np.all(self.data[:, 1:] == self.data[i, 1:], axis=1) & (n != i))[0]
    #         self.p[i, rpt_idx] = 0.
    #     # renorm
    #     col_sums = np.sum(self.p, axis=0)
    #     self.p /= col_sums

    def background_density(self, t, x, y):
        """
        Return the (unnormalised) density due to background events
        """
        return self.bg_t_kde.pdf(t, normed=False) * self.bg_xy_kde.pdf(x, y, normed=False) / self.ndata

    def trigger_density(self, t, x, y):
        """
        Return the (unnormalised) trigger density
        """
        # return self.trigger_kde.pdf(t, x, y, normed=False) / self.num_bg[-1]
        return self.trigger_kde.pdf(t, x, y, normed=False) / self.ndata

    def evaluate_conditional_intensity(self, t, x, y, data=None):
        """
        Evaluate the conditional intensity, lambda, at point (t, x, y) or at points specified in 1D arrays t, x, y.
        Optionally provide data matrix to incorporate new history, otherwise run with training data.
        """
        data = data or self.data
        bg = self.background_density(t, x, y)

        if not isinstance(t, np.ndarray):
            t = np.array(t)
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        ttarget, tdata = np.meshgrid(t, data[:, 0])
        xtarget, xdata = np.meshgrid(x, data[:, 1])
        ytarget, ydata = np.meshgrid(y, data[:, 2])

        dt = ttarget - tdata
        dx = xtarget - xdata
        dy = ytarget - ydata

        # find region within max_trigger t and max_trigger_d
        idx_t = dt <= self.max_trigger_t
        idx_d = np.sqrt(dx**2 + dy**2) <= self.max_trigger_d
        idx = idx_t & idx_d

        trigger = np.zeros_like(dt)
        trigger[idx] = self.trigger_density(dt[idx], dx[idx], dy[idx])
        trigger = np.sum(trigger, axis=0)

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
        colsum = self.p.sum(0)
        if np.any((colsum < (1 - 1e-12)) | (colsum > (1 + 1e-12))):
            raise AttributeError("Matrix P failed requirement that columns sum to 1 within tolerance.")
        if sparse.tril(self.p, k=-1).nnz != 0:
            raise AttributeError("Matrix P failed requirement that lower diagonal is zero.")

        # strip spatially overlapping points from p
        # self.delete_overlaps()

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
            self.p = self.estimator(self.data, self.linkage)

        ps = []

        for i in range(niter):
            ps.append(self.p)
            tic = time()
            try:
                self._iterate()
            except Exception as exc:
                print repr(exc)
                warnings.warn("Stopping training algorithm prematurely due to error on iteration %d." % (i+1))
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
        self.bg_t_kde.set_weights(p_bg)
        self.bg_xy_kde.set_weights(p_bg)
        self.trigger_kde.set_weights(self.p[self.linkage])

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

    def background_density(self, t, x, y):
        """
        Return the (unnormalised) density due to background events
        """
        return self.bg_t_kde.pdf(t, normed=False) * self.bg_xy_kde.pdf(x, y, normed=False) / self.ndata

    def trigger_density(self, t, x, y):
        """
        Return the (unnormalised) trigger density
        """
        # return self.trigger_kde.pdf(t, x, y, normed=False) / self.num_bg[-1]
        return self.trigger_kde.pdf(t, x, y, normed=False) / self.ndata

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
            ps.append(self.p)
            tic = time()
            try:
                self._iterate()
            except Exception as exc:
                print repr(exc)
                warnings.warn("Stopping training algorithm prematurely due to error on iteration %d." % (i+1))
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
                                                         bandwidths=[self.min_bandwidth[0]])
        self.bg_xy_kde = pp_kde.WeightedFixedBandwidthKde(self.data[:, 1:],
                                                          weights=p_bg,
                                                          bandwidths=self.min_bandwidth[1:])
        self.trigger_kde = pp_kde.WeightedFixedBandwidthKde(self.interpoint_distance_data,
                                                            weights=self.p[self.linkage],
                                                            bandwidths=self.min_bandwidth)
