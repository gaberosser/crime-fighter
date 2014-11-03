__author__ = 'gabriel'

from utils import linkages
import estimation
import ipdb
from kde import models as pp_kde
import numpy as np
from time import time
import warnings
from scipy import sparse
import math
import operator
import psutil
from data import models as data_models


class SepBase(object):
    def __init__(self,
                 data=None,
                 p=None,
                 max_delta_t=None,
                 max_delta_d=None,
                 bg_kde_kwargs=None,
                 trigger_kde_kwargs=None,
                 parallel=True):

        self.p = p
        self.data = None
        if len(data):
            self.set_data(data)
        self.parallel = parallel

        self.max_delta_t = None
        self.max_delta_d = None
        self.set_max_delta_t(max_delta_t)
        self.set_max_delta_d(max_delta_d)

        self.interpoint_data = None
        self.linkage = None
        self.linkage_cols = None
        self.set_linkages()

        self.bg_kde = None
        self.trigger_kde = None
        self.bg_kde_kwargs = bg_kde_kwargs or {}
        self.trigger_kde_kwargs = trigger_kde_kwargs or {}

        self.num_bg = []
        self.num_trig = []
        self.l2_differences = []
        self.run_times = []

        self.__init_extra__()

    def __init_extra__(self):
        """
        Function that gets called immediately after __init__
        Use this if any tweaks / defaults need to be applied at this stage
        """

    @property
    def ndata(self):
        return len(self.data)

    @property
    def ndim(self):
        raise NotImplementedError

    def reset(self):
        # reset storage containers
        self.num_bg = []
        self.num_trig = []
        self.l2_differences = []
        self.run_times = []
        self.bg_kde = None
        self.trigger_kde = None

    @property
    def data_time(self):
        raise NotImplementedError

    @property
    def data_space(self):
        raise NotImplementedError

    def initial_estimate(self, estimator):
        self.p = estimator(self.data_time, self.data_space, self.linkage)

    def set_data(self, data):
        """
        Can use this function to apply any preprocessing operations to data
        """
        self.data = data

    def set_max_delta_t(self, max_delta_t=None):
        """
        Define automatic parameter selection here if desired
        """
        if not max_delta_t:
            raise NotImplementedError
        self.max_delta_t = max_delta_t

    def set_max_delta_d(self, max_delta_d=None):
        """
        Define automatic parameter selection here if desired
        """
        if not max_delta_d:
            raise NotImplementedError
        self.max_delta_d = max_delta_d

    def set_linkages(self):
        # set self.linkage, self.linkage_col, self.interpoint_data
        raise NotImplementedError

    def target_source_linkages(self, target_data):
        """
        Compute the valid linkages between self.data and the supplied data set.
        :return: Same format as self.linkage, (idx_i array, idx_j array)
        """
        raise NotImplementedError

    def background_density(self, target_data, spatial_only=False):
        """
        Return the (unnormalised) density due to background events
        :param spatial_only: Boolean switch.  When enabled, only use the spatial component of the background, since
        using the time component leads to the background 'fading out' when predicting into the future.
        Integral over all data dimensions should return num_bg
        """
        raise NotImplementedError

    def trigger_density(self, delta_data):
        """
        Return the (unnormalised) trigger density
        Integral over all data dimensions should return num_trig / num_events
        """
        raise NotImplementedError

    def trigger_density_in_place(self, target_data, source_data=None):
        """
        Return the sum of trigger densities at the points in target_data.
        Optionally supply new source data to be used, otherwise self.data is used.
        """
        raise NotImplementedError

    def conditional_intensity(self, target_data, source_data=None, spatial_bg_only=False):
        """
        Evaluate the conditional intensity, lambda, at points in target_data.
        :param data: Optionally provide source data matrix to incorporate new history, affecting triggering,
         if None then run with training data.
        :param spatial_bg_only: Boolean.  When True, only the spatial components of the BG density are used.
        """
        raise NotImplementedError

    def predict(self, target_data):
        """
        Interface for validation code
        """
        return self.conditional_intensity(target_data)

    def predict_fixed_background(self, target_data):
        """
        When predicting in 'the future', cannot use the temporal component of the background as this vanishes.
        Therefore use ONLY the spatial components of the background here
        """
        return self.conditional_intensity(target_data, spatial_bg_only=True)

    def set_kdes(self):
        # set bg_kde and trigger_kde
        raise NotImplementedError

    def _iterate(self):
        colsum = self.p.sum(0)
        if np.any((colsum < (1 - 1e-12)) | (colsum > (1 + 1e-12))):
            raise AttributeError("Matrix P failed requirement that columns sum to 1 within tolerance.")
        if sparse.tril(self.p, k=-1).nnz != 0:
            raise AttributeError("Matrix P failed requirement that lower diagonal is zero.")

        tic = time()
        self.set_kdes()
        print "self._set_kdes() in %f s" % (time() - tic)

        # strip spatially overlapping points from p
        # self.delete_overlaps()

        # evaluate BG at data points
        tic = time()
        m = self.background_density(self.data)
        print "self.background_density() in %f s" % (time() - tic)

        # evaluate trigger KDE at all interpoint distances
        tic = time()
        trigger = self.trigger_density(self.interpoint_data)
        print "self.trigger_density() in %f s" % (time() - tic)
        g = sparse.csr_matrix((trigger, self.linkage), shape=(self.ndata, self.ndata))

        # recompute P
        # NB use LIL sparse matrix to avoid warnings about expensive structure changes, then convert at end.
        l = g.sum(axis=0) + m
        new_p = sparse.lil_matrix((self.ndata, self.ndata))
        new_p[range(self.ndata), range(self.ndata)] = m / l
        new_p[self.linkage] = trigger / l.flat[self.linkage[1]]
        new_p = new_p.tocsr()

        # compute difference
        q = new_p - self.p
        err_denom = float(self.p.nnz)
        self.l2_differences.append(math.sqrt(q.multiply(q).sum()) / err_denom)

        # update p
        self.p = new_p

    def train(self, data, niter=30, verbose=True, tol_p=None):

        self.set_data(data)
        # compute linkage indices
        self.set_linkages()

        # set tolerance if not specified
        tol_p = tol_p or 0.1 / float(self.ndata)

        # reset all other storage containers
        self.reset()

        # initial estimate for p if required
        if self.p is None:
            self.p = self.estimator(self.data, self.linkage)

        ps = []

        for i in range(niter):
            ps.append(self.p)
            tic = time()
            try:
                self._iterate()
            except Exception as exc:
                print repr(exc)
                raise
                # warnings.warn("Stopping training algorithm prematurely due to error on iteration %d." % (i+1))
                # break

            # record time taken
            self.run_times.append(time() - tic)

            if verbose:
                print "Completed %d / %d iterations in %f s.  L2 norm = %e" % (i+1, niter, self.run_times[-1], self.l2_differences[-1])

            if tol_p != 0. and self.l2_differences[-1] < tol_p:
                if verbose:
                    print "Training terminated in %d iterations as tolerance has been met." % (i+1)
                break
        return ps


class Sepp(SepBase):
    """
    Self-exciting point process class.
    Data type is CartesianSpaceTimeData class
    """
    data_class = data_models.CartesianSpaceTimeData

    @property
    def data_time(self):
        return self.data.time

    @property
    def data_space(self):
        return self.data.space

    def ndim(self):
        return self.data.nd

    def set_data(self, data):
        """
        Ensure that data has correct type
        """
        self.data = self.data_class(data)

    def set_linkages(self):
        # set self.linkage, self.linkage_col, self.interpoint_data
        self.linkage = linkages(self.data, self.max_delta_t, self.max_delta_d)
        self.interpoint_data = self.data[self.linkage[1]] - self.data[self.linkage[0]]
        self.linkage_cols = dict(
            [(i, np.concatenate((self.linkage[0][self.linkage[1] == i], [i,]))) for i in range(self.ndata)]
        )

    def target_source_linkages(self, target_data):
        """
        Compute the valid linkages between self.data and the supplied data set.
        :return: Same format as self.linkage, (idx_i array, idx_j array)
        """
        return linkages(self.data, self.max_delta_t, self.max_delta_d, data_target=target_data)

    def background_density(self, target_data, spatial_only=False):
        """
        Return the (unnormalised) density due to background events
        :param spatial_only: Boolean switch.  When enabled, only use the spatial component of the background, since
        using the time component leads to the background 'fading out' when predicting into the future.
        Integral over all data dimensions should return num_bg
        """

        num_bg = self.p.diagonal().sum()

        if spatial_only:
            ## FIXME: check norming here
            # estimate mean intensity per unit time
            T = self.data_time.ptp()
            k = num_bg / float(T)
            return k * self.bg_kde.partial_marginal_pdf(target_data.space, dim=0, normed=True)
        else:
            return self.bg_kde.pdf(target_data, normed=False)

    def trigger_density(self, delta_data):
        """
        Return the (unnormalised) trigger density
        Integral over all data dimensions should return num_trig / num_events
        """
        return self.trigger_kde.pdf(delta_data, normed=False) / self.ndata

    def trigger_density_in_place(self, target_data, source_data=None):
        """
        Return the sum of trigger densities at the points in target_data.
        Optionally supply new source data to be used, otherwise self.data is used.
        """
        if source_data is not None and len(source_data):
            pass
        else:
            source_data = self.data

        link_source, link_target = linkages(source_data, self.max_delta_t, self.max_delta_d, data_target=target_data)
        trigger = sparse.csr_matrix((source_data.ndata, target_data.ndata))

        if link_source.size:
            delta_data = target_data[link_target] - source_data[link_source]
            trigger[link_source, link_target] = self.trigger_density(delta_data)

        trigger = np.array(trigger.sum(axis=0))
        return trigger

    def conditional_intensity(self, target_data, source_data=None, spatial_bg_only=False):
        """
        Evaluate the conditional intensity, lambda, at points in target_data.
        :param data: Optionally provide source data matrix to incorporate new history, affecting triggering,
         if None then run with training data.
        :param spatial_bg_only: Boolean.  When True, only the spatial components of the BG density are used.
        """
        if source_data is not None and len(source_data):
            pass
        else:
            source_data = self.data

        bg = self.background_density(target_data, spatial_only=spatial_bg_only)
        trigger = self.trigger_density_in_place(target_data, source_data=source_data)

        return bg + trigger

    def set_kdes(self):
        # set bg_kde and trigger_kde
        raise NotImplementedError


class SeppStochastic(Sepp):

    bg_kde_class = pp_kde.FixedBandwidthKdeSeparable
    trigger_kde_class = pp_kde.FixedBandwidthKde


    def __init_extra__(self):
        super(SeppStochastic, self).__init_extra__()
        self.rng = np.random.RandomState()

    def set_seed(self, seed):
        self.rng.seed(seed)

    def sample_data(self):
        """
        Weighted sampling algorithm by Efraimidis and Spirakis. Weighted random sampling with a reservoir.
        Information Processing Letters 97 (2006) 181-185
        """
        urvs = self.rng.rand(self.p.nnz)
        ks_matrix = self.p.copy()
        ks_matrix.data = np.power(urvs, 1. / self.p.data)

        # find the largest value in each column
        causes = [self.linkage_cols[n][np.argmax(ks_matrix[:, n].data)] for n in range(self.ndata)]
        effects = range(self.ndata)

        bg_idx = [x for x, y in zip(causes, effects) if x == y]
        cause_idx, effect_idx = zip(*[(x, y) for x, y in zip(causes, effects) if x != y])

        return bg_idx, list(cause_idx), list(effect_idx)

    def set_kdes(self):
        # bg_idx, cause_idx, effect_idx = self.sample_data()
        bg_idx, cause_idx, effect_idx = self.sample_data()
        interpoint = self.data[effect_idx] - self.data[cause_idx]

        self.num_bg.append(len(bg_idx))
        self.num_trig.append(len(cause_idx))

        # compute KDEs
        try:
            self.bg_kde = self.bg_kde_class(self.data[bg_idx], **self.bg_kde_kwargs)
            self.trigger_kde = self.trigger_kde_class(interpoint, **self.trigger_kde_kwargs)

        except AttributeError as exc:
            print "Error.  Num BG: %d, num trigger %d" % (self.num_bg[-1], self.num_trig[-1])
            raise exc


class SeppStochasticNn(SeppStochastic):

    bg_kde_class = pp_kde.VariableBandwidthNnKdeSeparable
    trigger_kde_class = pp_kde.VariableBandwidthNnKde

    def __init_extra__(self):
        super(SeppStochastic, self).__init_extra__()
        self.rng = np.random.RandomState()
        if 'number_nn' not in self.trigger_kde_kwargs:
            self.trigger_kde_kwargs['number_nn'] = 100 if self.ndim == 1 else 15
        if 'number_nn' not in self.bg_kde_kwargs:
            self.bg_kde_kwargs['number_nn'] = [100, 15]
        else:
            if len(self.bg_kde_kwargs['number_nn']) != 2:
                raise AttributeError("Kwarg 'number_nn' in bg_kde_kwargs must have length 2")


class SeppStochasticNnAsymmetric(SeppStochasticNn):
    """
    As for parent class, except that the trigger KDE is manually reflected in the first dim about t=0.
    """

    trigger_kde_class = pp_kde.SpaceTimeVariableBandwidthNnTimeAsymmetricKde

    # def trigger_density(self, delta_data):
    #     """
    #     Return the (unnormalised) trigger density
    #     Integral over all data dimensions should return num_trig / num_events
    #     """
    #     res = self.trigger_kde.pdf(delta_data, normed=False) / self.ndata
    #     delta_date_neg = -delta_data.time[:]
    #     delta_date_neg = delta_date_neg.adddim(delta_data.getdim(1))
    #     delta_date_neg = delta_date_neg.adddim(delta_data.getdim(2))
    #     res_neg = self.trigger_kde.pdf(delta_date_neg, normed=False) / self.ndata
    #     res += res_neg
    #     res[delta_data.toarray(0) < 0] = 0.
    #     return res


class SeppStochasticNnSt(SeppStochasticNn):

    trigger_kde_class = pp_kde.SpaceTimeVariableBandwidthNnKde


class SeppStochasticNnStExp(SeppStochasticNn):
    trigger_kde_class = pp_kde.SpaceTimeExponentialVariableBandwidthNnKde


##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################



class PointProcess(object):
    def __init__(self, p=None, max_trigger_d=None, max_trigger_t=None, min_bandwidth=None, dtype=np.float64,
                 estimator=None, num_nn=None, parallel=True, sharedmem=False):
        self.dtype = dtype
        self.data = np.array([], dtype=dtype)
        self.T = 0.
        self.min_bandwidth = min_bandwidth
        self.parallel = parallel
        self.sharedmem = sharedmem

        if num_nn is not None:
            self.num_nn = num_nn
        else:
            self.num_nn = [None] * 3

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
            self.p = sparse.csr_matrix((self.ndata, self.ndata))
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

    def _set_linkages_meshed(self, data=None):
        """ Lightweight implementation for setting parent-offspring couplings, but consumes too much memory
            on larger datasets """

        data = data if data is not None else self.data

        td = reduce(operator.sub, np.meshgrid(data[:, 0], data[:, 0], copy=False))
        xd = reduce(operator.sub, np.meshgrid(data[:, 1], data[:, 1], copy=False))
        yd = reduce(operator.sub, np.meshgrid(data[:, 2], data[:, 2], copy=False))
        distances = np.sqrt(xd ** 2 + yd ** 2)

        return np.where((distances < self.max_trigger_d) & (td > 0) & (td < self.max_trigger_t))

    def _set_linkages_iterated(self, data=None, chunksize=2**16):
        """ Iteration-based approach to computing parent-offspring couplings, required when memory is limited """

        data = data if data is not None else self.data
        ndata = data.shape[0]

        chunksize = min(chunksize, ndata * (ndata - 1) / 2)
        idx_i, idx_j = estimation.pairwise_differences_indices(ndata)
        link_i = []
        link_j = []

        for k in xrange(0, len(idx_i), chunksize):
            i = idx_i[k:(k + chunksize)]
            j = idx_j[k:(k + chunksize)]
            t = data[j, 0] - data[i, 0]
            d = np.sqrt((data[j, 1] - data[i, 1])**2 + (data[j, 2] - data[i, 2])**2)
            mask = (t <= self.max_trigger_t) & (d <= self.max_trigger_d)
            link_i.extend(i[mask])
            link_j.extend(j[mask])

        return (np.array(link_i), np.array(link_j))

    def set_linkages(self):
        """
        Set the allowed parent-offspring couplings, based on the maximum permitted time and distance values.
        Meshgrid is a convenient but memory-heavy approach that fails on larger datasets
        """
        sysmem = psutil.virtual_memory().total
        N = self.ndata ** 2 * 8  # estimated nbytes for a square matrix
        if N / float(sysmem) > 0.05:
            self.linkage = self._set_linkages_iterated()
        else:
            self.linkage = self._set_linkages_meshed()

        # sanity check: no diagonals in linkage indices
        if not np.all(np.diff(np.vstack(self.linkage), axis=0)):
            raise AttributeError("Diagonal entries found in linkage indices")

        self.interpoint_distance_data = self.data[self.linkage[1], :] - self.data[self.linkage[0], :]
        self.linkage_cols = dict(
            [(i, np.concatenate((self.linkage[0][self.linkage[1] == i], [i,]))) for i in range(self.ndata)]
        )

    def sample_data(self):
        """
        Naive weighted sampling algorithm
        """
        bg_idx = []
        cause_idx = []
        effect_idx = []

        # iterate over columns / effects / offspring events
        for i in range(self.ndata):
            effect = i
            row_indices = self.linkage_cols[i]
            # p_nz = self.p[row_indices, i].toarray().flat
            p_nz = self.p[row_indices, i].data
            idx = np.argsort(p_nz)[::-1]
            sorted_p_nz = p_nz[idx]
            sampled_idx = estimation.weighted_choice_np(sorted_p_nz)
            cause = row_indices[idx[sampled_idx]]
            if cause == effect:
                # bg
                bg_idx.append(cause)
            else:
                # offspring
                cause_idx.append(cause)
                effect_idx.append(effect)

        return bg_idx, cause_idx, effect_idx

    def sample_data_efraimidis(self):
        """
        More optimised weighted sampling algorithm
        """
        urvs = np.random.random(self.p.nnz)
        ks_matrix = self.p.copy()
        ks_matrix.data = np.power(urvs, 1. / self.p.data)

        # find the largest value in each column
        causes = [self.linkage_cols[n][np.argmax(ks_matrix[:, n].data)] for n in range(self.ndata)]
        effects = range(self.ndata)

        bg_idx = [x for x, y in zip(causes, effects) if x == y]
        cause_idx, effect_idx = zip(*[(x, y) for x, y in zip(causes, effects) if x != y])

        return bg_idx, list(cause_idx), list(effect_idx)

    # def delete_overlaps(self):
    #     """ Prevent lineage links with exactly overlapping entries """
    #     n = np.arange(self.ndata)
    #     for i in n:
    #         rpt_idx = np.where(np.all(self.data[:, 1:] == self.data[i, 1:], axis=1) & (n != i))[0]
    #         self.p[i, rpt_idx] = 0.
    #     # renorm
    #     col_sums = np.sum(self.p, axis=0)
    #     self.p /= col_sums

    def _target_source_linkages(self, t, x, y, data=None, chunksize=2**16):
        """ Iteration-based approach to computing parent-offspring couplings for the arbitrary target locations
            supplied in t, x, y arrays.  Optionally can supply different data, in which case this is treated as the
            putative parent data instead of self.data.
            :return: Tuple of two index arrays, with same interpretation as self.linkages
            NB indices are treated as if t, x, y are flat"""

        data = data if data is not None else self.data
        ndata = data.shape[0]

        chunksize = min(chunksize, ndata ** 2)
        idx_i, idx_j = np.meshgrid(range(ndata), range(t.size), copy=False)
        link_i = []
        link_j = []

        for k in range(0, idx_i.size, chunksize):
            i = idx_i.flat[k:(k + chunksize)]
            j = idx_j.flat[k:(k + chunksize)]
            tt = t.flat[j] - data[i, 0]
            dd = np.sqrt((x.flat[j] - data[i, 1])**2 + (y.flat[j] - data[i, 2])**2)
            mask = (tt <= self.max_trigger_t) & (tt > 0.) & (dd <= self.max_trigger_d)
            link_i.extend(i[mask])
            link_j.extend(j[mask])

        return np.array(link_i), np.array(link_j)

    def background_density(self, t, x, y, spatial_only=False):
        """
        Return the (unnormalised) density due to background events
        :param spatial_only: Boolean switch.  When enabled, only use the spatial component of the background, since
        using the time component leads to the background 'fading out' when predicting into the future.
        NB normalise (x,y) components and keep t component unnormed
        Triple integral over all (t, x, y) should return num_bg
        """

        if spatial_only:
            # estimate mean intensity per unit time
            k = self.num_bg[-1] / float(self.T * self.ndata)
            return k * self.bg_xy_kde.pdf(x, y, normed=False)
        else:
            return self.bg_t_kde.pdf(t, normed=False) * self.bg_xy_kde.pdf(x, y, normed=True)

    def trigger_density(self, t, x, y):
        """
        Return the (unnormalised) trigger density
        Triple integral over all (t, x, y) should return num_trig / num_events
        """
        return self.trigger_kde.pdf(t, x, y, normed=False) / self.ndata

    def trigger_density_in_place(self, t, x, y, data=None):
        """
        Return the sum of trigger densities at the points in (t, x, y).
        Optionally supply new data to be used, otherwise self.data is used.
        """
        shp = t.shape

        if data is not None and len(data):
            data = np.array(data)
        else:
            data = self.data

        ndata = data.shape[0]
        link_source, link_target = self._target_source_linkages(t, x, y, data=data)
        trigger = sparse.csr_matrix((ndata, t.size))

        if link_source.size:
            dt = t.flat[link_target] - data[link_source, 0]
            dx = x.flat[link_target] - data[link_source, 1]
            dy = y.flat[link_target] - data[link_source, 2]
            trigger[link_source, link_target] = self.trigger_density(dt, dx, dy)

        trigger = np.array(trigger.sum(axis=0))
        # may need to reshape if t, x, y had shape before
        return trigger.reshape(shp)

    def _evaluate_conditional_intensity(self, t, x, y, data=None, spatial_bg_only=False):
        """
        Evaluate the conditional intensity, lambda, at point (t, x, y) or at points specified in 1D arrays t, x, y.
        :param data: Optionally provide data matrix to incorporate new history, affecting triggering,
         if None then run with training data.
        :param spatial_bg_only: Boolean.  When True, only the spatial components of the BG density are used.
        """
        shp = t.shape
        if (shp != x.shape) or (shp != y.shape):
            raise AttributeError("Dimensions of supplied t, x, y do not match")

        if data is not None and len(data):
            data = np.array(data)
        else:
            data = self.data

        bg = self.background_density(t, x, y, spatial_only=spatial_bg_only)
        trigger = self.trigger_density_in_place(t, x, y, data=data)

        return bg + trigger

    def predict(self, t, x, y):
        """
        Required for plugin to validation code
        """
        return self._evaluate_conditional_intensity(t, x, y)

    def predict_fixed_background(self, t, x, y):
        """
        When predicting in 'the future', cannot use the temporal component of the background as this vanishes.
        Therefore use ONLY the spatial components of the background here
        """
        return self._evaluate_conditional_intensity(t, x, y, spatial_bg_only=True)

    def _set_kdes(self):
        raise NotImplementedError()

    def _iterate(self):
        colsum = self.p.sum(0)
        if np.any((colsum < (1 - 1e-12)) | (colsum > (1 + 1e-12))):
            raise AttributeError("Matrix P failed requirement that columns sum to 1 within tolerance.")
        if sparse.tril(self.p, k=-1).nnz != 0:
            raise AttributeError("Matrix P failed requirement that lower diagonal is zero.")

        tic = time()
        self._set_kdes()
        print "self._set_kdes() in %f s" % (time() - tic)

        # strip spatially overlapping points from p
        # self.delete_overlaps()

        # evaluate BG at data points
        tic = time()
        m = self.background_density(self.data[:, 0], self.data[:, 1], self.data[:, 2])
        print "self.background_density() in %f s" % (time() - tic)

        # evaluate trigger KDE at all interpoint distances
        tic = time()
        trigger = self.trigger_density(
            self.interpoint_distance_data[:, 0],
            self.interpoint_distance_data[:, 1],
            self.interpoint_distance_data[:, 2],
            )
        print "self.trigger_density() in %f s" % (time() - tic)
        g = sparse.csr_matrix((trigger, self.linkage), shape=(self.ndata, self.ndata))

        # recompute P
        # NB use LIL sparse matrix to avoid warnings about expensive structure changes, then convert at end.
        l = g.sum(axis=0) + m
        new_p = sparse.lil_matrix((self.ndata, self.ndata))
        new_p[range(self.ndata), range(self.ndata)] = m / l
        new_p[self.linkage] = trigger / l.flat[self.linkage[1]]
        new_p = new_p.tocsr()

        # compute difference
        q = new_p - self.p
        err_denom = float(self.p.nnz)
        self.l2_differences.append(math.sqrt(q.multiply(q).sum()) / err_denom)

        # update p
        self.p = new_p


    def train(self, data, niter=30, verbose=True, tol_p=None):

        self.set_data(data)
        # compute linkage indices
        self.set_linkages()

        # set tolerance if not specified
        tol_p = tol_p or 0.1 / float(self.ndata)

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
                raise
                # warnings.warn("Stopping training algorithm prematurely due to error on iteration %d." % (i+1))
                # break

            # record time taken
            self.run_times.append(time() - tic)

            if verbose:
                print "Completed %d / %d iterations in %f s.  L2 norm = %e" % (i+1, niter, self.run_times[-1], self.l2_differences[-1])

            if tol_p != 0. and self.l2_differences[-1] < tol_p:
                if verbose:
                    print "Training terminated in %d iterations as tolerance has been met." % (i+1)
                break
        return ps


class PointProcessStochastic(PointProcess):
    ## FIXME: using the min_bandwidth argument for now - not cool.
    def _set_kdes(self):
        bg_idx, cause_idx, effect_idx = self.sample_data()
        interpoint = self.data[effect_idx] - self.data[cause_idx]

        self.num_bg.append(len(bg_idx))
        self.num_trig.append(len(cause_idx))

        # compute KDEs
        try:
            self.bg_t_kde = pp_kde.FixedBandwidthKde(self.data[bg_idx, 0], bandwidths=self.min_bandwidth[:1],
                                                     parallel=self.parallel)
            self.bg_xy_kde = pp_kde.FixedBandwidthKde(self.data[bg_idx, 1:], bandwidths=self.min_bandwidth[1:],
                                                      parallel=self.parallel)
            self.trigger_kde = pp_kde.FixedBandwidthKde(interpoint,
                                                        bandwidths=self.min_bandwidth,
                                                        parallel=self.parallel)
        except AttributeError as exc:
            print "Error.  Num BG: %d, num trigger %d" % (self.num_bg[-1], self.num_trig[-1])
            raise exc


class PointProcessStochasticNn(PointProcess):

    def _set_kdes(self):
        tic = time()
        bg_idx, cause_idx, effect_idx = self.sample_data_efraimidis()
        # bg_idx, cause_idx, effect_idx = self.sample_data()
        interpoint = self.data[effect_idx] - self.data[cause_idx]
        print "self.sample_data() in %f s" % (time() - tic)

        # compute KDEs
        try:
            self.bg_t_kde = pp_kde.VariableBandwidthNnKde(self.data[bg_idx, 0], nn=self.num_nn[0],
                                                          parallel=self.parallel)
            self.bg_xy_kde = pp_kde.VariableBandwidthNnKde(self.data[bg_idx, 1:], nn=self.num_nn[1],
                                                           parallel=self.parallel)
            self.trigger_kde = pp_kde.VariableBandwidthNnKde(interpoint,
                                                             min_bandwidth=self.min_bandwidth,
                                                             nn=self.num_nn[2],
                                                             parallel=self.parallel)
        except AttributeError as exc:
            print "Error.  Num BG: %d, num trigger %d" % (self.num_bg[-1], self.num_trig[-1])
            raise exc

        self.num_bg.append(len(bg_idx))
        self.num_trig.append(len(cause_idx))


class PointProcessDeterministicNn(PointProcess):

    def _set_kdes(self):
        p_bg = self.p.diagonal()
        p_trig = np.array(self.p[self.linkage].flat)

        self.bg_t_kde = pp_kde.WeightedVariableBandwidthNnKde(self.data[:, 0],
                                                              weights=p_bg,
                                                              nn=self.num_nn[0],
                                                              parallel=self.parallel)
        self.bg_xy_kde = pp_kde.WeightedVariableBandwidthNnKde(self.data[:, 1:],
                                                               weights=p_bg,
                                                               nn=self.num_nn[1],
                                                               parallel=self.parallel)
        self.trigger_kde = pp_kde.WeightedVariableBandwidthNnKde(self.interpoint_distance_data,
                                                                 weights=p_trig,
                                                                 min_bandwidth=self.min_bandwidth,
                                                                 nn=self.num_nn[2],
                                                                 parallel=self.parallel)
        self.num_bg.append(sum(p_bg))
        self.num_trig.append(self.ndata - sum(p_bg))


class PointProcessDeterministicFixedBandwidth(PointProcess):

    def _set_kdes(self):
        ## FIXME: using the min_bandwidth argument for now - not cool.
        p_bg = self.p.diagonal()

        self.bg_t_kde = pp_kde.WeightedFixedBandwidthKde(self.data[:, 0],
                                                         weights=p_bg,
                                                         bandwidths=[self.min_bandwidth[0]])
        self.bg_xy_kde = pp_kde.WeightedFixedBandwidthKde(self.data[:, 1:],
                                                          weights=p_bg,
                                                          bandwidths=self.min_bandwidth[1:])
        self.trigger_kde = pp_kde.WeightedFixedBandwidthKde(self.interpoint_distance_data,
                                                            weights=self.p[self.linkage],
                                                            bandwidths=self.min_bandwidth)
        self.num_bg.append(sum(p_bg))
        self.num_trig.append(self.ndata - sum(p_bg))
