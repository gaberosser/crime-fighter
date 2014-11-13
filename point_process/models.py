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
import copy


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

            if tol_p is not None and self.l2_differences[-1] < tol_p:
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
            delta_data = target_data.getrows(link_target) - source_data.getrows(link_source)
            trigger[link_source, link_target] = self.trigger_density(delta_data)

        trigger = np.array(trigger.sum(axis=0))
        # reshape if target_data has a shape
        if target_data.original_shape:
            trigger = trigger.reshape(target_data.original_shape)
        # else flatten
        else:
            trigger = trigger.flatten()
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
        if not len(bg_idx):
            raise ValueError("No BG events remaining")
        cause_effect = zip(*[(x, y) for x, y in zip(causes, effects) if x != y])
        if not len(cause_effect):
            raise ValueError("No trigger events remaining")
        cause_idx, effect_idx = cause_effect

        return bg_idx, list(cause_idx), list(effect_idx)

    def set_kdes(self):
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

    @property
    def weighted_bg_kde(self):
        return pp_kde.WeightedVariableBandwidthNnKdeSeparable(self.data, weights=self.p.diagonal(), **self.bg_kde_kwargs)

    @property
    def weighted_trigger_kde(self):
        p_trig = np.array(self.p[self.linkage].flat)
        return pp_kde.WeightedVariableBandwidthNnKde(self.interpoint_data, weights=p_trig, **self.trigger_kde_kwargs)


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


class SeppStochasticNnReflected(SeppStochasticNn):
    """
    As for parent class, except that the trigger KDE is manually reflected in the first dim about t=0.
    """
    trigger_kde_class = pp_kde.SpaceTimeVariableBandwidthNnTimeReflected


class SeppStochasticNnOneSided(SeppStochasticNn):
    """
    As for parent class, except that the trigger KDE is one sided (positive side of mean) in the first dim.
    """
    trigger_kde_class = pp_kde.SpaceTimeVariableBandwidthNnTimeOneSided


class SeppStochasticNnStExp(SeppStochasticNn):
    trigger_kde_class = pp_kde.SpaceTimeExponentialVariableBandwidthNnKde


class SeppDeterministicNn(Sepp):

    bg_kde_class = pp_kde.WeightedVariableBandwidthNnKdeSeparable
    trigger_kde_class = pp_kde.WeightedVariableBandwidthNnKde

    def set_kdes(self):
        p_bg = self.p.diagonal()
        p_trig = np.array(self.p[self.linkage].flat)

        self.num_bg.append(sum(p_bg))
        self.num_trig.append(sum(p_trig))

        self.bg_kde = self.bg_kde_class(self.data, weights=p_bg, **self.bg_kde_kwargs)
        self.trigger_kde = self.trigger_kde_class(self.interpoint_data, weights=p_trig, **self.trigger_kde_kwargs)


def fluctuation_pre_convergence(sepp_obj, niter=15):
    """
    Function to test the variability of the KDEs leading up to convergence.
    :param sepp_obj: pre-initialised SEPP object, with p already defined
    :param niter: number of iterations to use
    :return: dictionary of lists of: KDEs, weighted KDEs and probability matrices
    """

    # start recording kde
    triggers = []
    bgs = []
    weighted_triggers = []
    weighted_bgs = []
    ps = []

    for i in range(niter):
        sepp_obj._iterate()
        triggers.append(sepp_obj.trigger_kde)
        bgs.append(sepp_obj.bg_kde)
        weighted_triggers.append(sepp_obj.weighted_trigger_kde)
        weighted_bgs.append(sepp_obj.weighted_bg_kde)
        ps.append(sepp_obj.p.copy())

    return {
        'triggers': triggers,
        'weighted_triggers': weighted_triggers,
        'backgrounds': bgs,
        'weighted_backgrounds': weighted_bgs,
        'ps': ps,
        'sepp_obj': copy.deepcopy(sepp_obj),
    }


def fluctuation_at_convergence(sepp_obj, niter_initial=15, niter_after=30):
    """
    Function to test the variability of the KDEs after supposed convergence.
    :param sepp_obj: pre-initialised SEPP object, with p already defined
    :param niter_initial: number of iterations to use for initial training, after which convergence is assumed
    :param niter_after: number of iterations to carry out at convergence point
    :return: dictionary of lists of: KDEs, weighted KDEs and probability matrices
    """
    sepp_obj.train(sepp_obj.data, niter=niter_initial)

    # start recording kde
    triggers = [sepp_obj.trigger_kde]
    bgs = [sepp_obj.bg_kde]
    weighted_triggers = [sepp_obj.weighted_trigger_kde]
    weighted_bgs = [sepp_obj.weighted_bg_kde]
    ps = [sepp_obj.p.copy()]

    for i in range(niter_after):
        sepp_obj._iterate()
        triggers.append(sepp_obj.trigger_kde)
        bgs.append(sepp_obj.bg_kde)
        weighted_triggers.append(sepp_obj.weighted_trigger_kde)
        weighted_bgs.append(sepp_obj.weighted_bg_kde)
        ps.append(sepp_obj.p.copy())

    return {
        'triggers': triggers,
        'weighted_triggers': weighted_triggers,
        'backgrounds': bgs,
        'weighted_backgrounds': weighted_bgs,
        'ps': ps,
        'sepp_obj': copy.deepcopy(sepp_obj),
    }


def variation_in_convergent_state(sepp_obj, niter_initial=15, num_repeats=20):
    """
    Function to test the variability of the KDEs upon repeated training.
    :param sepp_obj: pre-initialised SEPP object, with p already defined
    :param niter_initial: number of iterations to use for initial training, after which convergence is assumed
    :param num_repeats: number of repeated trainings to carry out
    :return: dictionary of lists of: KDEs, weighted KDEs and probability matrices, one per repeat
    """
    sepp_obj.train(sepp_obj.data, niter=niter_initial)

    # start recording kde
    triggers = []
    bgs = []
    weighted_triggers = []
    weighted_bgs = []
    ps = []
    sepp_objs = []

    for i in range(num_repeats):
        sepp_obj.train(sepp_obj.data, niter=niter_initial)
        sepp_objs.append(copy.deepcopy(sepp_obj))
        triggers.append(sepp_obj.trigger_kde)
        bgs.append(sepp_obj.bg_kde)
        weighted_triggers.append(sepp_obj.weighted_trigger_kde)
        weighted_bgs.append(sepp_obj.weighted_bg_kde)
        ps.append(sepp_obj.p.copy())

    return {
        'triggers': triggers,
        'weighted_triggers': weighted_triggers,
        'backgrounds': bgs,
        'weighted_backgrounds': weighted_bgs,
        'ps': ps,
        'sepp_objs': sepp_objs,
    }
