__author__ = 'gabriel'

from utils import linkage_func_separable, linkages, random_sample_from_p
from kde import models as pp_kde
import numpy as np
from time import time
import warnings
from scipy import sparse, special
import math
from data import models as data_models
import copy
import collections
import logging


logger = logging.getLogger('point_process.models')
# default: output all logs to console
ch = logging.StreamHandler()
logger.setLevel(logging.DEBUG)
logger.addHandler(ch)

class SepBase(object):
    def __init__(self,
                 data=None,
                 p=None,
                 estimation_function=None,
                 max_delta_t=None,
                 max_delta_d=None,
                 bg_kde_kwargs=None,
                 trigger_kde_kwargs=None,
                 parallel=True,
                 remove_coincident_points=True,
                 **kwargs):

        self.p = p
        self.data = None
        self.parallel = parallel
        self.estimation_function = estimation_function

        self.max_delta_t = None
        self.max_delta_d = None
        self.set_max_delta_t(max_delta_t)
        self.set_max_delta_d(max_delta_d)
        self.remove_coincident_points = remove_coincident_points

        self.interpoint_data = None
        self.linkage = None
        self.linkage_cols = None

        if data is not None:
            self.set_data(data)
            self.set_linkages()
            if p is None and self.estimation_function is not None:
                self.initial_estimate()

        self.bg_kde = None
        self.trigger_kde = None
        self.bg_kde_kwargs = bg_kde_kwargs or {}
        self.trigger_kde_kwargs = trigger_kde_kwargs or {}

        self.num_bg = []
        self.num_trig = []
        self.l2_differences = []
        self.log_likelihoods = []
        self.run_times = []

        self.__init_extra__(**kwargs)

    def __init_extra__(self, **kwargs):
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
        self.log_likelihoods = []
        self.run_times = []
        self.bg_kde = None
        self.trigger_kde = None

    @property
    def data_time(self):
        raise NotImplementedError

    @property
    def data_space(self):
        raise NotImplementedError

    def initial_estimate(self):
        if self.estimation_function is None:
            raise AttributeError("No supplied estimator function")
        self.p = self.estimation_function(self.data.data, self.linkage)

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

    def predict(self, target_data, source_data=None):
        """
        Interface for validation code
        """
        return self.conditional_intensity(target_data, source_data=source_data)

    def predict_fixed_background(self, target_data, source_data=None):
        """
        When predicting in 'the future', cannot use the temporal component of the background as this vanishes.
        Therefore use ONLY the spatial components of the background here
        """
        return self.conditional_intensity(target_data, spatial_bg_only=True, source_data=source_data)

    def set_kdes(self):
        # set bg_kde and trigger_kde
        raise NotImplementedError

    def compute_p_l(self):
        """
        Compute the probability matrix P and conditional intensity l
        """
        # evaluate BG at data points
        m = self.background_density(self.data)

        # evaluate trigger KDE at all interpoint distances
        trigger = self.trigger_density(self.interpoint_data)
        g = sparse.csr_matrix((trigger, self.linkage), shape=(self.ndata, self.ndata))

        # recompute P, making sure to maintain stored zeros
        # NB: summing sparse matrices automatically collapses any zero entries in the result, so don't do it
        l = g.sum(axis=0) + m
        trigger_component = trigger / np.array(l.flat[self.linkage[1]]).squeeze()
        bg_component = np.array(m / l).squeeze()

        # create equivalent to linkage_idx for the diagonal BG entries
        bg_linkage = (range(self.ndata), range(self.ndata))

        # set new P matrix
        new_p = sparse.csr_matrix((bg_component, bg_linkage), shape=(self.ndata, self.ndata))
        new_p[self.linkage] = trigger_component

        return new_p, l

    def _iterate(self):
        colsum = self.p.sum(0)
        if np.any((colsum < (1 - 1e-12)) | (colsum > (1 + 1e-12))):
            raise AttributeError("Matrix P failed requirement that columns sum to 1 within tolerance.")
        if sparse.tril(self.p, k=-1).nnz != 0:
            raise AttributeError("Matrix P failed requirement that lower diagonal is zero.")

        tic = time()
        self.set_kdes()
        logger.info("self.set_kdes() in %f s" % (time() - tic))

        # compute new p and conditional intensities
        tic = time()
        new_p, l = self.compute_p_l()
        logger.info("self.compute_p_l() in %f s" % (time() - tic))

        # compute difference
        q = new_p - self.p
        err_denom = float(self.p.nnz)
        self.l2_differences.append(math.sqrt(q.multiply(q).sum()) / err_denom)
        self.log_likelihoods.append(np.sum(np.log(l)))

        # update p
        self.p = new_p

    def train(self, data=None, niter=30, verbose=True):
        if data is not None:
            # set data, linkages and p
            self.set_data(data)
            self.set_linkages()
            self.initial_estimate()  ## FIXME: remove this line?
        elif self.data is None:
            raise AttributeError("No data supplied")

        if self.p is None:
            self.initial_estimate()

        # reset all other storage containers
        self.reset()

        ps = []

        for i in range(niter):
            ps.append(self.p)
            tic = time()
            self._iterate()

            # record time taken
            self.run_times.append(time() - tic)

            if verbose:
                num_bg = self.p.diagonal().sum()
                logger.info("Completed %d / %d iterations in %.3f s.  Log likelihood = %.1f. No. BG: %.2f, no. trig.: %.2f" % (
                    i+1,
                    niter,
                    self.run_times[-1],
                    self.log_likelihoods[-1],
                    num_bg,
                    self.ndata - num_bg))

            # check for all trigger / BG situation and halt if that has happened
            if self.num_bg[-1] == 0:
                logger.info("Terminating training; no more BG component")
                break
            if self.num_trig[-1] == 0:
                logger.info("Terminating training; no more trigger component")
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

    @property
    def ndim(self):
        return self.data.nd

    def set_data(self, data):
        """
        Ensure that data has correct type
        """
        self.data = self.data_class(data)

    def set_linkages(self):
        # set self.linkage, self.linkage_col, self.interpoint_data
        linkage_fun = linkage_func_separable(self.max_delta_t, self.max_delta_d)
        self.linkage = linkages(self.data,
                                linkage_fun,
                                remove_coincident_pairs=self.remove_coincident_points)
        self.interpoint_data = self.data.getrows(self.linkage[1]) - self.data.getrows(self.linkage[0])
        self.linkage_cols = dict(
            [(i, np.concatenate((self.linkage[0][self.linkage[1] == i], [i,]))) for i in range(self.ndata)]
        )

    def target_source_linkages(self, target_data):
        """
        Compute the valid linkages between self.data and the supplied data set.
        :return: Same format as self.linkage, (idx_i array, idx_j array)
        """
        linkage_fun = linkage_func_separable(self.max_delta_t, self.max_delta_d)
        return linkages(self.data, linkage_fun, data_target=target_data)

    def background_density(self, target_data, spatial_only=False):
        """
        Return the (unnormalised) density due to background events
        :param spatial_only: Boolean switch.  When enabled, only use the spatial component of the background, since
        using the time component leads to the background 'fading out' when predicting into the future.
        Integral over all data dimensions should return num_bg
        """
        # if no background component, return zeroes
        if self.bg_kde is None:
            return np.zeros(target_data.ndata)

        num_bg = self.p.diagonal().sum()

        if spatial_only:
            ## FIXME: check norming here
            # estimate mean intensity per unit time
            T = np.ptp(self.data_time)
            k = num_bg / float(T)
            return k * self.bg_kde.partial_marginal_pdf(target_data.space, dim=0, normed=True)
        else:
            return self.bg_kde.pdf(target_data, normed=False)

    def trigger_density(self, delta_data, spatial_only=False):
        """
        Return the (unnormalised) trigger density
        Integral over all data dimensions should return num_trig / num_events... This kernel is then summed over all
        data points, returning a total mass of num_trig as required.
        """
        # if no trigger component, return zeroes
        if self.trigger_kde is None:
            return np.zeros(delta_data.ndata)

        if spatial_only:
            return self.trigger_kde.partial_marginal_pdf(delta_data.space, normed=False) / self.ndata
        else:
            return self.trigger_kde.pdf(delta_data, normed=False) / self.ndata

    def trigger_density_in_place(self, target_data, source_data=None, spatial_only=False):
        """
        Return the sum of trigger densities at the points in target_data.
        Optionally supply new source data to be used, otherwise self.data is used.
        """
        if source_data is not None and len(source_data):
            pass
        else:
            source_data = self.data

        linkage_fun = linkage_func_separable(self.max_delta_t, self.max_delta_d)
        link_source, link_target = linkages(source_data, linkage_fun, data_target=target_data)
        trigger = sparse.csr_matrix((source_data.ndata, target_data.ndata))

        if link_source.size:
            delta_data = target_data.getrows(link_target) - source_data.getrows(link_source)
            trigger[link_source, link_target] = self.trigger_density(delta_data, spatial_only=spatial_only)

        trigger = np.array(trigger.sum(axis=0))
        # reshape if target_data has a shape
        if target_data.original_shape:
            ## FIXME: double check this reshape order
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


class Senp(Sepp):
    """ Self-exciting network process """
    data_class = data_models.NetworkSpaceTimeData


class SeppStochastic(Sepp):

    bg_kde_class = pp_kde.FixedBandwidthKdeSeparable
    trigger_kde_class = pp_kde.FixedBandwidthKde

    def __init_extra__(self, **kwargs):
        super(SeppStochastic, self).__init_extra__(**kwargs)
        self.rng = np.random.RandomState()
        if 'seed' in kwargs:
            self.set_seed(kwargs['seed'])
        self.bg_kde_kwargs['parallel'] = self.parallel
        self.trigger_kde_kwargs['parallel'] = self.parallel

    def set_seed(self, seed):
        self.rng.seed(seed)

    def set_parallel(self, state):
        if state is True:
            self.parallel = True
        elif state is False:
            self.parallel = False
        else:
            raise AttributeError("input argument must be either True or False")

        if self.bg_kde:
            self.bg_kde.parallel = self.parallel
        if self.trigger_kde:
            self.trigger_kde.parallel = self.parallel
        self.bg_kde_kwargs['parallel'] = self.parallel
        self.trigger_kde_kwargs['parallel'] = self.parallel

    def sample_data(self):
        return random_sample_from_p(self.p, self.linkage_cols, rng=self.rng)


    def set_kdes(self):
        bg_idx, cause_idx, effect_idx = self.sample_data()
        interpoint = self.data[effect_idx] - self.data[cause_idx]

        self.num_bg.append(len(bg_idx))
        self.num_trig.append(len(cause_idx))

        # compute KDEs
        try:
            if len(bg_idx):
                self.bg_kde = self.bg_kde_class(self.data[bg_idx], **self.bg_kde_kwargs)
            else:
                # override the KDE
                self.bg_kde = None
            if len(interpoint) > 1:
                self.trigger_kde = self.trigger_kde_class(interpoint, **self.trigger_kde_kwargs)
            else:
                # override the KDE
                self.trigger_kde = None

        except AttributeError as exc:
            logger.error("Unable to set_kdes. Num BG: %d, num trigger %d" % (self.num_bg[-1], self.num_trig[-1]))
            raise exc

    @property
    def weighted_bg_kde(self):
        return pp_kde.WeightedVariableBandwidthNnKdeSeparable(self.data, weights=self.p.diagonal(), **self.bg_kde_kwargs)

    @property
    def weighted_trigger_kde(self):
        p_trig = np.array(self.p[self.linkage].flat)
        return pp_kde.WeightedVariableBandwidthNnKde(self.interpoint_data, weights=p_trig, **self.trigger_kde_kwargs)

    ## TODO: think about adding method for weighted_trigger_density, weighted_trigger_density_in_place, weighted_background_density


class SeppStochasticStationaryBg(SeppStochastic):
    """
    Stationary background (2D).
    Trigger function 3D.
    Bandwidths in both computed using Scott's rule-of-thumb plugin bandwidth
    """

    bg_kde_class = pp_kde.FixedBandwidthKdeScott
    trigger_kde_class = pp_kde.FixedBandwidthKdeScott

    def set_kdes(self):
        bg_idx, cause_idx, effect_idx = self.sample_data()
        interpoint = self.data[effect_idx] - self.data[cause_idx]

        self.num_bg.append(len(bg_idx))
        self.num_trig.append(len(cause_idx))

        # compute KDEs
        try:
            self.bg_kde = self.bg_kde_class(self.data[bg_idx, 1:], **self.bg_kde_kwargs)
            self.trigger_kde = self.trigger_kde_class(interpoint, **self.trigger_kde_kwargs)

        except AttributeError as exc:
            logger.error("Unable to set_kdes. Num BG: %d, num trigger %d" % (self.num_bg[-1], self.num_trig[-1]))
            raise exc

    def background_density(self, target_data, spatial_only=True):
        """
        Return the (unnormalised) density due to background events
        :param spatial_only: Ignore.  This MUST be True, as BG is spatial only.
        """

        num_bg = self.p.diagonal().sum()
        T = np.ptp(self.data_time)
        k = num_bg / float(T)
        return self.bg_kde.pdf(target_data.space, normed=False) / T


class SeppStochasticNn(SeppStochastic):

    bg_kde_class = pp_kde.VariableBandwidthNnKdeSeparable
    trigger_kde_class = pp_kde.VariableBandwidthNnKde

    def __init_extra__(self, **kwargs):
        """
        Check whether number_nn has been supplied in the KDE kwargs and add default values if not.
        The defaults should depend upon the number of dimensions, but we can't guarantee that data are defined at this
        stage, so just use sensible values. This may break if ndim == 1, but when is that ever the case?
        """
        super(SeppStochasticNn, self).__init_extra__(**kwargs)
        if 'number_nn' not in self.trigger_kde_kwargs:
            self.trigger_kde_kwargs['number_nn'] = 15
        if 'number_nn' not in self.bg_kde_kwargs:
            self.bg_kde_kwargs['number_nn'] = [100, 15]
        else:
            if len(self.bg_kde_kwargs['number_nn']) != 2:
                raise AttributeError("Kwarg 'number_nn' in bg_kde_kwargs must have length 2")


class SeppStochasticNnBgFixedTrigger(SeppStochastic):
    bg_kde_class = pp_kde.VariableBandwidthNnKdeSeparable
    trigger_kde_class = pp_kde.FixedBandwidthKde

    def __init_extra__(self, **kwargs):
        """
        Check whether number_nn has been supplied in the KDE kwargs and add default values if not.
        The defaults should depend upon the number of dimensions, but we can't guarantee that data are defined at this
        stage, so just use sensible values. This may break if ndim == 1, but when is that ever the case?
        """
        super(SeppStochasticNnBgFixedTrigger, self).__init_extra__(**kwargs)
        if 'number_nn' not in self.bg_kde_kwargs:
            self.bg_kde_kwargs['number_nn'] = [100, 15]
        if 'bandwidths' not in self.trigger_kde_kwargs:
            self.trigger_kde_class = pp_kde.FixedBandwidthKdeScott
        else:
            if len(self.bg_kde_kwargs['number_nn']) != 2:
                raise AttributeError("Kwarg 'number_nn' in bg_kde_kwargs must have length 2")

class SeppStochasticNnIsotropicTrigger(SeppStochasticNn):

    trigger_kde_class = pp_kde.VariableBandwidthNnRadialKde


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


class SeppDeterministicNnReflected(SeppDeterministicNn):
    trigger_kde_class = pp_kde.WeightedVariableBandwidthNnKdeReflective


class LocalSeppDeterministicNn(SeppDeterministicNn):

    trigger_kde_class = pp_kde.WeightedFixedBandwidthScottKde
    bg_kde_class = pp_kde.WeightedFixedBandwidthScottKdeSeparable

    def __init_extra__(self, **kwargs):
        # disable parallel KDE as all of them will have 'small' amounts of data
        self.trigger_kde_kwargs['parallel'] = False
        if not self.trigger_kde_kwargs.get('tol', None):
            # set default small tolerance to avoid numerical errors
            self.trigger_kde_kwargs['tol'] = 1e-16

        # bundle source indices to ensure at most one call per local trigger KDE
        self.source_groupings = collections.defaultdict(list)
        for (i, t) in enumerate(self.linkage[0]):
            self.source_groupings[t].append(i)

    def set_kdes(self):
        # BG KDE: as in parent
        p_bg = self.p.diagonal()
        self.bg_kde = self.bg_kde_class(self.data, weights=p_bg, **self.bg_kde_kwargs)
        self.num_bg.append(sum(p_bg))

        # one KDE defined for each data point, based on the weightings at those points
        self.trigger_kde = []
        trig_weights_total = 0.
        for i in range(self.ndata):
            # get relevant connections
            # cause_idx = np.where(self.linkage[0] == i)[0]  # this datum is the PARENT
            effect_idx = np.where(self.linkage[1] == i)[0]  # this datum is the CHILD
            # idx = np.concatenate((cause_idx, effect_idx))

            if len(effect_idx) < 2:
                # either only one point, in which case we can't compute a bandwidth, or no points, in which case
                # this point cannot be triggered by any other and must therefore be classified as 100% background
                self.trigger_kde.append(None)
                continue

            # extract interpoint data and weights
            this_interpoint = self.interpoint_data.getrows(effect_idx)
            # this_weights_cause = np.array(self.p[(self.linkage[0][cause_idx], self.linkage[1][cause_idx])].flat)
            # this_weights_effect = np.array(self.p[(self.linkage[0][effect_idx], self.linkage[1][effect_idx])].flat)
            ## should be the same as:
            this_weights_effect = np.array(self.p[(self.linkage[0][effect_idx], np.ones_like(effect_idx) * i)].flat)

            # this_weights_all = np.array(self.p[(self.linkage[0][idx], self.linkage[1][idx])].flat)
            this_weights_all = this_weights_effect

            # renormalise weights based on the total EFFECT density
            # the addition of 'cause' data merely serves to boost the effective number of sources in the KDE, but it
            # shouldn't increase the overall contribution to triggering TO this point
            # we need to maintain the total triggering influx to this datum

            total_effect = this_weights_effect.sum()
            if total_effect < (self.ndata * 1e-12):  # this point must arise from the background
                self.trigger_kde.append(None)
            else:
                trig_weights_total += total_effect
                # this_weights_all = this_weights_all / this_weights_all.sum() * total_effect
                try:
                    self.trigger_kde.append(
                        self.trigger_kde_class(this_interpoint, weights=this_weights_all, **self.trigger_kde_kwargs)
                    )
                except (AttributeError, ValueError) as exc:
                    # only reason to end up here is a 'zero std' error
                    # for now, just ignore this datum.
                    ## FIXME: this is a hack and loses triggering density.
                    # a better fix involves extending the local neighbourhood to second gen. connections
                    print "Unable to generate local KDE for datum %d" % i
                    print repr(exc)
                    self.trigger_kde.append(None)

        self.num_trig.append(trig_weights_total)


    def trigger_density(self, delta_data, source_idx=None):
        """
        :param source_idx: Iterable. For each element of delta_data, source_idx gives the index or indices of the relevant
        source dat(um/a).  The default behaviour assumes that delta_data = self.interpoint_data, so the indices
        are derived directly from linkages.
        """
        res = np.zeros(delta_data.ndata, dtype=float)
        # import ipdb; ipdb.set_trace()

        if source_idx is None:
            source_groupings = self.source_groupings
        else:
            source_groupings = collections.defaultdict(list)
            for (i, t) in enumerate(source_idx):
                if hasattr(t, '__iter__'):
                    x = t
                else:
                    x = [t]
                for tt in x:
                    source_groupings[tt].append(i)

        for t in source_groupings:
            this_delta_data = delta_data.getrows(source_groupings[t])
            if self.trigger_kde[t] is not None:
                this_res = self.trigger_kde[t].pdf(this_delta_data, normed=False)
            else:
                this_res = np.zeros(len(source_groupings[t]))
            res[source_groupings[t]] += this_res

        return res
        #  return res / self.ndata

    def trigger_density_in_place(self, target_data, source_data=None):
        raise NotImplementedError

    def compute_p_l(self):
        # evaluate BG at data points
        m = self.background_density(self.data)

        # evaluate trigger KDE at all interpoint distances
        trigger = self.trigger_density(self.interpoint_data)
        g = sparse.csr_matrix((trigger, self.linkage), shape=(self.ndata, self.ndata))

        # recompute P, making sure to maintain stored zeros
        # NB: summing sparse matrices automatically collapses any zero entries in the result, so don't do it
        l = g.sum(axis=0) + m
        trigger_component = trigger / np.array(l.flat[self.linkage[1]]).squeeze()
        bg_component = np.array(m / l).squeeze()

        # create equivalent to linkage_idx for the diagonal BG entries
        bg_linkage = (range(self.ndata), range(self.ndata))

        # set new P matrix
        new_p = sparse.csr_matrix((bg_component, bg_linkage), shape=(self.ndata, self.ndata))
        new_p[self.linkage] = trigger_component

        # import ipdb; ipdb.set_trace()

        return new_p, l



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
    sepp_obj.train(niter=niter_initial)

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
    sepp_obj.train(niter=niter_initial)

    # start recording kde
    triggers = []
    bgs = []
    weighted_triggers = []
    weighted_bgs = []
    ps = []
    sepp_objs = []

    for i in range(num_repeats):
        sepp_obj.train(niter=niter_initial)
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
