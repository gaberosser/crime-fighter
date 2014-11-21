__author__ = 'gabriel'
import numpy as np
import math
import runner
from utils import augmented_matrix
from analysis import validation
import models
from scipy import sparse
import copy
from data.models import DataArray, CartesianSpaceTimeData


def confusion_matrix(p_inferred, linkage_col, t=0.5):
    """
    Return a standard confusion matrix from an inferred probability matrix
    :param p_inferred: sparse or full probability matrix
    :param linkage_col: as c.data[:, -1], where c is of type MohlerSimulation
    :param t: probability threshold for assignment
    :return: dictionary with counts of FP, etc.
    """

    if p_inferred.shape[0] != p_inferred.shape[1]:
        raise AttributeError("Supplied matrix is not square")

    if sparse.issparse(p_inferred):
        p_inferred = p_inferred.tocsr()
        sum_fn = lambda x: x.sum()
        diag_fun = lambda x: x.diagonal()
    else:
        sum_fn = lambda x: np.sum(x)
        diag_fun = lambda x: np.diag(x)

    bg_idx = np.where(np.isnan(linkage_col))[0]

    tp = 0  # True Positive -> correctly infer trigger and lineage
    tn = 0  # True Negative -> correctly infer background
    fp = 0  # False Positive -> infer trigger when actually background
    fn = 0  # False Negative -> infer background when actually trigger
    ptp = 0  # Partially True Positive -> infer trigger but with incorrect lineage, including when parent is before sample
    ptn = 0  # Partially True Negative -> infer background when actually trigger with parent occurring before sample
    # to reduce to simple confusion matrix, take TP = TP + PTP, TN = TN + PTN

    ## BG
    bg_inferred = diag_fun(p_inferred)
    tn += sum_fn(bg_inferred[bg_idx])
    # deal with false positives below

    ## TRIGGER
    for i in range(p_inferred.shape[0]):
        # upper tri portion
        pi = p_inferred[:i, i]
        pid = p_inferred[i, i]

        # Stop here if event is BG -> FP
        if i in bg_idx:
            fp += sum_fn(pi)
            continue

        # beyond here, event is triggered...

        # Stop here if event is triggered with parent out of sample
        if linkage_col[i] == -1.:
            ptn += pid
            ptp += sum_fn(pi)
            continue

        # inferred as BG -> FN
        fn += pid

        # (fully) true positives
        # sum fun required in case of sparse matrix
        tp += sum_fn(pi[int(linkage_col[i])])

        # partially true positives
        ptp += sum_fn(pi[:int(linkage_col[i])])
        ptp += sum_fn(pi[int(linkage_col[i])+1:])

    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'ptp': ptp,
        'ptn': ptn,
    }


def compute_lineage_matrix(linkage_col):
    """ Compute the Boolean p matrix for annotated data, as returned by the simulator """
    n = linkage_col.size
    bg_idx = np.where((np.isnan(linkage_col)) | (linkage_col < 0))[0]
    trigger_idx_j = np.where((~np.isnan(linkage_col)) & (linkage_col >= 0))[0]
    trigger_idx_i = linkage_col[trigger_idx_j].astype(int)
    p = np.zeros((n, n))
    p[bg_idx, bg_idx] = 1.
    p[trigger_idx_i, trigger_idx_j] = 1.
    if not np.all(np.sum(p, axis=0) == 1.):
        raise AttributeError("Column sum is not equal to one in all cases")
    return p


class SeppValidation(validation.ValidationBase):

    data_class = CartesianSpaceTimeData

    def __init__(self,
                 data,
                 spatial_domain=None,
                 grid_length=None,
                 cutoff_t=None,
                 model_args=None,
                 model_kwargs=None,
                 pp_class=models.SeppStochasticNn):
        """ Thin wrapper for parent's init method, but pp model class is set """
        self.pp_class = pp_class or models.SeppStochasticNn
        super(SeppValidation, self).__init__(data, self.pp_class, spatial_domain=spatial_domain, grid_length=grid_length,
                                           cutoff_t=cutoff_t, model_args=model_args, model_kwargs=model_kwargs)


    def prediction_array(self, t):
        ts = np.ones(self.roc.ngrid) * t
        return self.data_class.from_args(ts, self.centroids[:, 0], self.centroids[:, 1])

    def predict(self, t, **kwargs):
        # estimate total propensity in each grid poly
        # use centroid method for speed
        # background varies in space AND time
        return self.model.predict(self.prediction_array(t), **kwargs)

    def predict_static(self, t, **kwargs):
        # use spatial background only
        return self.model.predict_fixed_background(self.prediction_array(t), **kwargs)

    def predict_bg(self, t, **kwargs):
        # estimate total propensity in each grid poly based on ONLY the background density
        # background varies in space AND time
        return self.model.background_density(self.prediction_array(t), spatial_only=False)

    def predict_bg_static(self, t, **kwargs):
        # estimate total propensity in each grid poly based on ONLY the background density
        # use spatial background only
        return self.model.background_density(self.prediction_array(t), spatial_only=True)

    def predict_trigger(self, t, **kwargs):
        # estimate total propensity in each grid poly based on ONLY the in-place trigger density
        return self.model.trigger_density_in_place(self.prediction_array(t), **kwargs)

    def _update(self, time_step, incremental=False, **train_kwargs):
        print "SEPP _update"
        # take a copy of P now if needed
        if incremental:
            pre_p = self.model.p.copy()

        self.set_t_cutoff(self.cutoff_t + time_step, b_train=False)

        # add new data to model, update linkages and re-estimate p
        self.model.set_data(self.training)
        self.model.set_linkages()
        self.model.initial_estimate()

        # if incremental updates are required, overwrite the previous portion
        if incremental:
            self.model.p = augmented_matrix(self.model.p, pre_p)

        # update time and train
        self.train_model(**train_kwargs)

    def _initial_setup(self, **train_kwargs):
        print "_initial_setup"
        """
        Initial setup for SEPP model.  NB, p matrix has not yet been computed.
        """
        self.train_model(**train_kwargs)

    def _iterate_run(self, pred_dt_plus, true_dt_plus, true_dt_minus, **kwargs):
        print "SEPP _iterate_run"

        true_dt_plus = true_dt_plus or pred_dt_plus
        testing_data = self.testing(dt_plus=true_dt_plus, dt_minus=true_dt_minus)
        self.roc.set_data(testing_data[:, 1:])

        # cycle through the various prediction options and append to results
        # NB: bg and bg_static will give the same result as the predictions are simply scaled
        pred_opt = {
            'full': self.predict,
            'full_static': self.predict_static,
            'bg': self.predict_bg,
            'bg_static': self.predict_bg_static,
            'trigger': self.predict_trigger,
        }
        res = {}

        for app, func in pred_opt.items():
            prediction = func(self.cutoff_t + pred_dt_plus, **kwargs)
            self.roc.set_prediction(prediction)
            this_res = self.roc.evaluate()
            for k, v in this_res.items():
                # append to name to maintain uniqueness
                name = k + '_' + app
                res[name] = v

        # store a copy of the full SEPP model
        # this contains p matrix and KDEs, plus data
        res['model'] = copy.deepcopy(self.model)

        return res

    ## TODO: remove this once the augmented_matrix method has checked out
    def compute_new_p(self, pre_training):
        """ Compute the new initial estimate of p based on the previous value.
        Assumes that the new training set is the old set with additional records. """
        print "SEPP compute_new_p"
        num_old = len(pre_training)
        num_new = len(self.training)
        if (num_new - num_old) < 0:
            raise AttributeError("More records in previous training set than new training set")
        pre_p = self.model.p.tocsc()
        if pre_p.shape[0] != len(pre_training):
            raise AttributeError("Model p matrix has incorrect shape")

        # recompute new P using initial estimator
        new_linkage = self.model._set_linkages_iterated(data=self.training)
        new_p = self.model.estimator(self.training, new_linkage).tocsc()

        # combine old and new indices
        pre_indices = pre_p.indices
        pre_indptr = pre_p.indptr
        new_indices = new_p.indices
        new_indptr = new_p.indptr

        comb_indices = np.concatenate((pre_indices, new_indices[new_indptr[num_old]:]))
        comb_indptr = np.concatenate((pre_indptr[:-1], new_indptr[num_old:]))
        comb_data = np.concatenate((pre_p.data, new_p.data[pre_p.nnz:]))
        comb_p = sparse.csc_matrix((comb_data, comb_indices, comb_indptr), shape=(num_new, num_new)).tocsr()

        return comb_p


class SeppValidationFixedModel(SeppValidation):
    """
    As for parent class, but model is NOT retrained each time, it is assumed to be correct for the duration of the
    validation run.  Probably fairly reasonable - we train these models on tons of data, one extra day isn't going to
    make a significant difference.
    """

    def _update(self, time_step, incremental=False, **train_kwargs):
        print "SEPP _update"
        # take a copy of P now if needed
        if incremental:
            pre_p = self.model.p.copy()

        self.set_t_cutoff(self.cutoff_t + time_step, b_train=False)

    def _initial_setup(self, **train_kwargs):
        """
        Initial setup for SEPP model.
        This is the ONLY time the model is trained
        """
        self.train_model(**train_kwargs)

    def _iterate_run(self, pred_dt_plus, true_dt_plus, true_dt_minus, **kwargs):

        # append source_data kwarg to force the use of new training data each time
        kwargs['source_data'] = CartesianSpaceTimeData(self.training)

        return super(SeppValidationFixedModel, self)._iterate_run(pred_dt_plus,
                                                                  true_dt_plus=true_dt_plus,
                                                                  true_dt_minus=true_dt_minus,
                                                                  **kwargs)


class mock_pp_class():
    def __init__(self, *args, **kwargs):
        pass

class SeppValidationPredefinedModel(SeppValidationFixedModel):

    def __init__(self,
                 data,
                 model,
                 spatial_domain=None,
                 grid_length=None,
                 cutoff_t=None):
        # pass a mock model to parent constructor
        pp_class = mock_pp_class
        super(SeppValidationPredefinedModel, self).__init__(
            data,
            spatial_domain=spatial_domain,
            grid_length=grid_length,
            cutoff_t=cutoff_t,
            pp_class=pp_class)
        self.pp_class = model.__class__
        self.model = model

    def _initial_setup(self, **train_kwargs):
        """
        The model has already been defined so no need to train at all
        """
        pass

    def set_t_cutoff(self, cutoff_t, b_train=False, **kwargs):
        """
        Set cutoff time that divides dataset into training and testing portions.
        Training NEVER happens
        """
        self.cutoff_t = cutoff_t


if __name__ == "__main__":
    from database import logic, models as d_models
    from scipy.stats import multivariate_normal
    from point_process import models as pp_models
    from point_process import simulate, estimation
    from analysis import hotspot
    from analysis import plotting
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    camden = d_models.Division.objects.get(name='Camden')
    xm = 526500
    ym = 186000
    nd = 1000
    # nice normal data
    # data = np.hstack((np.random.normal(loc=5, scale=5, size=(nd, 1)),
    #                   multivariate_normal.rvs(mean=[xm, ym], cov=np.diag([1e5, 1e5]), size=(nd, 1))))

    # moving cluster
    # data = np.hstack((
    #     np.linspace(0, 10, nd).reshape((nd, 1)),
    #     xm + np.linspace(0, 5000, nd).reshape((nd, 1)) + np.random.normal(loc=0, scale=100, size=(nd, 1)),
    #     ym + np.linspace(0, -4000, nd).reshape((nd, 1)) + np.random.normal(loc=0, scale=100, size=(nd, 1)),
    # ))

    # Point process simulated data
    c = simulate.MohlerSimulation()
    c.run()
    data = np.array(c.data)[:, :3]  # (t, x, y, b_is_BG)
    data = data[data[:, 0].argsort()]

    # use Bowers kernel
    # stk = hotspot.STKernelBowers(1, 1e-4)
    # vb = ValidationBase(data, hotspot.Hotspot, camden.mpoly, model_args=(stk,))
    # vb.set_grid(grid_length=200)
    # vb.set_t_cutoff(4.0)

    # use basic historic data spatial hotspot
    # sk = hotspot.SKernelHistoric(2) # use heatmap from final 2 days data
    # vb = ValidationBase(data, hotspot.Hotspot, camden.mpoly, model_args=(sk,))
    # vb.set_grid(grid_length=200)
    # vb.set_t_cutoff(4.0)

    # use Point process learning method
    # vb = SeppValidation(data, model_kwargs={
    #     'max_delta_t': 80,
    #     'max_delta_d': 0.75,
    #     'estimation_function': lambda x, y: estimation.estimator_bowers(x, y, ct=1, cd=10),
    #     })

    vb = SeppValidationFixedModel(data, model_kwargs={
        'max_delta_t': 80,
        'max_delta_d': 0.75,
        'estimation_function': lambda x, y: estimation.estimator_bowers(x, y, ct=1, cd=10),
        })

    vb.set_grid(3)
    vb.set_t_cutoff(400, b_train=False)
    res = vb.run(time_step=5, n_iter=5, train_kwargs={'niter': 10}, verbose=True)

    variants = ['full', 'full_static', 'bg', 'bg_static', 'trigger']
    colours = ['k', 'r', 'b', 'y', 'g']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    fig = plt.figure()
    ax2 = fig.add_subplot(111)

    for v, col in zip(variants, colours):
        x = np.mean(np.array(res['cumulative_area_' + v]), axis=0)
        y = np.array(res['cumulative_crime_' + v])
        ax1.fill_between(x, np.min(y, axis=0), np.max(y, axis=0), edgecolor='none', facecolor=col, alpha=0.4)
        ax2.plot(x, np.mean(y, axis=0), col)
        mp = np.mean(np.array(res['pai_' + v]), axis=0)