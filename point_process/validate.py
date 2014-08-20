__author__ = 'gabriel'
import numpy as np
import math
import runner
from analysis import validation
import models
from scipy import sparse
import copy


def confusion_matrix(p_inferred, linkage_col, t=0.5):

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


class PpValidation(validation.ValidationBase):

    pp_class = models.PointProcess # model class

    def __init__(self, data, spatial_domain=None, grid_length=None, tmax_initial=None, model_args=None, model_kwargs=None):
        """ Thin wrapper for parent's init method, but pp model class is set """
        super(PpValidation, self).__init__(data, self.pp_class, spatial_domain=spatial_domain, grid_length=grid_length,
                                           tmax_initial=tmax_initial, model_args=model_args, model_kwargs=model_kwargs)

    def predict(self, t, **kwargs):
        print "SEPP predict"
        # estimate total propensity in each grid poly
        # use centroid method for speed
        # use spatial background only to avoid background 'fade out'
        ts = np.ones(self.roc.ngrid) * t
        return self.model.predict_fixed_background(ts, self.centroids[:, 0], self.centroids[:, 1])

    def _update(self, time_step, **train_kwargs):
        print "SEPP _update"
        pre_training = self.training
        self.set_t_cutoff(self.cutoff_t + time_step, b_train=False)
        # update p based on previous
        self.model.p = self.compute_new_p(pre_training)
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
        # conventional assessment
        res = super(PpValidation, self)._iterate_run(pred_dt_plus, true_dt_plus, true_dt_minus, **kwargs)
        # also store p matrix and KDEs
        # res['p'] = self.model.p
        # res['trigger_kde'] = copy.deepcopy(self.model.trigger_kde)
        # res['bg_t_kde'] = copy.deepcopy(self.model.bg_t_kde)
        # res['bg_xy_kde'] = copy.deepcopy(self.model.bg_xy_kde)
        # store a copy full SEPP model
        # this contains p matrix and KDEs, plus data
        res['model'] = copy.deepcopy(self.model)

        return res

    def compute_new_p(self, pre_training):
        """ Compute the new initial estimate of p based on the previous value.
        Assumes that the new training set is the old set with additional records. """
        print "SEPP compute_new_p"
        num_old = len(pre_training)
        num_new = len(self.training)
        if (num_new - num_old) < 0:
            raise AttributeError("More records in previous training set than new training set")
        new_recs = self.training[num_old:]
        pre_p = self.model.p
        if pre_p.shape[0] != len(pre_training):
            raise AttributeError("Model p matrix has incorrect shape")
        # recompute new P using initial estimator
        new_linkage = self.model._set_linkages_iterated(data=self.training)
        new_p = self.model.estimator(self.training, new_linkage)
        # restore former probs
        new_p[:num_old, :num_old] = pre_p
        return new_p


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
    vb = PpValidation(data, model_kwargs={
        'max_trigger_t': 80,
        'max_trigger_d': 0.75,
        'estimator': lambda x, y: estimation.estimator_bowers(x, y, ct=1, cd=10),
        })
    vb.set_grid(grid_length=3)
    vb.set_t_cutoff(400, b_train=False)
    res = vb.run(time_step=5, t_upper=450, train_kwargs={'niter': 15}, verbose=True)

    mc = np.mean(np.array(res['carea']), axis=0)
    mf = np.mean(np.array(res['cfrac']), axis=0)
    mp = np.mean(np.array(res['pai']), axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.array(res['carea']).transpose(), np.array(res['cfrac']).transpose(), color='k', alpha=0.2)
    ax.plot(mc, mf, 'r-')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.array(res['carea']).transpose(), np.array(res['pai']).transpose(), color='k', alpha=0.2)
    ax.plot(mc, mp, 'r-')