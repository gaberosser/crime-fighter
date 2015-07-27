from validation import validation

__author__ = 'gabriel'
import numpy as np
from utils import augmented_matrix
import models
from scipy import sparse
import copy
import pickle
import re
import importlib
from data.models import CartesianSpaceTimeData


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

    def predict_all(self, t, include=None, **kwargs):
        """
        Carry out all prediction methods at time t.  Doing this in one go means computing linkages only once per call.
        :param t: Time at which to make the prediction
        :param include: List of methods to include. If None, all methods are included
        :return: Dictionary, values correspond to the prediction output array, keys describe the method used
        """

        include_dict = {
            'full': lambda bgd, bgs, tr: bgd + tr,
            'full_static': lambda bgd, bgs, tr: bgs + tr,
            'bg': lambda bgd, bgs, tr: bgd,
            'bg_static': lambda bgd, bgs, tr: bgs,
            'trigger': lambda bgd, bgs, tr: tr,
        }

        if include is None:
            include = include_dict.keys()
        else:
            assert len(include), "No methods requested for predict/assess cycle."

        target_data = self.prediction_array(t)
        bg_dynamic = None
        bg_static = None
        trigger = None
        if 'full' in include or 'bg' in include:
            bg_dynamic = self.model.background_density(target_data, spatial_only=False)
        if 'full_static' in include or 'bg_static' in include:
            bg_static = self.model.background_density(target_data, spatial_only=True)
        if 'full' in include or 'full_static' in include or 'trigger' in include:
            ## NB set the new source data here!
            trigger = self.model.trigger_density_in_place(target_data, source_data=self.training)

        return dict([(meth, include_dict[meth](bg_dynamic, bg_static, trigger)) for meth in include])

    def _update(self, time_step, incremental=False, **train_kwargs):
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
        """
        Initial setup for SEPP model.  NB, p matrix has not yet been computed.
        """
        self.train_model(**train_kwargs)

    def _iterate_run(self, pred_dt_plus, true_dt_plus, true_dt_minus, **kwargs):

        true_dt_plus = true_dt_plus or pred_dt_plus
        testing_data = self.testing(dt_plus=true_dt_plus, dt_minus=true_dt_minus)
        testing_ind = self.testing_data_index(dt_plus=true_dt_plus, dt_minus=true_dt_minus)
        self.roc.set_data(testing_data[:, 1:], index=testing_ind)

        res = {}

        # cycle through the various prediction options and append to results
        # NB: bg and bg_static will give the same result as the predictions are simply scaled
        # select the methods to use by adding an 'include' variable to kwargs, otherwise all methods are run by default
        all_predictions = self.predict_all(self.cutoff_t + pred_dt_plus, **kwargs)
        for pred_method, pred_values in all_predictions.items():
            self.roc.set_prediction(pred_values)
            this_res = self.roc.evaluate()
            res[pred_method] = this_res

        # store a copy of the full SEPP model
        # this contains p matrix and KDEs, plus data
        res['model'] = copy.deepcopy(self.model)

        return res

    ## TODO: remove this once the augmented_matrix method has checked out
    def compute_new_p(self, pre_training):
        """ Compute the new initial estimate of p based on the previous value.
        Assumes that the new training set is the old set with additional records. """
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

    def post_process(self, res):
        methods = ['full', 'full_static', 'bg', 'bg_static', 'trigger']

        # restructure results
        # only need one copy of cumulative_crime_max
        # these are all the same, as they are independent of the prediction
        ccm = None
        for m in methods:
            if m in res:
                ccm = res[m][0]['cumulative_crime_max']
                break
        assert ccm is not None, "Unable to find results for any of the methods."

        for m in methods:
            if m not in res:
                # method not included in this run; skip
                continue
            this_res = {}
            for k in res[m][0].keys():
                if k == 'cumulative_crime_max':
                    pass
                else:
                    # fix errors upon finding missing data by providing default NaN get argument
                    val = np.array(
                        [res[m][i].get(k, np.zeros(self.roc.n_sample_units) * np.nan) for i in range(len(res[m]))]
                    )
                    this_res[k] = val
            # overwrite with new restructured results
            res[m] = this_res
        res['cumulative_crime_max'] = ccm


class SeppValidationIntegration(validation.ValidationIntegration, SeppValidation):
    pass


class SeppValidationFixedModel(SeppValidation):
    """
    As for parent class, but model is NOT retrained each time, it is assumed to be correct for the duration of the
    validation run.  Probably fairly reasonable - we train these models on tons of data, one extra day isn't going to
    make a significant difference.
    """

    def _update(self, time_step, incremental=False, **train_kwargs):
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

    def post_process(self, res):
        super(SeppValidationFixedModel, self).post_process(res)
        # overwrite all models with the first entry since they are all identical - copies are wasteful
        # maintain a list because this way the repeat_run code is simpler
        for i in range(len(res['model'])):
            res['model'][i] = res['model'][0]


class SeppValidationFixedModelIntegration(validation.ValidationIntegration, SeppValidationFixedModel):
    pass


class mock_pp_class():
    def __init__(self, *args, **kwargs):
        pass


class SeppValidationPreTrainedModel(SeppValidationFixedModel):
    """
    As for parent class, except that the model is supplied already trained, so that the validation can
    commence immediately with the predict-assess run.
    """

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


class SeppValidationPreTrainedModelIntegration(validation.ValidationIntegration, SeppValidationPreTrainedModel):
    pass


def validate_pickled_model(filename,
                           sample_unit_size,
                           n_sample_per_grid=20,
                           time_step=1,
                           n_iter=100,
                           validation_class=SeppValidationPreTrainedModelIntegration,
                           domain=None,
                           cutoff_t=None,
                           include_predictions=False,
                           pred_kwargs=None,
                           train_kwargs=None):
    # load and instantiate
    with open(filename, 'r') as f:
        obj = pickle.load(f)
        cls = re.sub(r"^<class '(.*)'>$", r"\1", obj['class'])
        module = '.'.join(cls.split('.')[:-1])
        cls = cls.split('.')[-1]
    if module == '':
        module = importlib.import_module('models', package='.')
    else:
        module = importlib.import_module(module)
    cls = getattr(module, cls)

    obj = cls.from_pickle(filename)
    vb = validation_class(data=obj.data,
                          model=obj,
                          spatial_domain=domain,
                          cutoff_t=cutoff_t,
                          include_predictions=include_predictions)
    vb.set_sample_units(sample_unit_size, n_sample_per_grid=n_sample_per_grid)
    res = vb.run(time_step,
                 n_iter=n_iter,
                 pred_kwargs=pred_kwargs,
                 train_kwargs=train_kwargs)
    return vb, res


if __name__ == "__main__":
    from database import models as d_models
    from point_process import simulate, estimation
    from matplotlib import pyplot as plt
    # camden = d_models.Division.objects.get(name='Camden')
    # xm = 526500
    # ym = 186000
    # nd = 1000
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
    # vb.set_sample_units(grid_length=200)
    # vb.set_t_cutoff(4.0)

    # use basic historic data spatial hotspot
    # sk = hotspot.SKernelHistoric(2) # use heatmap from final 2 days data
    # vb = ValidationBase(data, hotspot.Hotspot, camden.mpoly, model_args=(sk,))
    # vb.set_sample_units(grid_length=200)
    # vb.set_t_cutoff(4.0)

    # use Point process learning method
    # vb = SeppValidation(data, model_kwargs={
    #     'max_delta_t': 80,
    #     'max_delta_d': 0.75,
    #     'estimation_function': lambda x, y: estimation.estimator_bowers(x, y, ct=1, cd=10),
    #     })

    sepp = models.SeppStochasticNn(max_delta_t=80,
                                   max_delta_d=0.75,
                                   estimation_function=lambda x, y: estimation.estimator_bowers(x, y, ct=1, cd=10))

    vb = SeppValidationFixedModel(data, sepp)

    pred_kwargs = {
        'include': ('full', 'full_static', 'bg', 'trigger')
    }

    vb.set_sample_units(3, n_sample_per_grid=10)
    vb.set_t_cutoff(400, b_train=False)
    res = vb.run(time_step=5, n_iter=5, train_kwargs={'niter': 10}, verbose=True, pred_kwargs=pred_kwargs)

    from point_process import plotting
    plotting.validation_multiplot(res)