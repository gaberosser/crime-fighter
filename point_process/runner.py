__author__ = 'gabriel'

import simulate
import plotting
from point_process import models, estimation
from matplotlib import pyplot as plt
import numpy as np
from scipy import sparse
from data.models import DataArray
from copy import deepcopy
import logging


def initial_simulation(t_total=None):
    # simulate data
    c = simulate.MohlerSimulation()
    c.seed(42)
    if t_total:
        c.t_total = t_total
    c.run()
    data = np.array(c.data)[:, :3]  # (t, x, y, b_is_BG)
    return c, data


def noisy_init(c, noise_level=0.):
    ndata = c.data.shape[0]

    # make 'perfect' init matrix
    p_init = sparse.csr_matrix((ndata, ndata))
    bg_map = np.isnan(c.data[:, -1]) | (c.data[:, -1] < 0)
    bg_idx = np.where(bg_map)[0]
    effect_idx = np.where(~bg_map)[0]
    cause_idx = c.data[effect_idx, -1].astype(int)
    p_init[bg_idx, bg_idx] = 1.
    p_init[cause_idx, effect_idx] = 1.

    if noise_level > 0.:
        ## FIXME: too slow, need to apply noise to a subselection of elements
        noise = sparse.csr_matrix(np.abs(np.random.normal(loc=0.0, scale=noise_level, size=(ndata, ndata))))
        p_init = p_init + noise
        colsum = p_init.sum(axis=0).flat
        for i in range(ndata):
            p_init[:, i] = p_init[:, i] / colsum[i]

    r = models.SeppStochasticNn(max_delta_d=0.75, max_delta_t=80)
    r.train(data=data, niter=20, tol_p=1e-5)

    return r


def consistency_of_trigger_after_convergence(niter_initial=15, niter_after=30, sepp_class=models.SeppStochasticNn,
                                             plot=True):
    c, data = initial_simulation()
    models.estimation.set_seed(42)
    bg_kde_kwargs = {
        'number_nn': [101, 16],
    }

    trigger_kde_kwargs = {
        # 'min_bandwidth': [1., .005, .05],
        'number_nn': 15,
    }

    max_delta_t = 80
    max_delta_d = 1

    r = sepp_class(data=data, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
                   bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)

    r.p = estimation.estimator_bowers(data, r.linkage)
    r.train(niter=niter_initial)

    # start recording kde
    triggers = [r.trigger_kde]
    bgs = [r.bg_kde]
    weighted_triggers = [r.weighted_trigger_kde]
    weighted_bgs = [r.weighted_bg_kde]

    for i in range(niter_after):
        r._iterate()
        triggers.append(r.trigger_kde)
        bgs.append(r.bg_kde)
        weighted_triggers.append(r.weighted_trigger_kde)
        weighted_bgs.append(r.weighted_bg_kde)

    if plot:
        # envelope plots
        # trigger t
        t = np.linspace(0, max_delta_t, 500)
        gt = np.array([a.marginal_pdf(t, dim=0, normed=False) / float(r.ndata) for a in triggers])
        w = c.off_omega
        th = c.off_theta
        gt_true = th * w * np.exp(-w * t)
        plt.figure()
        plt.fill_between(t, np.min(gt, axis=0), np.max(gt, axis=0), edgecolor='none', facecolor='k', alpha=0.5)
        plt.plot(t, gt_true, 'k--')

        # trigger x
        x = np.linspace(-.06, .06, 500)
        gx = np.array([a.marginal_pdf(x, dim=1, normed=False) / float(r.ndata) for a in triggers])
        th = c.off_theta
        sx = c.off_sigma_x
        gx_true = th / (np.sqrt(2 * np.pi) * sx) * np.exp(-(x**2) / (2 * sx**2))
        plt.figure()
        plt.fill_between(x, np.min(gx, axis=0), np.max(gx, axis=0), edgecolor='none', facecolor='k', alpha=0.5)
        plt.plot(x, gx_true, 'k--')
        plt.xlim([np.min(x), np.max(x)])
        plt.ylim([0, 1.02 * max(max(gx_true), np.max(gx))])

        # background
        bx = c.bg_sigma
        xy = DataArray.from_meshgrid(*np.meshgrid(*[np.linspace(-3 * bx, 3 * bx, 100)] * 2))
        fxy = np.array([a.partial_marginal_pdf(xy) for a in bgs])
        fxy_mean = fxy.mean(axis=0)
        fxy_range = fxy.ptp(axis=0) / fxy_mean
        wfxy = np.array([a.partial_marginal_pdf(xy) for a in weighted_bgs])
        wfxy_mean = wfxy.mean(axis=0)
        wfxy_range = wfxy.ptp(axis=0) / wfxy_mean

        plt.figure()
        plt.contourf(xy.toarray(0), xy.toarray(1), fxy_mean, 40)
        plt.colorbar()

        plt.figure()
        plt.contourf(xy.toarray(0), xy.toarray(1), fxy_range, 40)
        plt.colorbar()

        plt.figure()
        plt.contourf(xy.toarray(0), xy.toarray(1), wfxy_mean, 40)
        plt.colorbar()

        # look at the scale here - over 10 times lower variability
        plt.figure()
        plt.contourf(xy.toarray(0), xy.toarray(1), wfxy_range, 40)
        plt.colorbar()

    return r, triggers, bgs, weighted_triggers, weighted_bgs


def consistency_of_trigger_at_convergence(nrepeat=20, niter=15, sepp_class=models.SeppStochasticNn, plot=True):
    c, data = initial_simulation()
    models.estimation.set_seed(42)
    bg_kde_kwargs = {
        'number_nn': [101, 16],
    }

    trigger_kde_kwargs = {
        # 'min_bandwidth': [1., .005, .05],
        'number_nn': 15,
    }

    max_delta_t = 80
    max_delta_d = 1

    r = sepp_class(data=data, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
                   bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)

    p = estimation.estimator_bowers(data, r.linkage)

    triggers = []
    bgs = []
    weighted_triggers = []
    weighted_bgs = []
    sepps = []

    for i in range(nrepeat):
        r.reset()
        r.p = p
        r.train(data, niter=niter)
        # record kdes
        triggers.append(r.trigger_kde)
        bgs.append(r.bg_kde)
        weighted_triggers.append(r.weighted_trigger_kde)
        weighted_bgs.append(r.weighted_bg_kde)
        sepps.append(deepcopy(r))

    return sepps, triggers, bgs, weighted_triggers, weighted_bgs


if __name__ == '__main__':

    logger = logging.getLogger('kde.models')

    num_iter = 25
    parallel = True
    t_total = None
    # c, data = initial_simulation(t_total=t_total)
    # max_delta_t = 100
    # max_delta_d = 0.75

    c = simulate.MySimulation1()
    c.t_total = 1000
    c.num_to_prune = 2000  # should leave ~2000 datapoints
    c.run()
    data = c.data[:, :3]
    max_delta_t = 100
    max_delta_d = 0.75
    # init_est_params = {
    #     'ct': 1/15.,
    #     'cd': 4.,
    # }
    init_est_params = {
        'ct': 10,
        'cd': .1,
    }


    ndata = data.shape[0]

    bg_kde_kwargs = {
        'number_nn': [100, 15],
        'strict': False,
    }

    trigger_kde_kwargs = {
        # 'min_bandwidth': [1., .005, .05],
        'number_nn': 15,
        'strict': False,
    }



    r = models.SeppStochasticNn(data=data, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
                                bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)
    # r = models.SeppStochasticStationaryBg(data=data, max_delta_d=max_delta_d, max_delta_t=max_delta_t)
    # r = models.SeppStochasticNnStExp(data=data, max_delta_d=0.75, max_delta_t=80,
    #                             bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)
    # r = models.SeppStochasticNnSt(data=data, max_delta_d=0.75, max_delta_t=80,
    #                             bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)
    # r = models.SeppStochasticNnReflected(data=data, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
    #                                     bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)
    # r = models.SeppStochasticNnOneSided(data=data, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
    #                                     bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)
    # r = models.SeppDeterministicNn(data=data, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
    #                                bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)
    # r = models.SeppDeterministicNnReflected(data=data, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
    #                                bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs)



    # p = estimation.estimator_bowers(data, r.linkage, **init_est_params)
    p = estimation.estimator_exp_gaussian(data, r.linkage, **init_est_params)
    r.p = p

    # set seed for consistency
    r.set_seed(42)

    try:
        r.train(niter=num_iter)
    except KeyboardInterrupt:
        num_iter = len(r.num_bg)

    # r = noisy_init(c)
    # num_iter = len(r.num_bg)

    # plots
    plotting.multiplots(r, c)
