__author__ = 'gabriel'

import simulate
import plotting
from point_process import models, estimation
from matplotlib import pyplot as plt
import numpy as np
from scipy import sparse


def initial_simulation():
    print "Starting simulation..."
    # simulate data
    c = simulate.MohlerSimulation()
    c.seed(42)
    # c.bg_mu_bar = 1.0
    # c.number_to_prune = 4000
    c.run()
    data = np.array(c.data)[:, :3]  # (t, x, y, b_is_BG)
    # sort data by time ascending (may be done already?)
    data = data[data[:, 0].argsort()]
    print "Complete"
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

    r = models.PointProcess(max_trigger_d=0.75, max_trigger_t=80)
    r.train(data, niter=20, tol_p=1e-5)

    return r


if __name__ == '__main__':

    num_iter = 20
    c, data = initial_simulation()
    ndata = data.shape[0]

    # set estimation seed for consistency
    models.estimation.set_seed(42)
    # r = models.PointProcessStochastic(max_trigger_d=0.75, max_trigger_t=80, min_bandwidth=[1., .05, .05])
    # r = models.PointProcessStochasticNn(max_trigger_d=0.75, max_trigger_t=80)
    r = models.PointProcessDeterministicNn(max_trigger_d=0.75, max_trigger_t=80)
    # r = models.PointProcessDeterministicFixedBandwidth(max_trigger_d=0.75, max_trigger_t=80, min_bandwidth=[1., .05, .05])

    try:
        r.train(data, niter=num_iter, tol_p=1e-5)
    except KeyboardInterrupt:
        num_iter = len(r.num_bg)

    # r = noisy_init(c)
    # num_iter = len(r.num_bg)

    # plots
    plotting.multiplots(r, c)