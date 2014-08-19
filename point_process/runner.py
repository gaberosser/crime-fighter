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
    r = models.PointProcess(max_trigger_d=0.75, max_trigger_t=80)
    try:
        r.train(data, niter=num_iter, tol_p=1e-5)
    except Exception:
        num_iter = len(r.num_bg)

    # r = noisy_init(c)
    # num_iter = len(r.num_bg)

    # plots

    # fig A1
    number_bg = c.number_bg
    number_aftershocks = c.number_aftershocks
    fig = plt.figure()
    ax = fig.add_subplot(111)
    h = []
    h.append(ax.plot(range(r.niter), r.num_bg, 'k-'))
    h.append(ax.plot(range(r.niter), c.number_bg * np.ones(r.niter), 'k--'))
    h.append(ax.plot(range(r.niter), r.num_trig, 'r-'))
    h.append(ax.plot(range(r.niter), c.number_aftershocks * np.ones(r.niter), 'r--'))
    ax.set_xlabel('Number iterations')
    ax.set_ylabel('Number events')
    ax.legend([t[0] for t in h], ('B/g, inferred', 'B/g, true', 'Trig, inferred', 'Trig, true'), 'right')

    # fig A2
    t = np.linspace(0, 60, 200)
    w = c.off_omega
    th = c.off_theta
    zt = th * w * np.exp(-w * t)
    fig = plotting.plot_txy_t_marginals(r.trigger_kde, norm=r.ndata, t_max=60)
    plt.plot(t, zt, 'k--')
    ax = fig.gca()
    ax.set_ylim([0, w * th * 1.02])
    ax.legend(ax.get_lines(), ('Inferred', 'True'), 'upper right')

    x = np.linspace(-0.05, 0.05, 200)
    sx = c.off_sigma_x
    zx = th / (np.sqrt(2 * np.pi) * sx) * np.exp(-(x**2) / (2 * sx**2))
    fig = plotting.plot_txy_x_marginals(r.trigger_kde, norm=r.ndata, x_max=0.05)
    plt.plot(x, zx, 'k--')
    ax = fig.gca()
    ax.set_ylim([0, 1.05 * th / (np.sqrt(2 * np.pi) * sx)])
    ax.set_xlim([-0.05, 0.05])
    ax.legend(ax.get_lines(), ('Inferred', 'True'), 'upper right')

    y = np.linspace(-0.5, 0.5, 200)
    sy = c.off_sigma_y
    zy = th/(np.sqrt(2 * np.pi) * sy) * np.exp(-(y**2) / (2 * sy**2))
    fig = plotting.plot_txy_y_marginals(r.trigger_kde, norm=r.ndata, y_max=0.5)
    ax = fig.gca()
    line = ax.get_lines()[0]
    plt.plot(y, zy, 'k--')
    ax = fig.gca()
    ymax_theor = th/(np.sqrt(2 * np.pi) * sy)
    ymax_infer = max(line.get_ydata())
    ax.set_ylim([0, 1.05 * max(ymax_infer, ymax_theor)])
    ax.set_xlim([-0.5, 0.5])
    ax.legend(ax.get_lines(), ('Inferred', 'True'), 'upper right')
