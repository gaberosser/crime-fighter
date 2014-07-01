__author__ = 'gabriel'

import simulate, plotting
from models import PointProcess
from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':

    num_iter = 20

    print "Starting simulation..."
    # simulate data
    c = simulate.MohlerSimulation()
    # c.bg_mu_bar = 1.0
    # c.number_to_prune = 4000
    c.run()
    data = np.array(c.data)[:, :3]  # (t, x, y, b_is_BG)
    ndata = data.shape[0]
    # sort data by time ascending (may be done already?)
    data = data[data[:, 0].argsort()]
    print "Complete"

    r = PointProcess(max_trigger_d=0.75, max_trigger_t=80)
    try:
        r.train(data, niter=num_iter)
    except Exception:
        num_iter = len(r.num_bg)

    ## TODO: analyse the labelling of events: has the trained algorithm correctly identified background events?
    ## has it correctly identified lineage?  Consider simple confusion matrix approach.

    # plots

    # fig A1
    number_bg = c.number_bg
    number_aftershocks = c.number_aftershocks
    fig = plt.figure()
    ax = fig.add_subplot(111)
    h = []
    h.append(ax.plot(range(num_iter), r.num_bg, 'k-'))
    h.append(ax.plot(range(num_iter), number_bg * np.ones(num_iter), 'k--'))
    h.append(ax.plot(range(num_iter), r.num_trig, 'r-'))
    h.append(ax.plot(range(num_iter), number_aftershocks * np.ones(num_iter), 'r--'))
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
