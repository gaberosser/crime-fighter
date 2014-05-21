__author__ = 'gabriel'

import estimation, simulate, plotting
from kde.methods import pure_python as pp_kde
import numpy as np
from time import time
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

class PointProcess(object):
    def __init__(self, data, p=None):
        self.data = np.array(data)
        # sort by time
        self.data = self.data[self.data[:, 0].argsort()]
        if p:
            self.p = p
            self.pset = True
        else:
            self.p = np.zeros((self.ndata, self.ndata))
            self.pset = False
        self.run_times = []
        self.num_bg = []
        self.num_trig = []
        self.l2_differences = []
        self.bg_t_kde = None
        self.bg_xy_kde = None
        self.trigger_kde = None

    @property
    def ndata(self):
        return self.data.shape[0]

    def run(self, niter=30, verbose=True):
        # precompute error norm denominator
        err_denom = float(self.ndata * (self.ndata + 1)) / 2.

        # initial estimate for p if required
        if not self.pset:
            self.p = estimation.initial_guess_educated(self.data)

        for i in range(niter):

            tic = time()

            # sanity check
            colsum = np.sum(self.p, axis=0)
            if np.any((colsum < (1 - 1e-12)) | (colsum > (1 + 1e-12))):
                raise AttributeError("Matrix P failed requirement that columns sum to 1 within tolerance.")
            if np.any(np.tril(self.p, k=-1) != 0.):
                raise AttributeError("Matrix P failed requirement that lower diagonal is zero.")

            bg, interpoint = estimation.sample_bg_and_interpoint(self.data, self.p)
            self.num_bg.append(bg.shape[0])
            self.num_trig.append(interpoint.shape[0])

            # compute KDEs
            self.bg_t_kde = pp_kde.VariableBandwidthNnKde(bg[:, 0], normed=False)
            self.bg_xy_kde = pp_kde.VariableBandwidthNnKde(bg[:, 1:], normed=False)
            self.trigger_kde = pp_kde.VariableBandwidthNnKde(interpoint, normed=False)

            # evaluate BG at data points
            m_xy = self.bg_xy_kde.pdf(self.data[:, 1], self.data[:, 2])
            m_t = self.bg_t_kde.pdf(self.data[:, 0])
            m = m_xy * m_t

            # evaluate trigger KDE at all interpoint distances
            g = estimation.evaluate_trigger_kde(self.trigger_kde, self.data, tol=0.99, ngrid=100)

            # sanity check
            if np.any(np.diagonal(g) != 0):
                raise AttributeError("Non-zero diagonal values found in g.")

            # recompute P
            l = np.sum(g, axis=0) + m
            new_p = (m / l) * np.eye(self.ndata) + (g / l)

            # compute difference
            q = new_p - self.p
            self.l2_differences.append(np.sqrt(np.sum(q**2)) / err_denom)

            # update p
            self.p = new_p

            # record time taken
            self.run_times.append(time() - tic)
            if verbose:
                print "Completed %d / %d iterations in %f s" % (i+1, niter, self.run_times[-1])


if __name__ == '__main__':

    num_iter = 50

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

    r = PointProcess(data)
    r.run(num_iter)

    # precompute error norm denominator
    # err_denom = float(ndata*(ndata + 1)) / 2.

    # P = np.zeros((ndata, ndata, num_iter + 1))
    # k_bgt = []
    # k_bgxy = []
    # k_ash = []
    # l2_errors = []
    # n_bg = []
    # n_ash = []
    #
    # P[:, :, 0] = estimation.initial_guess_educated(data)
    # # start main simulation loop
    # for i in range(num_iter):
    #     print "Iteration", i
    #     tic = time()
    #     # sampling
    #     bg, interpoint = estimation.sample_bg_and_interpoint(data, P[:, :, i])
    #     n_bg.append(bg.shape[0])
    #     n_ash.append(interpoint.shape[0])
    #
    #     # compute KDEs
    #     # BG
    #     bg_t_kde = pp_kde.VariableBandwidthKde(bg[:, 0], normed=False)
    #     bg_xy_kde = pp_kde.VariableBandwidthKde(bg[:, 1:], normed=False)
    #     # interpoint / trigger KDE
    #     trigger_kde = pp_kde.VariableBandwidthKde(interpoint, normed=False)
    #     k_bgt.append(bg_t_kde)
    #     k_bgxy.append(bg_xy_kde)
    #     k_ash.append(trigger_kde)
    #
    #     # evaluate BG at data points
    #     m_xy = bg_xy_kde.pdf(data[:, 1], data[:, 2])
    #     m_t = bg_t_kde.pdf(data[:, 0])
    #     m = m_xy * m_t
    #
    #     # evaluate trigger KDE
    #     g = estimation.evaluate_trigger_kde(trigger_kde, data, tol=0.99, ngrid=100)
    #
    #     # sanity check
    #     if np.any(g[range(ndata), range(ndata)] != 0):
    #         raise AttributeError("Non-zero diagonal values found in g.")
    #
    #     # recompute P
    #     l = np.sum(g, axis=0) + m
    #     P[:, :, i+1] = (m / l) * np.eye(ndata) + (g / l)
    #
    #     # sanity check
    #     eps = 1e-12
    #     colsum = np.sum(P[:, :, i+1], axis=0)
    #     if np.any((colsum < (1 - eps)) | (colsum > (1 + eps))):
    #         raise AttributeError("Matrix P failed requirement that columns sum to 1 within tolerance.")
    #     if np.any(np.tril(P[:, :, i+1], k=-1) != 0.):
    #         raise AttributeError("Matrix P failed requirement that lower diagonal is zero.")
    #
    #     # error analysis between iterations
    #     q = P[:, :, i+1] - P[:, :, i]
    #     l2_errors.append(np.sqrt(np.sum(q**2)) / err_denom)
    #     print "Completed in %f s" % (time() - tic)


    # plots

    # fig A1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    h = []
    h.append(ax.plot(range(num_iter), r.num_bg, 'k-'))
    h.append(ax.plot(range(num_iter), c.number_bg * np.ones(num_iter), 'k--'))
    h.append(ax.plot(range(num_iter), r.num_trig, 'r-'))
    h.append(ax.plot(range(num_iter), c.number_aftershocks * np.ones(num_iter), 'r--'))
    ax.set_xlabel('Number iterations')
    ax.set_ylabel('Number events')
    ax.legend([t[0] for t in h], ('B/g, inferred', 'B/g, true', 'Trig, inferred', 'Trig, true'), 'right')

    # fig A2
    t = np.linspace(0, 60, 200)
    w = c.off_omega
    th = c.off_theta
    z = th * w * np.exp(-w * t)
    fig = plotting.plot_txy_t_marginals(r.trigger_kde, t_max=60)
    plt.plot(t, z, 'k--')
    ax = fig.gca()
    ax.set_ylim([0, w * 1.02])
    ax.legend(ax.get_lines(), ('Inferred', 'True'), 'upper right')

    t = np.linspace(-0.05, 0.05, 200)
    sx = c.off_sigma_x
    z = 1/(np.sqrt(2 * np.pi) * sx) * np.exp(-(t**2) / (2 * sx**2))
    fig = plotting.plot_txy_x_marginals(r.trigger_kde, x_max=0.05)
    plt.plot(t, z, 'k--')
    ax = fig.gca()
    ax.set_ylim([0, 1.05/(np.sqrt(2 * np.pi) * sx)])
    ax.set_xlim([-0.05, 0.05])
    ax.legend(ax.get_lines(), ('Inferred', 'True'), 'upper right')

    t = np.linspace(-0.5, 0.5, 200)
    sy = c.off_sigma_y
    z = 1/(np.sqrt(2 * np.pi) * sy) * np.exp(-(t**2) / (2 * sy**2))
    fig = plotting.plot_txy_y_marginals(r.trigger_kde, y_max=0.5)
    ax = fig.gca()
    line = ax.get_lines()[0]
    plt.plot(t, z, 'k--')
    ax = fig.gca()
    ymax_theor = 1.05/(np.sqrt(2 * np.pi) * sy)
    ymax_infer = 1.05 * max(line.get_ydata())
    ax.set_ylim([0, max(ymax_infer, ymax_theor)])
    ax.set_xlim([-0.5, 0.5])
    ax.legend(ax.get_lines(), ('Inferred', 'True'), 'upper right')
