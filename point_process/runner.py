__author__ = 'gabriel'

import estimation, simulate, plotting
from kde.methods import pure_python as pp_kde
import numpy as np
from time import time
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

num_iter = 30

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


# precompute error norm denominator
err_denom = float(ndata*(ndata + 1)) / 2.

P = np.zeros((ndata, ndata, num_iter + 1))
k_bgt = []
k_bgxy = []
k_ash = []
l2_errors = []
n_bg = []
n_ash = []

P[:, :, 0] = estimation.initial_guess_educated(data)
# start main simulation loop
for i in range(num_iter):
    print "Iteration", i
    tic = time()
    # sampling
    bg, interpoint = estimation.sample_bg_and_interpoint(data, P[:, :, i])
    n_bg.append(bg.shape[0])
    n_ash.append(interpoint.shape[0])

    # compute KDEs
    # BG
    bg_t_kde = pp_kde.VariableBandwidthKde(bg[:, 0])
    bg_xy_kde = pp_kde.VariableBandwidthKde(bg[:, 1:])
    # interpoint / trigger KDE
    trigger_kde = pp_kde.VariableBandwidthKde(interpoint)
    k_bgt.append(bg_t_kde)
    k_bgxy.append(bg_xy_kde)
    k_ash.append(trigger_kde)

    # evaluate BG at data points
    m_xy = bg_xy_kde.pdf(data[:, 1], data[:, 2])
    m_t = bg_t_kde.pdf(data[:, 0])
    m = m_xy * m_t

    # evaluate trigger KDE
    g = estimation.evaluate_trigger_kde(trigger_kde, data, tol=0.95, ngrid=100)

    # sanity check
    if np.any(g[range(ndata), range(ndata)] != 0):
        raise AttributeError("Non-zero diagonal values found in g.")

    # recompute P
    l = np.sum(g, axis=0) + m
    P[:, :, i+1] = (m / l) * np.eye(ndata) + (g / l)

    # sanity check
    eps = 1e-12
    colsum = np.sum(P[:, :, i+1], axis=0)
    if np.any((colsum < (1 - eps)) | (colsum > (1 + eps))):
        raise AttributeError("Matrix P failed requirement that columns sum to 1 within tolerance.")
    if np.any(np.tril(P[:, :, i+1], k=-1) != 0.):
        raise AttributeError("Matrix P failed requirement that lower diagonal is zero.")

    # error analysis between iterations
    q = P[:, :, i+1] - P[:, :, i]
    l2_errors.append(np.sqrt(np.sum(q**2)) / err_denom)
    print "Completed in %f s" % (time() - tic)


# plots

# fig A1
fig = plt.figure()
ax = fig.add_subplot(111)
h = []
h.append(ax.plot(range(num_iter), n_bg, 'k-'))
h.append(ax.plot(range(num_iter), c.number_bg * np.ones(num_iter), 'k--'))
h.append(ax.plot(range(num_iter), n_ash, 'r-'))
h.append(ax.plot(range(num_iter), c.number_aftershocks * np.ones(num_iter), 'r--'))
ax.set_xlabel('Number iterations')
ax.set_ylabel('Number events')
ax.legend([t[0] for t in h], ('B/g, inferred', 'B/g, true', 'Trig, inferred', 'Trig, true'), 'right')

# fig A2
t = np.linspace(0, 60, 200)
w = c.off_omega
z = w * np.exp(-w * t)
fig = plotting.plot_txy_t_marginals(k_ash[-1], t_max=60)
plt.plot(t, z, 'k--')
ax = fig.gca()
ax.set_ylim([0, w * 1.02])
ax.legend(ax.get_lines(), ('Inferred', 'True'), 'upper right')

t = np.linspace(-0.05, 0.05, 200)
sx = c.off_sigma_x
z = 1/(np.sqrt(2 * np.pi) * sx) * np.exp(-(t**2) / (2 * sx**2))
fig = plotting.plot_txy_x_marginals(k_ash[-1], x_max=0.05)
plt.plot(t, z, 'k--')
ax = fig.gca()
ax.set_ylim([0, 1.05/(np.sqrt(2 * np.pi) * sx)])
ax.set_xlim([-0.05, 0.05])
ax.legend(ax.get_lines(), ('Inferred', 'True'), 'upper right')

t = np.linspace(-0.5, 0.5, 200)
sy = c.off_sigma_y
z = 1/(np.sqrt(2 * np.pi) * sy) * np.exp(-(t**2) / (2 * sy**2))
fig = plotting.plot_txy_y_marginals(k_ash[-1], y_max=0.5)
ax = fig.gca()
line = ax.get_lines()[0]
plt.plot(t, z, 'k--')
ax = fig.gca()
ymax_theor = 1.05/(np.sqrt(2 * np.pi) * sy)
ymax_infer = 1.05 * max(line.get_ydata())
ax.set_ylim([0, max(ymax_infer, ymax_theor)])
ax.set_xlim([-0.5, 0.5])
ax.legend(ax.get_lines(), ('Inferred', 'True'), 'upper right')
