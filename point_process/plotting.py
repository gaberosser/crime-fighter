__author__ = 'gabriel'
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import datetime
import os


def plot_t_kde(k, max_t=50):
    t = np.linspace(0, max_t, 200)
    y = k.pdf(t)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, y)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Density')
    ax.set_ylim([0., max(y) * 1.02])
    return fig


def plot_xy_kde(k, max_x, max_y, npt_1d=50, **kwargs):
    x, y = np.meshgrid(np.linspace(-max_x, max_x, npt_1d), np.linspace(-max_y, max_y, npt_1d))
    z = k.pdf(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_contours = kwargs.pop('n_contours', 40)
    cax = ax.contourf(x, y, z, n_contours, cmap='binary')
    if 'clim' in kwargs:
        clim = kwargs.pop('clim')
        cax.set_clim(clim)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    if kwargs.pop('colorbar', True):
            fig.colorbar(cax)
    return fig


def plot_txy_kde(k, max_x, max_y, npt_1d=50, tpt=None, **kwargs):
    cmap = mpl.cm.binary
    tpt = tpt[:4] if tpt else [1, 5, 10, 20]
    n_contours = kwargs.pop('n_contours', 40)
    x, y = np.meshgrid(np.linspace(-max_x, max_x, npt_1d), np.linspace(-max_y, max_y, npt_1d))
    fig, axarr = plt.subplots(2, 2)
    axarr = axarr.flatten()
    caxarr = []
    z_max = 0.
    for i in range(4):
        ax = axarr[i]
        t = np.ones(x.shape) * tpt[i]
        z = k.pdf(t, x, y)
        caxarr.append(ax.contourf(x, y, z, n_contours, cmap=cmap))
        ax.set_title("t=%d days" % tpt[i])
        z_max = max(z_max, caxarr[-1].get_clim()[1])
    clim = kwargs.pop('clim', (0, z_max))
    [cax.set_clim(clim) for cax in caxarr]
    if kwargs.pop('colorbar', True):
        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.85, 0.1, 0.03, 0.8])
        norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                   norm=norm,
                                   orientation='vertical')
    return fig


def _plot_marginals(k, dim, norm=1.0, data_min=0., data_max=None, npt_1d=200, **kwargs):
    style = kwargs.pop('style', 'k-')
    # if data_max is missing, use the 95th percentile
    if data_max is None:
        data_max = k.marginal_icdf(0.95, dim=dim)
    t = np.linspace(data_min, data_max, npt_1d)
    z = k.marginal_pdf(t, dim=dim) / float(norm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, z, style)
    ax.set_ylim([0, max(z) * 1.02])
    return fig, ax


def plot_txy_t_marginals(k, norm=1.0, t_max=50, npt_1d=200, **kwargs):
    fig, ax = _plot_marginals(k, 0, norm=norm, data_max=t_max, npt_1d=npt_1d, **kwargs)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Density')
    return fig


def plot_txy_x_marginals(k, norm=1.0, x_max=50, npt_1d=200, **kwargs):
    fig, ax = _plot_marginals(k, 1, norm=norm, data_min=-x_max, data_max=x_max, npt_1d=npt_1d)
    ax.set_xlabel('X (metres)')
    ax.set_ylabel('Density')
    return fig


def plot_txy_y_marginals(k, norm=1.0, y_max=50, npt_1d=200, **kwargs):
    fig, ax = _plot_marginals(k, 2, norm=norm, data_min=-y_max, data_max=y_max, npt_1d=npt_1d)
    ax.set_xlabel('Y (metres)')
    ax.set_ylabel('Density')
    return fig


def data_scatter_movie(data, outdir=None, **kwargs):

    from matplotlib import animation

    outdir = outdir or 'output/%s/' % datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    dt = kwargs.pop('dt', 0.5)
    fade = kwargs.pop('fade', 0.03)
    t_max = np.max(data[:, 0])
    t_min = np.min(data[:, 0])
    t_fade = (t_max - t_min) * fade
    x_min = np.min(data[:, 1])
    x_max = np.max(data[:, 1])
    y_min = np.min(data[:, 2])
    y_max = np.max(data[:, 2])
    xlim = np.array([x_min, x_max]) * 1.02
    ylim = np.array([y_min, y_max]) * 1.02

    niter = int((t_max - t_min) / dt) + 1
    fig = plt.figure()
    ax = fig.add_subplot(111)

    t = 0.
    for n in range(niter):
        ax.cla()
        fname = os.path.join(outdir, "%04d.png" % (n+1))
        vis_idx = np.where(data[:, 0] <= t)[0]
        vis_data = data[vis_idx, :]
        tdiff = t - vis_data[:, 0]
        tdiff /= float(t_fade)

        vis_idx = vis_idx[tdiff <= 1.0]
        vis_data = vis_data[tdiff <= 1.0]
        tdiff = tdiff[tdiff <= 1.0]

        bg_idx = np.isnan(vis_data[:, 3])
        ash_idx = ~np.isnan(vis_data[:, 3])
        bg = vis_data[bg_idx, :]
        ash = vis_data[ash_idx, :]

        if bg.size:
            bg_c = np.zeros((sum(bg_idx), 4))
            bg_c[:, 3] = 1. - tdiff[bg_idx]

        links = []
        if ash.size:
            ash_c = np.zeros((sum(ash_idx), 4))
            ash_c[:, 0] = 1.0
            ash_c[:, 3] = 1.0 - tdiff[ash_idx]
            # compute links
            for i, a in enumerate(ash):

                if a[3] < 0:
                    continue

                if a[3] not in vis_idx:
                    # too far back to be visible
                    continue

                src = data[a[3], :]
                if np.isnan(src[3]):
                    # parent is bg
                    this_col = [0, 0, 0, ash_c[i, 3]]
                else:
                    # parent is offspring
                    this_col = [1, 0, 0, ash_c[i, 3]]
                links.append(([src[1], src[2], a[1]-src[1], a[2]-src[2]], list(this_col)))

        try:
            if bg.size:
                h1 = ax.scatter(bg[:, 1], bg[:, 2], marker='o', c=bg_c)
                h1.set_edgecolor('face')
            if ash.size:
                h2 = ax.scatter(ash[:, 1], ash[:, 2], marker='o', c=ash_c)
                h2.set_edgecolor('face')
            if links:
                hlines = []
                harrows = []
                # import ipdb; ipdb.set_trace()
                for i in range(len(links)):
                    # hlines.append(ax.plot(links[i][0], links[i][1], linestyle='-', c=[0, 0, 0, ash_c[i][3]]))
                    harrows.append(ax.arrow(*links[i][0], head_width=0.25, length_includes_head=True, color=links[i][1]))

        except Exception as exc:
            import ipdb; ipdb.set_trace()
        ax.set_title("t = %.2f s" % t)
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)

        fig.savefig(fname)
        t += dt


