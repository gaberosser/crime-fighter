__author__ = 'gabriel'
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import dblquad


def plot_t_kde(k, max_t=50):
    t = np.linspace(0, max_t, 200)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, k.pdf(t))
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Density')
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


def plot_txy_t_marginals(k, t_max=50, npt_1d=50, **kwargs):
    # gather max_x and max_y from the supplied KDE data
    min_x = min(k.data[:, 1])
    min_y = min(k.data[:, 2])
    max_x = max(k.data[:, 1])
    max_y = max(k.data[:, 2])
    t = np.linspace(0., t_max, npt_1d)
    z = k.marginal_pdf(t, dim=0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, z)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Density')
    return fig