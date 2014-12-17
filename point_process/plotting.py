__author__ = 'gabriel'
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import datetime
import os
from analysis import roc
from analysis.plotting import plot_surface_function_on_polygon
from data.models import CartesianSpaceTimeData, DataArray
from django.contrib.gis import geos
import dill
import collections


def plot_t_kde(k, max_t=50):
    t = np.linspace(0, max_t, 200)
    y = k.pdf(t, normed=False)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, y)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Density')
    ax.set_xlim([0., max_t])
    ax.set_ylim([0., max(y) * 1.02])
    return fig


def plot_xy_kde(k, x_range, y_range, npt_1d=50, **kwargs):
    x, y = np.meshgrid(np.linspace(x_range[0], x_range[1], npt_1d), np.linspace(y_range[0], y_range[1], npt_1d))
    z = k.pdf(x, y, normed=False)
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
        z = k.pdf(t, x, y, normed=False)
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


def _plot_marginals(k, dim, norm=1.0, data_min=None, data_max=None, **kwargs):
    style = kwargs.pop('style', 'k-')
    npt = kwargs.pop('npt', 500)  # number of points
    symm = kwargs.pop('symm', True)  # toggles whether x axes are symmetric
    perc = kwargs.pop('percentile', 0.01)  # lower percentile to use for cutoff

    # if data_max is missing, use the 99th percentile
    if data_max is None:
        data_max = k.marginal_icdf(1. - perc, dim=dim)
    if data_min is None:
        data_min = k.marginal_icdf(perc, dim=dim)

    if symm:
        tmp = max(np.abs(data_min), np.abs(data_max))
        data_min = -tmp  # assumed to be negative
        data_max = tmp  # assumed to be positive

    t = np.linspace(data_min, data_max, npt)
    z = k.marginal_pdf(t, dim=dim, normed=False) / float(norm)
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
        fig = ax.figure
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(t, z, style)
    ax.set_ylim([0, max(z) * 1.02])
    return fig, ax


def plot_txy_t_marginals(k, norm=1.0, t_max=None, **kwargs):
    fig, ax = _plot_marginals(k, 0, norm=norm, data_min=0., data_max=t_max, symm=False, **kwargs)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Density')
    return fig


def plot_txy_x_marginals(k, norm=1.0, x_max=None, **kwargs):
    x_min = -x_max if x_max else None
    fig, ax = _plot_marginals(k, 1, norm=norm, data_min=x_min, data_max=x_max, **kwargs)
    ax.set_xlabel('X (metres)')
    ax.set_ylabel('Density')
    return fig


def plot_txy_y_marginals(k, norm=1.0, y_max=None, **kwargs):
    y_min = -y_max if y_max else None
    fig, ax = _plot_marginals(k, 2, norm=norm, data_min=y_min, data_max=y_max, **kwargs)
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


def multiplots(ppobj, simobj=None, maxes=None):
    """
    Convenience function.  Provided with an object of type PP model, produce various useful plots.  Optionally provide
    a simulation object with 'ground truth' information.
    """
    npt = 500
    ci = 0.99
    fig_kwargs = {
        'figsize': (8, 6),
        'dpi': 100,
        'facecolor': 'w',
        }

    if maxes:
        t_max,x_max, y_max = maxes
    else:
        t_max = x_max = y_max = None

    niter = len(ppobj.num_bg)
    iterx = range(1, niter + 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    h = []
    h.append(ax.plot(iterx, ppobj.num_bg, 'k-'))
    h.append(ax.plot(iterx, ppobj.num_trig, 'r-'))
    ymax = max(max(ppobj.num_bg), max(ppobj.num_trig))
    if simobj:
        h.append(ax.plot(iterx, simobj.number_bg * np.ones(niter), 'k--'))
        h.append(ax.plot(iterx, simobj.number_aftershocks * np.ones(niter), 'r--'))
        ymax = max(ymax, simobj.number_bg, simobj.number_aftershocks)
    ax.set_ylim([0, 1.05 * ymax])
    ax.set_xlabel('Number iterations')
    ax.set_ylabel('Number events')
    ax.legend([t[0] for t in h], ('B/g, inferred', 'Trig, inferred', 'B/g, true', 'Trig, true'), 'right')

    # fig A2
    t_max = t_max or ppobj.trigger_kde.marginal_icdf(ci, dim=0)
    fig = plot_txy_t_marginals(ppobj.trigger_kde, norm=ppobj.ndata, t_max=t_max, npt=npt)
    ax = fig.gca()
    if simobj:
        t = np.linspace(0, t_max, npt)
        w = simobj.off_omega
        th = simobj.off_theta
        zt = th * w * np.exp(-w * t)
        plt.plot(t, zt, 'k--')
        ax.set_ylim([0, w * th * 1.02])
        ax.legend(ax.get_lines(), ('Inferred', 'True'), 'upper right')
    ax.set_xlim([0, t_max])

    x_max = x_max or ppobj.trigger_kde.marginal_icdf(ci, dim=1)
    fig = plot_txy_x_marginals(ppobj.trigger_kde, norm=ppobj.ndata, x_max=x_max, npt=npt)
    ax = fig.gca()
    if simobj:
        x = np.linspace(-x_max, x_max, npt)
        sx = simobj.off_sigma_x
        zx = th / (np.sqrt(2 * np.pi) * sx) * np.exp(-(x**2) / (2 * sx**2))
        plt.plot(x, zx, 'k--')
        ax.set_ylim([0, 1.05 * th / (np.sqrt(2 * np.pi) * sx)])
        ax.legend(ax.get_lines(), ('Inferred', 'True'), 'upper right')
    ax.set_xlim([-x_max, x_max])

    y_max = y_max or ppobj.trigger_kde.marginal_icdf(ci, dim=2)
    fig = plot_txy_y_marginals(ppobj.trigger_kde, norm=ppobj.ndata, y_max=y_max, npt=npt)
    ax = fig.gca()
    line = ax.get_lines()[0]
    zmax_infer = max(line.get_ydata())
    if simobj:
        y = np.linspace(-y_max, y_max, npt)
        sy = simobj.off_sigma_y
        zy = th/(np.sqrt(2 * np.pi) * sy) * np.exp(-(y**2) / (2 * sy**2))
        plt.plot(y, zy, 'k--')
        zmax_theor = th/(np.sqrt(2 * np.pi) * sy)
        ax.set_ylim([0, 1.05 * max(zmax_infer, zmax_theor)])
    else:
        ax.set_ylim([0, 1.05 * zmax_infer])
    ax.set_xlim([-y_max, y_max])
    ax.legend(ax.get_lines(), ('Inferred', 'True'), 'upper right')


def plot_trigger_marginals(trigger_kde, percentile=0.01):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(12, 6))

    plot_txy_t_marginals(trigger_kde, ax=ax1, symm=False, percentile=percentile)
    plot_txy_x_marginals(trigger_kde, ax=ax2, symm=True, percentile=percentile)
    plot_txy_y_marginals(trigger_kde, ax=ax3, symm=True, percentile=percentile)

    ax1.yaxis.set_ticklabels([])
    ax1.yaxis.set_ticks([])
    # only ax1 requires a visible y-axis
    ax2.yaxis.set_visible(False)
    ax3.yaxis.set_visible(False)

    plt.tight_layout()


def txy_to_cartesian_data_array(t, x, y):
    ndim = t.ndim
    if x.ndim != ndim or y.ndim != ndim:
        raise AttributeError("Ndim does not match")

    shape = t.shape
    if x.shape != shape or y.shape != shape:
        raise AttributeError("Shape does not match")

    return CartesianSpaceTimeData(np.concatenate(
        (t[..., np.newaxis], x[..., np.newaxis], y[..., np.newaxis]),
        axis=ndim))


def prediction_heatmap(sepp, t, poly=None, kind=None, **kwargs):

    if poly:
        _poly = poly.simplify()
    else:
        # use basic bounding rectangle
        r = roc.RocSpatialGrid(sepp.data[:, 1:])
        _poly = r.generate_bounding_poly()

    arr_fun = lambda x, y: txy_to_cartesian_data_array(np.ones_like(x) * t, x, y)

    if kind is None or kind == "" or kind == "dynamic":
        # full prediction (BG and trigger), BG is time-dependent
        pred_fun = lambda x, y: sepp.predict(arr_fun(x, y))
    elif kind == "static":
        # full prediction (BG and trigger), BG is spatial-only
        pred_fun = lambda x, y: sepp.predict_fixed_background(arr_fun(x, y))
    elif kind == "trigger":
        # trigger only prediction
        pred_fun = lambda x, y: sepp.trigger_density_in_place(arr_fun(x, y))
    elif kind == "bg":
        # BG only prediction, BG is time-dependent
        pred_fun = lambda x, y: sepp.background_density(arr_fun(x, y), spatial_only=False)
    elif kind == "bgstatic":
        # BG only prediction, BG is spatial-only
        pred_fun = lambda x, y: sepp.background_density(arr_fun(x, y), spatial_only=True)
    else:
        raise AttributeError("Supplied kind %s is not recognised", kind)

    return plot_surface_function_on_polygon(_poly, pred_fun, **kwargs)


def validation_multiplot(res):
    max_coverage = 0.20
    methods = ['full', 'full_static', 'bg', 'bg_static', 'trigger']
    methods = collections.OrderedDict([
        ('full', 'Full'),
        ('full_static', 'Full model, static background'),
        ('bg', 'Background only'),
        ('trigger', 'Trigger only'),
    ])
    styles = ['k-', 'k--', 'r-', 'r--']

    x = []
    cc = []
    cc_max = 0.
    for m in methods:
        x.append(np.mean(res['cumulative_area_%s' % m], axis=0))
        cc.append(np.mean(res['cumulative_crime_%s' % m], axis=0))
        cc_max = max(cc_max, cc[-1][x[-1] <= max_coverage].max())

    plt.figure()
    ax = plt.gca()
    [ax.plot(x[i], cc[i], styles[i]) for i in range(len(methods))]
    # import ipdb; ipdb.set_trace()
    ax.set_xlim([0, max_coverage])
    ax.set_ylim([0, np.ceil(10 * cc_max) * 0.1])


def fluctuation_pre_convergence(res, conv_region=10):
    '''
    Various plots relating to the output of the function with the same name in models.
    :param conv_region: Number of datapoints to take before convergence is assumed.
    '''
    niter = len(res['triggers'])
    i = range(niter)
    mean_l2 = np.mean(res['sepp_obj'].l2_differences[conv_region:])
    std_l2 = np.std(res['sepp_obj'].l2_differences[10:], ddof=1)

    # plot l2 differences with region of convergence indicated
    plt.figure()
    plt.plot(i, res['sepp_obj'].l2_differences, 'k-')
    plt.fill_between(i, np.ones(niter) * (mean_l2 - 3 * std_l2), np.ones(niter) * (mean_l2 + 3 * std_l2),
                     edgecolor='none', facecolor='k', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('L2 difference')


def fluctuation_at_convergence(res, poly=None):
    '''
    Various plots relating to the output of the function with the same name in models.
    '''
    r = res['sepp_obj']
    # envelope plots
    # trigger t
    t = np.linspace(0, r.max_delta_t, 500)
    gt = np.array([a.marginal_pdf(t, dim=0, normed=False) / float(r.ndata) for a in res['triggers']])
    gt_normed = np.array([a.marginal_pdf(t, dim=0, normed=True) for a in res['triggers']])

    plt.figure()
    plt.fill_between(t, np.min(gt, axis=0), np.max(gt, axis=0), edgecolor='none', facecolor='k', alpha=0.4)
    plt.plot(t, gt.mean(axis=0), 'k-')
    plt.xlabel('Time (days)', fontsize=18)
    plt.ylabel('Intensity', fontsize=18)

    plt.figure()
    plt.fill_between(t, np.min(gt_normed, axis=0), np.max(gt_normed, axis=0), edgecolor='none', facecolor='k', alpha=0.5)
    plt.plot(t, gt_normed.mean(axis=0), 'k-')
    plt.xlabel('Time (days)', fontsize=18)
    plt.ylabel('Normalised intensity', fontsize=18)

    # trigger x
    # get best axis scaling
    xmax = res['triggers'][-1].marginal_icdf(0.99, dim=1)
    x = np.linspace(-xmax, xmax, 500)
    gx = np.array([a.marginal_pdf(x, dim=1, normed=False) / float(r.ndata) for a in res['triggers']])

    plt.figure()
    plt.fill_between(x, np.min(gx, axis=0), np.max(gx, axis=0), edgecolor='none', facecolor='k', alpha=0.4)
    plt.plot(x, gx.mean(axis=0), 'k-')
    plt.xlim([-xmax, xmax])
    plt.ylim([0, 1.02 * np.max(gx)])
    plt.xlabel('X (metres)', fontsize=18)
    plt.ylabel('Intensity', fontsize=18)

    # background
    if not poly:
        xmin = res['backgrounds'][-1].marginal_icdf(0.005, dim=1)
        ymin = res['backgrounds'][-1].marginal_icdf(0.005, dim=2)
        xmax = res['backgrounds'][-1].marginal_icdf(0.995, dim=1)
        ymax = res['backgrounds'][-1].marginal_icdf(0.995, dim=2)
        poly = geos.Polygon([
            (xmin, ymin),
            (xmin, ymax),
            (xmax, ymax),
            (xmax, ymin),
            (xmin, ymin)
        ])

    def bg_mean_density(x, y):
        xy = DataArray.from_meshgrid(x, y)
        fxy = np.array([a.partial_marginal_pdf(xy) for a in res['backgrounds']])
        return fxy.mean(axis=0).reshape(xy.original_shape)

    def weighted_bg_mean_density(x, y):
        xy = DataArray.from_meshgrid(x, y)
        fxy = np.array([a.partial_marginal_pdf(xy) for a in res['weighted_backgrounds']])
        return fxy.mean(axis=0).reshape(xy.original_shape)

    def bg_rel_range(x, y):
        xy = DataArray.from_meshgrid(x, y)
        fxy = np.array([a.partial_marginal_pdf(xy) for a in res['backgrounds']])
        fxy_mean = fxy.mean(axis=0)
        return (fxy.ptp(axis=0) / fxy_mean).reshape(xy.original_shape)

    def weighted_bg_rel_range(x, y):
        xy = DataArray.from_meshgrid(x, y)
        fxy = np.array([a.partial_marginal_pdf(xy) for a in res['weighted_backgrounds']])
        fxy_mean = fxy.mean(axis=0)
        return (fxy.ptp(axis=0) / fxy_mean).reshape(xy.original_shape)

    plot_surface_function_on_polygon(poly, bg_mean_density, colorbar=True)
    plot_surface_function_on_polygon(poly, bg_rel_range, colorbar=True)
    plot_surface_function_on_polygon(poly, weighted_bg_mean_density, colorbar=True)
    plot_surface_function_on_polygon(poly, weighted_bg_rel_range, colorbar=True)


def delta_effect(res, nrow=8, ncol=7):
    """ to prepare res:
        import dill
        f = open('delta_effect_north_side2.dill', 'r')
        res = []
        while True:
            res.append(dill.load(f))
    """
    niter = len(res[0]['model'].num_bg)
    xiter = range(1, niter + 1)
    res_arr = np.array(res, dtype=object)
    res_arr = res_arr.reshape((nrow, ncol))
    dt = np.array([t['max_delta_t'] for t in res_arr.flat]).reshape((nrow, ncol))
    dd = np.array([t['max_delta_d'] for t in res_arr.flat]).reshape((nrow, ncol))

    majorFormatter = ticker.FormatStrFormatter('%.2e')  # scientific notation

    # l2 differences plot

    fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey='row')
    for (a, r) in zip(axs.flat, res_arr.flat):
        a.plot(xiter, r['model'].l2_differences)

    # iterate over axes with yticks
    for a, d in zip(axs[:, 0], dd[:, 0]):
        a.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
        a.yaxis.set_major_formatter(majorFormatter)
        a.set_ylabel(r'$\Delta d_{{max}} = %d$' % d, fontsize=18)

    # iterate over axes with x ticks
    for a, t in zip(axs[-1, :], dt[-1, :]):
        a.set_xlabel(r'$\Delta t_{{max}} = %d$' % t, fontsize=18)

    plt.draw()
    plt.tight_layout(h_pad=0.0, w_pad=0.1)

    # num trigger / BG plot

    fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey='row')
    for (a, r) in zip(axs.flat, res_arr.flat):
        a.plot(xiter, r['model'].num_bg, 'k-')
        a.plot(xiter, r['model'].num_trig, 'r-')

    # iterate over axes with yticks
    for a, d in zip(axs[:, 0], dd[:, 0]):
        a.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
        a.yaxis.set_major_formatter(majorFormatter)
        a.set_ylabel(r'$\Delta d_{{max}} = %d$' % d, fontsize=18)

    # iterate over axes with x ticks
    for a, t in zip(axs[-1, :], dt[-1, :]):
        a.set_xlabel(r'$\Delta t_{{max}} = %d$' % t, fontsize=18)

    plt.draw()
    plt.tight_layout(h_pad=0.0, w_pad=0.1)

    # trigger T

    fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True)
    for (a, r) in zip(axs.flat, res_arr.flat):
        _plot_marginals(r['model'].trigger_kde, 0, data_min=0, data_max=45, ax=a, symm=False)

    # iterate over axes with yticks
    for a, d in zip(axs[:, 0], dd[:, 0]):
        a.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
        a.yaxis.set_major_formatter(majorFormatter)
        a.set_ylabel(r'$\Delta d_{{max}} = %d$' % d, fontsize=18)

    # iterate over axes with x ticks
    for a, t in zip(axs[-1, :], dt[-1, :]):
        a.set_xlabel(r'$\Delta t_{{max}} = %d$' % t, fontsize=18)

    plt.draw()
    plt.tight_layout(h_pad=0.0, w_pad=0.1)

    # trigger X

    fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True)
    for (a, r) in zip(axs.flat, res_arr.flat):
        _plot_marginals(r['model'].trigger_kde, 1, data_max=200, ax=a)

    # iterate over axes with yticks
    for a, d in zip(axs[:, 0], dd[:, 0]):
        a.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
        a.yaxis.set_major_formatter(majorFormatter)
        a.set_ylabel(r'$\Delta d_{{max}} = %d$' % d, fontsize=18)

    # iterate over axes with x ticks
    for a, t in zip(axs[-1, :], dt[-1, :]):
        a.set_xlabel(r'$\Delta t_{{max}} = %d$' % t, fontsize=18)

    plt.draw()
    plt.tight_layout(h_pad=0.0, w_pad=0.1)