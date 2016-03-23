from matplotlib import pyplot as plt
import matplotlib.ticker as plticker
from tools import get_ellipse_coords


def plot_spatial_ellipse_array(sepp_model,
                               icdf=0.95,
                               max_d=800.,
                               ax=None,
                               plot_kwargs=None):
    """
    Plot the ellipsoids containing icdf of the spatial triggering density
    :param icdf: The proportion of spatial density in each dimension contained within the ellipse boundary
    :param max_d: The axis maximum value. This is increased automatically if it is too small to contain any of the
    ellipses.
    :return:
    """
    icdf_two_tailed = 0.5 + icdf / 2.
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    plot_kwargs = plot_kwargs or {}

    loc = plticker.MultipleLocator(base=400.0) # this locator puts ticks at regular intervals

    k = sepp_model.trigger_kde
    if k.ndim == 3:
        a = k.marginal_icdf(icdf_two_tailed, dim=1)
        b = k.marginal_icdf(icdf_two_tailed, dim=2)
        coords = get_ellipse_coords(a=a, b=b)
        max_d = max(max(a, b), max_d)
    else:
        a = k.marginal_icdf(icdf, dim=1)
        coords = get_ellipse_coords(a, a)
        max_d = max(a, max_d)

    ax.plot(coords[:, 0], coords[:, 1], **plot_kwargs)
    ax.set_xlim([-max_d, max_d])
    ax.set_ylim([-max_d, max_d])
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.set_aspect('equal', adjustable='box-forced')

    # plt.tight_layout(h_pad=0.2, w_pad=0.2)

    # big_ax = fig.add_subplot(111)
    # big_ax.spines['top'].set_color('none')
    # big_ax.spines['bottom'].set_color('none')
    # big_ax.spines['left'].set_color('none')
    # big_ax.spines['right'].set_color('none')
    # big_ax.set_xticks([])
    # big_ax.set_yticks([])
    # big_ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    # big_ax.set_xlabel(r'$\Delta x$')
    # big_ax.set_ylabel(r'$\Delta y$')
    # big_ax.patch.set_visible(False)

    # big_ax.set_position([0.05, 0.05, 0.95, 0.9])
