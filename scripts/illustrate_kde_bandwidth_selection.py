__author__ = 'gabriel'
import numpy as np
from matplotlib import pyplot as plt
from kde.models import FixedBandwidthKde
from data import models as data_models
import bisect


def create_figure():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    return fig, ax

def scatterplot_data(data_xy, s, ax):
    ax.scatter(data_xy.toarray(0), data_xy.toarray(1), s=s, c='k', marker='x')


if __name__ == '__main__':
    sx = 4.
    mx = 5.
    ndata = 10
    tf = 10.  # end time
    npt = 500  # number of points to use when evaluating KDE

    # simulate some data
    data_xy = data_models.CartesianData.from_args(
        *np.random.multivariate_normal([mx, mx], [[sx, 0], [0, sx]], ndata).transpose()
    )
    data_t = data_models.DataArray(np.linspace(0, tf, ndata))
    data_txy = data_t.adddim(data_xy)
    s = np.exp(-(tf - data_t.toarray())) * 1000.  # marker sizes

    # build some KDEs
    ssx = [.5, 1., 2.]
    sst = [.1, 2., 8.]
    ks = []
    for i in range(3):
        for j in range(3):
            ks.append(
                FixedBandwidthKde(data_txy, bandwidths=(sst[i], ssx[j], ssx[j]))
            )

    # compute PDFs
    txy = data_models.CartesianSpaceTimeData.from_meshgrid(
        *np.meshgrid(
            [tf],
            np.linspace(0, 2 * mx, npt),
            np.linspace(0, 2 * mx, npt),
        )
    )
    t = data_models.DataArray(np.ones(npt ** 2) * tf)
    xx = txy.toarray(1).squeeze()
    yy = txy.toarray(2).squeeze()
    zs = [k.pdf(txy).squeeze() for k in ks]

    # plots
    fig, ax = create_figure()
    scatterplot_data(data_xy, s, ax)
    fig.savefig('toy_data.png')

    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            z = zs[j + 3 * i]
            # norm
            idx = bisect.bisect_left(
                np.linspace(0, 1, z.size),
                0.95
            )
            vmax = sorted(z.flat)[idx]
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            z[z>vmax] = vmax
            clevels = np.linspace(z.min(), vmax, 50)
            ax.contourf(xx, yy, z, clevels, cmap='Reds')
            scatterplot_data(data_xy, s, ax)
    plt.tight_layout()
    fig.savefig('toy_data_plus_kde_array.png')
