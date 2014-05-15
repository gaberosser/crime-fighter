__author__ = 'gabriel'
import estimation, simulate, plotting
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

def plot_simulation_bg(n_pt=300):
    from scipy.stats import multivariate_normal
    bg_x_max = 10
    c = simulate.MohlerSimulation()
    cov=c.bg_sigma**2 * np.eye(2)
    x = np.linspace(-bg_x_max, bg_x_max, 100)
    x, y = np.meshgrid(x, x)
    xy = np.vstack((x.flatten(), y.flatten())).transpose()
    z = multivariate_normal.pdf(xy, mean=[0, 0], cov=cov).reshape((100, 100))
    r = multivariate_normal.rvs(mean=[0, 0], cov=cov, size=(n_pt))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(x, y, z, 50, cmap='binary')
    ax.scatter(r[:, 0], r[:, 1], facecolor='r')
    ax.set_xlim([-bg_x_max, bg_x_max])
    ax.set_ylim([-bg_x_max, bg_x_max])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.tick_params(labelsize=16)
    ax.xaxis.get_label().set_size(16)
    ax.yaxis.get_label().set_size(16)
    return fig


def plot_simulation_trigger_fun():
    from scipy.stats import multivariate_normal
    c = simulate.MohlerSimulation()
    trigger_x_max = c.off_sigma_y
    trigger_y_max = c.off_sigma_y*3
    cov = np.eye(2) * np.array([c.off_sigma_x, c.off_sigma_y]) ** 2
    t = np.linspace(0, 40, 100)
    zt = c.off_omega * c.off_theta * np.exp(-c.off_theta * t)
    x = np.linspace(-trigger_x_max, trigger_x_max, 100)
    y = np.linspace(-trigger_y_max, trigger_y_max, 100)
    x, y = np.meshgrid(x, y)
    xy = np.vstack((x.flatten(), y.flatten())).transpose()
    z = multivariate_normal.pdf(xy, mean=[0, 0], cov=cov).reshape((100, 100)).reshape((100, 100))

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.contourf(x, y, z, 50, cmap='binary')
    ax.set_xlim([-trigger_x_max, trigger_x_max])
    ax.set_ylim([-trigger_y_max, trigger_y_max])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.tick_params(labelsize=16)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=70 )
    ax.xaxis.get_label().set_size(16)
    ax.yaxis.get_label().set_size(16)
    ax.set_aspect('equal')

    # fig2 = plt.figure()
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t, zt, 'k-')
    ax2.set_xlim([0, 40])
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Trigger density')
    ax2.tick_params(labelsize=16)
    ax2.xaxis.get_label().set_size(16)
    ax2.yaxis.get_label().set_size(16)

    fig.subplots_adjust(bottom=0.18, top=0.95)
    return fig


def compute_generation(idx, data):
    gen = 0
    while True:
        if np.isnan(idx):
            return gen
        gen += 1
        idx = data[int(idx)][-1]


def plot_offsrping(nevents=20):
    msdict = {
        0: 20,
        1: 10,
        2: 5,
        3: 3,
        4: 2,
        5: 1,
    }
    c = simulate.MohlerSimulation()
    c.data = np.array([[0, 0., 0., 0., np.nan]])
    while True:
        c.generate_aftershocks()
        if c.data.shape[0] > nevents:
            break
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for row in c.data:
        gen = compute_generation(row[-1], c.data)
        colour = np.array([1, 1, 1]) * min(gen, 5) / float(5.)
        try:
            ms = msdict[gen]
        except AttributeError:
            ms = 1
        ax.plot(row[2], row[3], 'o', color=colour, markersize=ms)
        if gen > 0:
            # compute linking line
            prev = c.data[int(row[-1])]
            ax.plot([prev[2], row[2]], [prev[3], row[3]], '-', color='gray')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.tick_params(labelsize=16)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=70 )
    ax.xaxis.get_label().set_size(16)
    ax.yaxis.get_label().set_size(16)
    xmin = min(np.min(c.data[:, 2]), -0.01)
    xmax = max(np.max(c.data[:, 2]), 0.01)
    ymin = min(np.min(c.data[:, 3]), -0.1)
    ymax = max(np.max(c.data[:, 3]), 0.1)
    ax.set_xlim(1.1 * np.array([xmin, xmax]))
    ax.set_ylim(1.1 * np.array([ymin, ymax]))
    # ax.set_aspect('equal')
    fig.subplots_adjust(bottom=0.18, top=0.95)
    return fig

