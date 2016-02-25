__author__ = 'gabriel'
import estimation, simulate, plotting
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

def plot_simulation_bg(n_pt=300):
    from scipy.stats import multivariate_normal
    bg_x_max = 10
    cov = np.diag([4.5, 4.5]) ** 2
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
    trigger_x_max = c.trigger_sigma[0] * 30
    trigger_y_max = c.trigger_sigma[1] * 3
    cov = c.trigger_cov()

    t = np.linspace(0, 40, 100)
    zt = c.trigger_decay * c.trigger_intensity * np.exp(-c.trigger_decay * t)
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
        if idx is None:
            return gen
        gen += 1
        idx = data[int(idx)][-1]


def plot_offspring(nevents=20):
    msdict = {
        0: 20,
        1: 10,
        2: 5,
        3: 3,
        4: 2,
        5: 1,
    }
    c = simulate.PlanarGaussianSpaceExponentialTime(t_total=100)
    # manually set first event
    c._data = [(0, 0., (0., 0.), None)]
    while True:
        c.append_triggers()
        if c.ndata > nevents:
            break
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for row in c._data:
        x, y = row[2]
        gen = compute_generation(row[-1], c._data)
        colour = np.array([1, 1, 1]) * min(gen, 5) / float(5.)
        try:
            ms = msdict[gen]
        except AttributeError:
            ms = 1
        ax.plot(x, y, 'o', color=colour, markersize=ms)
        if gen > 0:
            # compute linking line
            prevx, prevy = c._data[int(row[-1])][2]
            ax.plot([prevx, x], [prevy, y], '-', color='gray')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.tick_params(labelsize=16)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=70 )
    ax.xaxis.get_label().set_size(16)
    ax.yaxis.get_label().set_size(16)
    xmin = min(np.min(c.data[:, 1]), -0.01)
    xmax = max(np.max(c.data[:, 1]), 0.01)
    ymin = min(np.min(c.data[:, 2]), -0.1)
    ymax = max(np.max(c.data[:, 2]), 0.1)
    ax.set_xlim(1.1 * np.array([xmin, xmax]))
    ax.set_ylim(1.1 * np.array([ymin, ymax]))
    # ax.set_aspect('equal')
    fig.subplots_adjust(bottom=0.18, top=0.95)
    return fig

