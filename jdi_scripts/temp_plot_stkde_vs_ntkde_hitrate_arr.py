from jdi_scripts import analyse_bandwidth_optimisation as abo
from matplotlib import pyplot as plt
from jdi.data import consts
import numpy as np

net_subdir = 'network_bandwidth_linearexponential'
planar_subdir = 'planar_bandwidth_linearexponential'
boroughs = ('ni', 'qk', 'sx', 'yr', 'ek', 'cw')
plot_shape = (2, 3)
xmax = 0.25
crime_type = 'Burglary In A Dwelling'

hr_grid = abo.load_aggregated_validation_hit_rate_results(crime_type, planar_subdir, method='grid')
hr_inter = abo.load_aggregated_validation_hit_rate_results(crime_type, planar_subdir, method='intersection')
hr_net = abo.load_aggregated_validation_hit_rate_results(crime_type, net_subdir, method='')


fig, axs = plt.subplots(*plot_shape, sharex=True, sharey=True, figsize=(10, 8))

for i in range(len(boroughs)):
    bo = boroughs[i]
    ax = axs.flat[i]
    # gx = hr_grid['coverage'][bo]
    # gy = hr_grid['hit_rate'][bo]
    # ix = hr_inter['coverage'][bo]
    # iy = hr_inter['hit_rate'][bo]
    # nx = hr_net['coverage'][bo]
    # ny = hr_net['hit_rate'][bo]
    gx = hr_grid['coverage'][bo].mean(axis=0) * 100
    gy = hr_grid['hit_rate'][bo].mean(axis=0) * 100
    ix = hr_inter['coverage'][bo].mean(axis=0) * 100
    iy = hr_inter['hit_rate'][bo].mean(axis=0) * 100
    nx = hr_net['coverage'][bo].mean(axis=0) * 100
    ny = hr_net['hit_rate'][bo].mean(axis=0) * 100

    ax.plot(gx, gy, 'r--', label='Grid')
    ax.plot(ix, iy, 'r-', label='Intersection')
    ax.plot(nx, ny, 'k-', label='Network')
    ax.set_xlim([0, xmax * 100])
    ax.set_title(consts.BOROUGH_NAME_MAP[bo.upper()])

    ii, jj = np.unravel_index(i, axs.shape)
    if ii == plot_shape[0] - 1:
        ax.set_xlabel('Coverage (%)')
    if jj == 0:
        ax.set_ylabel('Mean hit rate (%)')
    if ii == 0 and jj == plot_shape[1] - 1:
        ax.legend(loc='upper left')

    plt.tight_layout(0.4)