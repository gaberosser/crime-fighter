__author__ = 'gabriel'
from analysis import chicago
from plotting import spatiotemporal
from database.chicago import consts
import datetime
from matplotlib import pyplot as plt
import numpy as np


start_date = datetime.date(2011, 3, 1)
end_date = start_date + datetime.timedelta(days=366)

all_domains = chicago.get_chicago_side_polys(as_shapely=True)
max_d = 500
max_t = 120
fmax = 0.99
nbin = 60

abbreviated_regions = {
    'South': 'S',
    'Southwest': 'SW',
    'West': 'W',
    'Northwest': 'NW',
    'North': 'N',
    'Central': 'C',
    'Far North': 'FN',
    'Far Southwest': 'FSW',
    'Far Southeast': 'FSE',
}

for ct in consts.CRIME_TYPES:
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True,
                            figsize=(8, 8))
    for i, r in enumerate(consts.REGIONS):
        domain = all_domains[r]
        data, t0, cid = chicago.get_crimes_by_type(start_date=start_date,
                                                   end_date=end_date,
                                                   crime_type=ct,
                                                   domain=domain)
        ax = axs.flat[i]
        spatiotemporal.pairwise_distance_histogram_manual(data,
                                                          max_t=max_t,
                                                          max_d=max_d,
                                                          nbin=nbin,
                                                          ax=ax,
                                                          colorbar=False,
                                                          fmax=fmax)
        ax.set_xlim(np.array([-1, 1]) * max_d * 1.05)
        ax.set_ylim(np.array([-1, 1]) * max_d * 1.05)
        ax.set_aspect('equal')
        ax.set_yticks([])
        ax.set_xticks([])

        ax.text(-max_d, max_d, abbreviated_regions[r], fontsize=14)

    [axs[2, j].set_xticks([-max_d, 0, max_d]) for j in range(3)]
    [axs[j, 0].set_yticks([-max_d, 0, max_d]) for j in range(3)]

    big_ax = fig.add_subplot(111)
    big_ax.spines['top'].set_color('none')
    big_ax.spines['bottom'].set_color('none')
    big_ax.spines['left'].set_color('none')
    big_ax.spines['right'].set_color('none')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    big_ax.set_xlabel(r'$\Delta x$ (m)')
    big_ax.set_ylabel(r'$\Delta y$ (m)')
    big_ax.patch.set_visible(False)

    plt.tight_layout(pad=0.5, h_pad=0.1, w_pad=0.1)
    big_ax.set_position([0.05, 0.05, 0.95, 0.95])

    filename = 'pairwise_distance_maxd_%d_maxt_%d_%s' % (max_d, max_t, ct)
    fig.savefig(filename + '.png', dpi=200)
    fig.savefig(filename + '.pdf')

