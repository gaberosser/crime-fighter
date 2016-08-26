__author__ = 'gabriel'
import numpy as np
from matplotlib import pyplot as plt
import dill
import os
from plotting import utils


INDIR = '/home/gabriel/Dropbox/research/results/birmingham_jqc'
TYPESTEM = 'start_max24hours'
# TYPESTEM = 'end'
# TYPESTEM = 'start'
FILESTEM = 'linearexponentialkde_start_day_180_60_iterations_%s.dill' % TYPESTEM


if __name__ == '__main__':
    for m in ['network', 'planar']:
        fn = os.path.join(INDIR, '%s_%s' % (m, FILESTEM))
        with open(fn, 'rb') as f:
            res = dill.load(f)
        tt = res['tt']
        dd = res['dd']
        ll = np.array(res['ll'])
        ll_total = ll.sum(axis=0)
        vmin_pre, vmin, vmax = utils.abs_bound_from_rel(ll_total, [0.2, 0.25, 0.98])

        iopt, jopt = np.where(ll_total == ll_total.max())
        if len(iopt) > 1:
            print "WARNING: found %d maxima in log likelihood" % len(iopt)
            iopt = iopt[0]
            jopt = jopt[0]
        topt = tt[iopt, jopt]
        dopt = dd[iopt, jopt]

        ll_total[ll_total < vmin_pre] = np.nan
        ll_total[ll_total > vmax] = vmax

        fig = plt.figure(figsize=[6, 4])
        ax = fig.add_subplot(111)
        plt.contourf(tt, dd, ll_total, 100, vmin=vmin, cmap='Reds')
        ax.plot([topt, topt], [0, dopt], 'k--')
        ax.plot([0, topt], [dopt, dopt], 'k--')

        ax.set_xlabel('Time bandwidth (days)', size=16)
        ax.set_ylabel('Spatial bandwidth (metres)', size=16)
        plt.colorbar()
        plt.tight_layout()

        fnout = 'optimal_bandwidth_%s_%s' % (m, TYPESTEM)
        fig.savefig('%s.eps' % fnout, dpi=300)
        fig.savefig('%s.pdf' % fnout, dpi=300)
        fig.savefig('%s.png' % fnout, dpi=300)
