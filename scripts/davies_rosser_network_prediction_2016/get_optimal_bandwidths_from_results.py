__author__ = 'gabriel'
import numpy as np
from matplotlib import pyplot as plt
import dill
import os
import csv


INDIR = '/home/gabriel/Dropbox/research/results/birmingham_jqc'
TYPESTEMS = [
    'start_max24hours',
    'end',
    'start'
    ]
OUTFILE = 'optimal_bandwidths.csv'

if __name__ == '__main__':
    fout = open(OUTFILE, 'wb')
    cout = csv.writer(fout)
    cout.writerow(['Method', 'data', 'topt', 'dopt'])
    for t in TYPESTEMS:
        for m in ['network', 'planar']:
            FILESTEM = 'linearexponentialkde_start_day_180_60_iterations_%s.dill' % t
            fn = os.path.join(INDIR, '%s_%s' % (m, FILESTEM))
            with open(fn, 'rb') as f:
                res = dill.load(f)
            tt = res['tt']
            dd = res['dd']
            ll = np.array(res['ll'])
            ll_total = ll.sum(axis=0)

            iopt, jopt = np.where(ll_total == ll_total.max())
            if hasattr(iopt, '__iter__'):
                if len(iopt) > 1:
                    print "WARNING: found %d maxima in log likelihood" % len(iopt)
                iopt = iopt[0]
                jopt = jopt[0]

            topt = tt[iopt, jopt]
            dopt = dd[iopt, jopt]
            cout.writerow([m, t, topt, dopt])

    fout.close()