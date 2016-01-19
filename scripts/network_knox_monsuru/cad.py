from analysis import cad
import datetime

from data.models import NetworkData, CartesianData
from analysis.spatial import network_spatial_linkages
import csv
from database.camden import loader
import numpy as np


START_DATE = datetime.date(2011, 3, 1)
END_DATE = datetime.date(2012, 1, 6)
MAX_DAY_NUMBER = 311

if __name__ == '__main__':
    crime_types = ('burglary',
                   'violence',
                   'shoplifting')
    net = loader.load_network()
    data = {}
    cids = {}
    failed = {}

    for ct in crime_types:
        print "Started %s, getting data..." % ct
        data, t0, cid = cad.get_crimes_from_dump('monsuru_cad_%s' % ct)
        cid = np.array(cid)
        # filter by end date
        idx = np.where(data[:, 0] < (MAX_DAY_NUMBER + 1))[0]
        data = data[idx]
        cid = cid[idx]
        cids[ct] = cid

        jiggled = cad.jiggle_all_points_on_grid(data[:, 1], data[:, 2])
        b_jiggled = [np.all(a != b) for a, b in zip(data[:, 1:], jiggled)]

        snapped, fail = NetworkData.from_cartesian(net, jiggled, return_failure_idx=True)
        keep_idx = [k for k in range(len(cid)) if k not in fail]
        cid = cid[keep_idx]
        times = data[keep_idx, 0]
        failed[ct] = fail

        print "Done. writing data to a file."
        fields = ['my_idx', 'original_idx', 'days_since_1_mar_2011', 'x', 'y', 'jiggled?']
        xy = snapped.to_cartesian()
        out_data = []
        for k in range(cid.size):
            out_data.append(
                (k, cid[k], times[k], xy[k, 0], xy[k, 1], b_jiggled[k])
            )
        with open("data_for_pairwise_comparison_%s.csv" % ct, 'w') as f:
            c = csv.writer(f)
            c.writerow(fields)
            c.writerows(out_data)

        print "Done. Computing pairwise distances..."

        ii, jj, dd, nd = network_spatial_linkages(snapped, 1000)

        # filter out nonsense results where net distance > euclidean cutoff of 1000
        filt_idx = np.where(nd <= 1000)[0]
        ii = ii[filt_idx]
        jj = jj[filt_idx]
        dd = dd[filt_idx]
        nd = nd[filt_idx]

        # create CSV
        fields = ['from_idx', 'to_idx', 'from_id', 'to_id', 'euclidean_distance', 'network_distance']
        out_data = []
        for k in range(len(ii)):
            i = ii[k]
            j = jj[k]
            ci = cid[i]
            cj = cid[j]
            d = dd[k]
            n = nd[k]
            out_data.append([i, j, ci, cj, d, n])

        print "Completed %s, saving..." % ct

        with open("pairwise_network_distances_%s.csv" % ct, 'w') as f:
            c = csv.writer(f)
            c.writerow(fields)
            c.writerows(out_data)

        print "Done"