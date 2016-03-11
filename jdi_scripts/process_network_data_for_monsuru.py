from network import itn
import os
from jdi import IN_DIR, OUT_DIR
from shapely import geometry
from jdi.data import cris, net, boundary, consts
from data.models import NetworkData, CartesianData
import datetime
import csv
import numpy as np
import fiona
import collections


MINOR_CRIME_TYPE = 'Burglary In A Dwelling'
START_DATE = datetime.datetime(2013, 7, 1)
N_TRAIN = 180
N_VALIDATION = 60
N_TEST = 100


def load_ma_shapefile(long_bo):
    subdir = os.path.join(IN_DIR, 'greater_london', 'from_monsuru', 'shapefiles')    
    p = os.path.join(subdir, long_bo)
    fn = os.path.join(p, "%s_network.shp" % long_bo)

    with fiona.open(fn, 'r') as s:
        arr = list(s.values())
        attrs = dict([
            (t['properties']['fid_1'], t['properties']) for t in arr
        ])
    return attrs
    
    
def process_snapped_crime_data(bo, outdir):
    
    long_bo = consts.BOROUGH_FILENAME_MAP[bo.upper()]
    out_file = os.path.join(outdir, "%s.csv" % consts.BOROUGH_FILENAME_MAP[bo.upper()])
    
    # load crimes
    end_date = START_DATE + datetime.timedelta(days=(N_TRAIN + N_VALIDATION + N_TEST + 1))
    data, t0, cr_num = cris.get_cris_data(start_dt=START_DATE, end_dt=end_date, borough=bo, minor_crime_type=MINOR_CRIME_TYPE, convert_dt=False)    
    pre_snap_xy = CartesianData(data[:, 1:].astype(float))
    
    # boundary
    domain = boundary.get_borough_boundary(bo)
    
    # my network
    itn_obj = net.get_itn_network(bo)
    itn_clip = itn_obj.within_boundary(domain, clip_lines=False)
    
    net_data, failed = NetworkData.from_cartesian(itn_clip, 
                                                pre_snap_xy,
                                                grid_size=100, 
                                                radius=50,
                                                return_failure_idx=True)
    
    to_keep = sorted(list(set(range(data.shape[0])) - set(failed)))
    cr_num = np.array(cr_num)[to_keep]
    dt = data[to_keep, 0]
    
    post_snap_xy = net_data.to_cartesian()
    x, y = post_snap_xy.separate
    
    snap_dist = post_snap_xy.distance(pre_snap_xy.getrows(to_keep)).toarray()
    xpre, ypre = pre_snap_xy.getrows(to_keep).separate
    
    # load MA shapefile for fid:id mapping
    attrs = load_ma_shapefile(long_bo)
    
    # check network consistency
    ma_fids = set(attrs.keys())
    gr_fids = set([t.fid for t in itn_clip.edges()])
    print "%d edges in MA not in GR" % len(ma_fids - gr_fids)
    if len(gr_fids - ma_fids):
        # a segment in GR network is NOT present in MA network
        import ipdb; ipdb.set_trace()
    
    fields = (
        'crime_id',
        't_year',
        't_month',
        't_day',
        't_hour',
        't_minute',
        't_second',
        'x',
        'y',
        'x_pre_snap',
        'y_pre_snap',        
        'snap_distance',
        'itn_edge_id',
        'ma_edge_id'
    )
    
    with open(out_file, 'wb') as f:
        c = csv.DictWriter(f, fields)
        c.writeheader()
        for i in range(x.size):
            fid = net_data[i, 0].edge.fid
            ma_row = attrs.get(fid, None)
            r = {
                'crime_id': cr_num[i],
                't_year': dt[i].year,
                't_month': dt[i].month,
                't_day': dt[i].day,
                't_hour': dt[i].hour,
                't_minute': dt[i].minute,
                't_second': dt[i].second,
                'x': x[i],
                'y': y[i],
                'x_pre_snap': xpre[i],
                'y_pre_snap': ypre[i],
                'snap_distance': snap_dist[i],
                'itn_edge_id': fid,
                'ma_edge_id': ma_row.get('ids', None) if ma_row else None,
            }
            c.writerow(r)
    

def create_adjacency_matrix(bo, outdir):
    long_bo = consts.BOROUGH_FILENAME_MAP[bo.upper()]
    out_file = os.path.join(outdir, "%s.csv" % consts.BOROUGH_FILENAME_MAP[bo.upper()])    
        
    domain = boundary.get_borough_boundary(bo)        
        
    itn_obj = net.get_itn_network(bo)
    itn_clip = itn_obj.within_boundary(domain, clip_lines=False)
    
    print "Computing adjacency matrix for %s..." % long_bo    
    
    # build adjacency matrix
    node_adj = itn_clip.g.adjacency_list()
    nodes = itn_clip.nodes()
    fids = [t.fid for t in itn_clip.edges()]
    adj = collections.defaultdict(list)
    fid_adj = collections.defaultdict(set)
    for n0, na in zip(nodes, node_adj):
        this_inds = []
        this_fids = []
        for n1 in na:                
            this_fids.extend(itn_clip.edge[n0][n1].keys())
            this_inds.extend([fids.index(t) for t in itn_clip.edge[n0][n1].keys()])
        for e in this_inds:
            adj[e].extend(this_inds)
        for f in this_fids:
            [fid_adj[f].add(x) for x in this_fids]
    
    fields = ['fid'] + fids
    n = len(fids)

    print "Complete. Saving to CSV..."    

    with open(out_file, 'wb') as f:
        c = csv.writer(f)
        c.writerow(fields)
        for i, fid in enumerate(fids):
            if i % 500 == 0:
                print "%d / %d" % (i, n)
            
            this_adj = adj[i]
            row = np.zeros(n, dtype=int)
            row[np.unique(adj[i])] = 1
            row = [fid] + list(row)
            c.writerow(row)
            
#    out_file = os.path.join(outdir, "%s_links.csv" % consts.BOROUGH_FILENAME_MAP[bo.upper()])
#
#    with open(out_file, 'wb') as f:
#        c = csv.writer(f)
#        for i, fid in enumerate(fids):
#            c.writerow([fid] + list(fid_adj[fid]))


if __name__ == "__main__":
    boroughs = ('sx', 'qk', 'ek', 'yr', 'ni', 'cw')
    #outdir = os.path.join(OUT_DIR, 'for_monsuru')    
    #if not os.path.isdir(outdir):
    #    os.makedirs(outdir)
    #for bo in boroughs:
    #    process_snapped_crime_data(bo, outdir)
    
    #outdir = os.path.join(OUT_DIR, 'for_monsuru', 'adjacency')
    #if not os.path.isdir(outdir):
    #    os.makedirs(outdir)
    #for bo in boroughs:
    #    create_adjacency_matrix(bo, outdir)
