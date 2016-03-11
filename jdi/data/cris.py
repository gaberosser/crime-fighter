import settings
import os
import csv
from jdi.data import net, consts
from data.models import NetworkData, NetworkSpaceTimeData, DataArray
import datetime
from shapely import geometry
from functools import partial
import numpy as np
import pickle
import dill
from kde import models
from matplotlib import pyplot as plt
from matplotlib import colors

# subdirectory for processed data output
CRIS_PROCESSED_SUBDIR = 'cris_processed'

# to avoid data coding issues: define a min time.
# any records dated earlier than this are discarded
MIN_T = datetime.datetime(2012, 12, 1)


def date_parser(t):
    return datetime.datetime.strptime(t, '%Y-%m-%d').date()
    

def time_parser(t):
    return datetime.datetime.strptime(t, '%H:%M:%S').time()
    
    
def datetime_parser(row, prefix='GEN_Reported_'):
    kd = prefix + 'Date'
    kt = prefix + 'Time'
    if not row[kd] or not row[kt]:
        return
    d = date_parser(row[kd])
    t = time_parser(row[kt])
    return datetime.datetime.combine(d, t)
    
    
CRIS_FIELDS = {
    'dt_reported': partial(datetime_parser, prefix='GEN_Reported_'),
    'dt_committed_from': partial(datetime_parser, prefix='GEN_Committed_on_from_'),
    'dt_committed_to': partial(datetime_parser, prefix='GEN_Committed_to_'),
    'crime_number': lambda x: int(x.get('CR_No')),
    'major_type': lambda x: x.get('Major Class Description'),
    'minor_type': lambda x: x.get('Minor Class Description'),
    'borough': lambda x: x.get('Owning_Borough'),
    'x': lambda x: int(x.get('X')),
    'y': lambda x: int(x.get('Y')),
}


def parse_one(x):
    return dict([(k, f(x)) for k, f in CRIS_FIELDS.items()])


def cris_raw_data_generator(borough=None):
    if borough and not hasattr(borough, '__iter__'):        
        infile = os.path.join(settings.BLUE_DATA, 'cris_%s.csv' % borough.lower())
    else:
        infile = os.path.join(settings.BLUE_DATA, 'cris.csv')
    f = open(infile, 'r')
    c = csv.DictReader(f)
    def gen():
        for r in c:
            yield parse_one(r)
    return gen()


def cris_data_generator(borough=None):
    if borough is not None: 
        if hasattr(borough, '__iter__'):
            infiles = [os.path.join(
                settings.BLUE_DATA,
                CRIS_PROCESSED_SUBDIR, 
                'cris_%s.dill' % bo.lower()
            ) for bo in borough]
        else:
            infiles = [os.path.join(
                settings.BLUE_DATA,
                CRIS_PROCESSED_SUBDIR, 
                'cris_%s.dill' % borough.lower()
            )]
    else:
        infiles = [os.path.join(
            settings.BLUE_DATA,
            CRIS_PROCESSED_SUBDIR,
            'cris_%s.dill' % bo.lower()
        ) for bo in consts.BOROUGH_CODES.keys()]
    data = []
    for fn in infiles:
        with open(fn, 'rb') as f:
            data.extend(dill.load(f))
            #c = csv.DictReader(f)
            #data.extend(list(c))
    return data
         
     
def get_raw_data():
    infile = os.path.join(settings.BLUE_DATA, 'cris.csv')
    with open(infile, 'r') as f:
        c = csv.DictReader(f)
        res = list(c)
    return res
    
    
def slice_data_by_borough():
    """ Slice the raw data into per-borough CSV files """
    all_data = get_raw_data()
    borough_codes = set([t['Owning_Borough'] for t in all_data])
    for bc in borough_codes:
        outfile = os.path.join(settings.BLUE_DATA, 'cris_%s.csv' % bc.lower())
        this_data = [t for t in all_data if t['Owning_Borough'] == bc]
        with open(outfile, 'wb') as f:
            c = csv.DictWriter(f, fieldnames=all_data[0].keys())
            c.writeheader()
            c.writerows(this_data)
        print "Completed slicing borough %s" % bc
        

def process_raw_slices(overwrite=False):
    """ Reduce the raw per-borough CSVs to a more minimal representation for
    better loading times and reduced memory consumption """
    the_dir = os.path.join(settings.BLUE_DATA, CRIS_PROCESSED_SUBDIR)
    if not os.path.isdir(the_dir):
        os.makedirs(the_dir)
    for bc in consts.BOROUGH_CODES:
        outfile = os.path.join(the_dir, 'cris_%s.dill' % bc.lower())
        if os.path.isfile(outfile) and not overwrite:
            print "File already exists for borough %s. Skipping..." % bc
            continue

        with open(outfile, 'wb') as f:
            data = list(cris_raw_data_generator(borough=bc))
            dill.dump(data, f)
        print "Completed borough: %s" % bc
    

def get_cris_data(start_dt=datetime.datetime(2013, 1, 1), 
                    end_dt=None, 
                    borough=None, 
                    domain=None, 
                    major_crime_type=None,
                    minor_crime_type=None, 
                    max_time_diff_hrs=24., 
                    st_only=True, 
                    convert_dt=True,
                    snap_to_network=None):
    """
    CRIS data getter.
    :param domain: shapely (multi)polygon for spatial lookups
    :param st_only: If True, output data is a N x 3 matrix
    :param convert_dt: If True, datetimes are converted to days since t0 (first date)
    :param snap_to_network: If a net object is supplied here, the spatial data 
    will be snapped and converted to NetworkData
    """
    # convert from date to datetime if necessary
    if major_crime_type is not None:
        if hasattr(major_crime_type, '__iter__'):
            major_crime_type = [t.lower() for t in major_crime_type]    
        else:
            major_crime_type = [major_crime_type.lower()]

    if minor_crime_type is not None:
        if hasattr(minor_crime_type, '__iter__'):
            minor_crime_type = [t.lower() for t in minor_crime_type]    
        else:
            minor_crime_type = [minor_crime_type.lower()]
    
    if max_time_diff_hrs is not None:
        mtdiff_sec = max_time_diff_hrs * 3600
    # shortcut: if a single borough is specified, we only use that reduced CSV
    g = cris_data_generator(borough=borough)
    times = []
    cr_num = []
    keep = []
    for r in g:
        dt_from = r.pop('dt_committed_from')
        dt_to = r.pop('dt_committed_to', None)
        #import pdb; pdb.set_trace()
        if not dt_from:
            continue
        if dt_from < MIN_T:
            continue
        if not dt_to:
            t = dt_from
        elif dt_to < MIN_T:
            continue
        else:
            dt = dt_to - dt_from
            if max_time_diff_hrs is not None and dt.total_seconds() > mtdiff_sec:
                continue
            else:
                t = dt_from + dt / 2
        
        if start_dt is not None and t < start_dt:
            continue
        if end_dt is not None and t > end_dt:
            continue
        if borough is not None:
            if hasattr(borough, '__iter__'): 
                if r['borough'].lower() not in borough:
                    continue
            elif r['borough'].lower() != borough:
                continue                
        if major_crime_type is not None:
            if r['major_type'].lower() not in major_crime_type:
                continue
        if minor_crime_type is not None:
            if r['minor_type'].lower() not in minor_crime_type:
                continue
                
        if domain is not None:
            pt = geometry.Point(r['x'], r['y'])
            if not pt.within(domain):
                continue
        
        times.append(t)
        cr_num.append(r['crime_number'])
        keep.append(r)

    cr_num = np.array(cr_num)        
    # finally, convert times if required
    t0 = min(times)
    if convert_dt:
        times = np.array([(t - t0).total_seconds() / (24. * 3600.) for t in times])
    else:
         times = np.array(times, dtype=object)
         
    space = np.array([(r.pop('x'), r.pop('y')) for r in keep])
    if snap_to_network is not None:
        space = NetworkData.from_cartesian(snap_to_network, space)
        
    if st_only:
        res = np.hstack((times.reshape((len(times), 1)), space))
    else:
        for (r, t, s) in zip(keep, times, space):
            r['time'] = t
            r['location'] = s
        res = keep
    
    # order it by time
    order_idx = np.argsort(res[:, 0])
    res = res[order_idx]
    cr_num = cr_num[order_idx]
    return res, t0, cr_num
    

def basic_heatmap_plot(borough='py', crime_type='burglary', start_dt=None, end_dt=None,
bandwidths=(200, 200), npts=100, fmax=0.99):
    from plotting.utils import transparent_colour_map
    netobj = net.get_itn_network(borough=borough)
    data, t0, cr_num = get_cris_data(crime_type=crime_type, borough=borough, start_dt=start_dt, end_dt=end_dt)
    k = models.FixedBandwidthKde(data[:, 1:], bandwidths=bandwidths, parallel=False)
    xmin, ymin, xmax, ymax = netobj.extent
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, npts), np.linspace(ymin, ymax, npts))
    xy = DataArray.from_meshgrid(xx, yy)
    zz = k.pdf(xy, normed=False)
    netobj.plot_network()
    cmap = transparent_colour_map()
    if fmax:
        zs = zz.flatten()
        zs.sort()
        vmax = zs[int(zs.size * fmax)]
    else:
        vmax = zz.max()
    norm = colors.Normalize(vmin=0, vmax=vmax)
    levels = np.concatenate((np.linspace(0, vmax, 25), [zz.max()]))
    plt.contourf(xx, yy, zz, cmap=cmap, levels=levels, norm=norm)
    
    
#def temporal_trends_daily_multi_boroughs(
if __name__ == "__main__":
    boroughs=('ek', 'py', 'gd')
    crime_type='burglary'
    start_date=datetime.datetime(2014, 1, 1)
    end_date=datetime.datetime(2015, 1, 1)
    sharey=True
    
    from analysis.spacetime import DailyAggregate
    from jdi.plotting.trends import temporal_trend_plot_one

    left_buffer = 0.12
    fig, axs = plt.subplots(len(boroughs), sharex=True, sharey=sharey, figsize=(9.0, 9.0))
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_frame_on(False)
    big_ax.set_position([0., 0., 1. - 2 * left_buffer, 1.])
    big_ax.set_ylabel('Daily crime count', fontsize=12)

    for ax in axs:
        bbox = ax.get_position()
        ax.set_position([left_buffer, bbox.y0, 1 - 2 * left_buffer, bbox.height])    
            
    data = {}
    da = {}
    for i, b in enumerate(boroughs):
        ax = axs[i]
        data[b], t0, cr_num = get_cris_data(crime_type=crime_type, start_dt=start_date, end_dt=end_date, borough=b, convert_dt=False)
        da[b] = DailyAggregate(data[b])
        da[b].aggregate()
        res = da[b].data
        temporal_trend_plot_one(res, ax=ax)
        text_y = np.mean(ax.get_ylim())
        k = consts.BOROUGH_NAME_MAP[b.upper()]
        ax.legend((k,), fontsize=12, frameon=False)
        
    plt.show()

        
    