from network.itn import read_gml, ITNStreetNet
from data import models, iterator
import os
import pickle
import dill
from shapely import geometry
import settings
import datetime
import numpy as np
from matplotlib import pyplot as plt
from validation import validation, hotspot, roc
import plotting.spatial
from jdi.data import boundary, cris, net, consts
from jdi.plotting import geos
from jdi import OUT_DIR
import time
import sys


START_DATE = datetime.datetime(2013, 7, 1)
N_TRAIN = 180
N_VALIDATION = 60
N_TEST = 100
DIST_BETWEEN_SAMPLE_POINTS = 50

#N_TRAIN = 5
#N_VALIDATION = 5
#N_TEST = 5
#DIST_BETWEEN_SAMPLE_POINTS = 400


def camden_network_plot(netobj, boundary):

    netobj.plot_network()
    geos.plot_shapely_geos(boundary, ec='r', fc='none', lw=3)
    

def get_snapped_data(borough, major_crime_type=None, minor_crime_type=None):
    itn_net = net.get_itn_network(borough=borough)
    data, t0, cr_num = cris.get_cris_data(borough=borough, 
                                          start_dt=START_DATE, 
                                          major_crime_type=major_crime_type,
                                          minor_crime_type=minor_crime_type)    
    net_data = models.NetworkSpaceTimeData.from_cartesian(itn_net, data, grid_size=50)
    return net_data, t0, cr_num
    

def compute_cutoff_day(data, niter=N_VALIDATION):
    """
    When we ran the validation, we actually used as many days as it takes to get
    to 60 iterations, skipping days with no test crimes.
    This isn't stored, so we need a quick way to recompute the correct
    cutoff time.
    """
    rw = iterator.RollingOrigin(data, initial_cutoff_t=N_TRAIN)
    list(rw.iterator(60))
    return rw.cutoff_t


def validate_one(borough,
                 major_crime_type=None,
                 minor_crime_type=None):    

    borough = borough.lower()

    assert (major_crime_type is not None) or (minor_crime_type is not None), \
    "Must supply major or minor crime type"

    assert not ((major_crime_type is not None) and (minor_crime_type is not None)), \
    "Must supply major or minor crime type, not both"
    
    crime_type = major_crime_type if major_crime_type else minor_crime_type
    
    print "Network prediction validation. Crime type: %s, borough: %s" % (crime_type, borough)    
    
    out_base = os.path.join(
        OUT_DIR,
        'network_bandwidth_linearexponential',
        consts.CRIME_TYPE_NAME_MAP.get(crime_type),
    )

    out_dir = os.path.join(
        out_base,
        'validation',
    )
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(
        out_dir, "%s.dill" % borough,
    )
    
    out_dir_hr = os.path.join(
        out_base,
        'validation_hit_rate',
    )
    if not os.path.isdir(out_dir_hr):
        os.makedirs(out_dir_hr)
    out_file_hr = os.path.join(out_dir_hr, "%s.dill" % borough)
        
    net_data, t0, cr_num = get_snapped_data(borough, major_crime_type=major_crime_type, minor_crime_type=minor_crime_type)

    # get optimal bandwidth
    from optimise_kde_bandwidth_network import load_aggregated_results
    from analyse_bandwidth_optimisation import compute_optimum_bandwidth
    tt, dd, lltot = load_aggregated_results(crime_type)    
    topt, dopt = compute_optimum_bandwidth(tt, dd, lltot[borough])
    print "Loaded optimal bandwidths: %.1f days, %.1f metres" % (topt, dopt)
    
    # compute the number of days skipped due to no crimes - need to add these on
    # so that we start the testing on the correct date
    #cutoff_day = compute_cutoff_day(net_data)
    
    # FOR NOW, ASSUME THAT THE FIRST DAY OF TESTING IS FIXED
    # this is necessary to compare correctly with MA
    cutoff_day = N_TRAIN + N_VALIDATION
    
    sk = hotspot.STNetworkLinearSpaceExponentialTime(radius=dopt, time_decay=topt)
    vb = validation.NetworkValidationMean(net_data, sk, spatial_domain=None, include_predictions=True)
    vb.set_t_cutoff(cutoff_day)
    vb.set_sample_units(None, DIST_BETWEEN_SAMPLE_POINTS)

    tic = time.time()
    vb_res = vb.run(1, n_iter=N_TEST)
    toc = time.time()

    print "Done in %f s. Saving to file %s" % (toc - tic, out_file)

    tic = time.time()
    with open(out_file, 'wb') as f:
        dill.dump(vb_res, f)
    
    # save hit rates only to make reloading faster
    x = vb_res['cumulative_area']
    y = vb_res['cumulative_crime']
    with open(out_file_hr, 'wb') as f:
        dill.dump({'x': x, 'y': y}, f)
                
    toc = time.time()    
    print "Completed in %.1f s." % (toc - tic)

        
def validate_for_crime_type(major_crime_type=None, minor_crime_type=None):
    for bo in consts.BOROUGH_CODES:
        try:
            validate_one(bo.lower(), major_crime_type=major_crime_type, minor_crime_type=minor_crime_type)
        except Exception:
            print "Failed."    


if __name__ == "__main__":
    #borough = 'tx'
    #crime_type = 'Burglary'
    #validate_one(borough, crime_type)
    pass