from jdi_scripts.cris_network_prediction import get_snapped_data
from data import models
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
from jdi.data import boundary, consts
from jdi.plotting import geos
from jdi import OUT_DIR
import time


START_DATE = datetime.datetime(2013, 7, 1)
N_TRAIN = 180
N_VALIDATION = 60
N_TEST = 100
GRID_LENGTH = 150
N_SAMPLES_PER_GRID = 10
#N_TRAIN = 5
#N_VALIDATION = 5
#N_TEST = 5
#GRID_LENGTH = 400
#N_SAMPLES_PER_GRID = 2


def validate_one(borough,
                 major_crime_type=None,
                 minor_crime_type=None):    

    borough = borough.lower()

    assert (major_crime_type is not None) or (minor_crime_type is not None), \
    "Must supply major or minor crime type"

    assert not ((major_crime_type is not None) and (minor_crime_type is not None)), \
    "Must supply major or minor crime type, not both"
    
    crime_type = major_crime_type if major_crime_type else minor_crime_type
    
    print "Planar prediction validation. Crime type: %s, borough: %s" % (crime_type, borough)

    out_dir_grid = os.path.join(
        OUT_DIR,
        'planar_bandwidth_linearexponential',
        consts.CRIME_TYPE_NAME_MAP.get(crime_type),
        'validation_grid',
    )
    if not os.path.isdir(out_dir_grid):
        os.makedirs(out_dir_grid)
    out_file_grid = os.path.join(out_dir_grid,
                                 '%s.dill' % borough)

    out_dir_grid_hr = os.path.join(
        OUT_DIR,
        'planar_bandwidth_linearexponential',
        consts.CRIME_TYPE_NAME_MAP.get(crime_type),
        'validation_grid_hit_rate',
    )
    if not os.path.isdir(out_dir_grid_hr):
        os.makedirs(out_dir_grid_hr)
    out_file_grid_hr = os.path.join(out_dir_grid_hr,
                                    '%s.dill' % borough)
        
    out_dir_intersection = os.path.join(
        OUT_DIR,
        'planar_bandwidth_linearexponential',
        consts.CRIME_TYPE_NAME_MAP.get(crime_type),
        'validation_intersection',
    )
    if not os.path.isdir(out_dir_intersection):
        os.makedirs(out_dir_intersection)
    out_file_intersection = os.path.join(out_dir_intersection,
                                 '%s.dill' % borough)

    out_dir_intersection_hr = os.path.join(
        OUT_DIR,
        'planar_bandwidth_linearexponential',
        consts.CRIME_TYPE_NAME_MAP.get(crime_type),
        'validation_intersection_hit_rate',
    )
    if not os.path.isdir(out_dir_intersection_hr):
        os.makedirs(out_dir_intersection_hr)
    out_file_intersection_hr = os.path.join(out_dir_intersection_hr,
                                            '%s.dill' % borough)

    borough = borough.lower()
    net_data, t0, cr_num = get_snapped_data(borough, major_crime_type=major_crime_type, minor_crime_type=minor_crime_type)
    data_txy = net_data.time.adddim(net_data.space.to_cartesian(), type=models.CartesianSpaceTimeData)
    domain = boundary.get_borough_boundary(borough)

    # get optimal bandwidth
    from jdi_scripts.optimise_kde_bandwidth_planar import load_aggregated_results
    from jdi_scripts.analyse_bandwidth_optimisation import compute_optimum_bandwidth
    tt, dd, lltot = load_aggregated_results(crime_type)    
    topt, dopt = compute_optimum_bandwidth(tt, dd, lltot[borough])
    print "Loaded optimal bandwidths: %.1f days, %.1f metres" % (topt, dopt)
    
    # compute the number of days skipped due to no crimes - need to add these on
    # so that we start the testing on the correct date
    #cutoff_day = compute_cutoff_day(net_data)
    
    # FOR NOW, ASSUME THAT THE FIRST DAY OF TESTING IS FIXED
    # this is necessary to compare correctly with MA
    cutoff_day = N_TRAIN + N_VALIDATION
    
    sk_planar = hotspot.STLinearSpaceExponentialTime(radius=dopt, mean_time=topt)
    vb_grid = validation.ValidationIntegration(data_txy, sk_planar, spatial_domain=domain, include_predictions=True)
    vb_grid.set_t_cutoff(cutoff_day)
    vb_grid.set_sample_units(GRID_LENGTH, N_SAMPLES_PER_GRID)

    tic = time.time()
    vb_res_grid = vb_grid.run(1, n_iter=N_TEST)
    toc = time.time()
    print "Grid version completed in %.1f s. Saving to file %s" % (toc - tic, out_file_grid)

    tic = time.time()    
    with open(out_file_grid, 'wb') as f:
        dill.dump(vb_res_grid, f)
        
    # save hit rates only to make reloading faster
    x = vb_res_grid['cumulative_area']
    y = vb_res_grid['cumulative_crime']
    with open(out_file_grid_hr, 'wb') as f:
        dill.dump({'x': x, 'y': y}, f)

    toc = time.time()   
    print "Completed in %.1f s." % (toc - tic)


    # compare with grid-based method using intersecting network segments to measure sample unit size
    vb_intersect = validation.ValidationIntegrationByNetworkSegment(
        data_txy, sk_planar, spatial_domain=domain, graph=net_data.graph
    )
    vb_intersect.set_t_cutoff(cutoff_day)
    vb_intersect.set_sample_units(GRID_LENGTH, N_SAMPLES_PER_GRID)

    print "Starting again with the net intersection method"
    tic = time.time()
    vb_res_intersect = vb_intersect.run(1, n_iter=N_TEST)
    toc = time.time()
    print "Intersection version completed in %.1f s. Saving to file %s" % (toc - tic, out_file_intersection)
    
    tic = time.time()    
    with open(out_file_intersection, 'wb') as f:
        dill.dump(vb_res_intersect, f)
        
    # save hit rates only to make reloading faster
    x = vb_res_intersect['cumulative_area']
    y = vb_res_intersect['cumulative_crime']
    with open(out_file_intersection_hr, 'wb') as f:
        dill.dump({'x': x, 'y': y}, f)

    toc = time.time()
    print "Completed in %.1f s." % (toc - tic)
    

def validate_for_crime_type(crime_type):
    for bo in consts.BOROUGH_CODES:
        try:
            validate_one(crime_type, bo.lower())
        except Exception:
            print "Failed."    
    
    
if __name__ == "__main__":
    #borough = 'tx'
    #crime_type = 'Burglary'
    #validate_one(borough, crime_type)
    import sys
    crime_type = sys.argv[1]
    validate_for_crime_type(crime_type)