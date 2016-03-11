from kde import models, optimisation
from jdi.data import consts, cris, net
import datetime
import sys
from settings import JDI_OUT_DIR
import os
import pickle
import dill
import numpy as np
import operator

START_DATE = datetime.datetime(2013, 7, 1)
NCPU = False
N_TRAIN = 180
N_VALIDATION = 60
N_PARAM = 30
EXTRA_DAYS = N_VALIDATION  # extra days to avoid running out if some are skipped

#NCPU = 4
#N_TRAIN = 36
#N_VALIDATION = 10
#N_PARAM = 50


PARAM_EXTENT = (1., 120., 50., 1000.)  # tmin, tmax, dmin, dmax


def run_one(borough,
             major_crime_type=None,
             minor_crime_type=None,
             overwrite=False):
    
    assert (major_crime_type is not None) or (minor_crime_type is not None), \
    "Must supply major or minor crime type"
    
    assert not ((major_crime_type is not None) and (minor_crime_type is not None)), \
    "Must supply major or minor crime type, not both" 
    
    crime_type = major_crime_type if major_crime_type else minor_crime_type   
    
    out_dir = os.path.join(
        JDI_OUT_DIR,
        'planar_bandwidth_linearexponential',        
        consts.CRIME_TYPE_NAME_MAP.get(crime_type),
        'optimal_bandwidths',        
    )

    out_file = os.path.join(
        out_dir, "%s.dill" % borough,
    )    
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)    
    
    if os.path.isfile(out_file) and not overwrite:
        print "File %s already exists, skipping." % out_file
        return
    
    # load crime data
    # add 2 full days on as datetimes are used
    end_date = START_DATE + datetime.timedelta(days=N_TRAIN + N_VALIDATION + EXTRA_DAYS + 1)

    # load crime data
    data, t0, cid = cris.get_cris_data(start_dt=START_DATE, 
                                       end_dt=end_date, 
                                       major_crime_type=major_crime_type,
                                       minor_crime_type=minor_crime_type,
                                       borough=borough)

    opt = optimisation.PlanarFixedBandwidth(data, data_index=cid, initial_cutoff=N_TRAIN,
                                            parallel=NCPU, kde_class=models.FixedBandwidthLinearSpaceExponentialTimeKde)
    opt.set_logger(verbose=True)
    opt.set_parameter_grid(N_PARAM, *PARAM_EXTENT)
    
    print "Borough: %s, crime_type: %s. Running optimisation..." % (borough, crime_type)    
    opt.run(N_VALIDATION)

    tt, dd = zip(opt.grid)
    tt = tt[0]
    dd = dd[0]

    print "Done. Saving to file %s" % out_file    

    with open(out_file, 'wb') as f:
        pickle.dump({
            'tt': tt,
            'dd': dd,
            'll': opt.res_arr
        }, f)
        

def run_for_crime_type(major_crime_type=None, minor_crime_type=None):
    for bo in consts.BOROUGH_CODES:
        try:
            run_one(bo.lower(), major_crime_type=major_crime_type, minor_crime_type=minor_crime_type)
        except Exception:
            print "Failed."


def compute_and_save_aggregated_likelihood_grids_one_crime(crime_type):
    from analyse_bandwidth_optimisation import \
    compute_and_save_aggregated_likelihood_grids_one_crime as casalgo
    subdir = 'planar_bandwidth_linearexponential'
    return casalgo(crime_type, subdir)


def load_aggregated_results(crime_type):
    fn = os.path.join(
        JDI_OUT_DIR,
        'planar_bandwidth_linearexponential',
        'aggregated_likelihoods',
        '%s.pkl' % consts.CRIME_TYPE_NAME_MAP.get(crime_type)        
    )
    with open(fn ,'rb') as f:
        tt, dd, lltot = dill.load(f)
    return tt, dd, lltot


if __name__ == "__main__":
    boroughs = ('tx', 'tw', 'fh', 'bs', 'ni', 'ww', 'yr', 'ye', 'ji')
    # boroughs = ('ni', 'qk', 'sx', 'yr', 'ek', 'cw')
    crime_type = 'Robbery'
    fmin = 0.25
    from matplotlib import pyplot as plt
    from plotting.utils import abs_bound_from_rel
    from analyse_bandwidth_optimisation import compute_optimum_bandwidth
    tt, dd, lltot = load_aggregated_results(crime_type)
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(12.5, 10.))
    
    for i, bo in enumerate(boroughs):
        if i == axs.size:
            break
        if bo not in lltot:
            continue
        ax = axs.flat[i]
        v = lltot[bo]
        vmin = abs_bound_from_rel(v, 0.25)
        vmax = v.max()
        bins = np.linspace(vmin, vmax, 50)
        ax.contourf(tt, dd, v, bins, cmap='Reds')
        ax.set_title(consts.BOROUGH_NAME_MAP[bo.upper()])
        topt, dopt = compute_optimum_bandwidth(tt, dd, v)
        ax.plot([0, topt], [dopt, dopt], 'k--')
        ax.plot([topt, topt], [0, dopt], 'k--')

    plt.tight_layout(pad=0.3, rect=(0.04, 0.04, 0.99, 0.98))

    #crime_type = sys.argv[1]
    #run_for_crime_type(crime_type)
