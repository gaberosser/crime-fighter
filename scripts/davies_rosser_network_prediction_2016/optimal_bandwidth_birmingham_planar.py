__author__ = 'gabriel'
from kde import optimisation, models
from database.birmingham.loader import load_network, BirminghamCrimeLoader
import dill
import numpy as np
import datetime
import sys


START_DATE = datetime.datetime(2013, 7, 1)  # first date for which data are required
# START_DAY_NUMBER = 180  # number of days (after start date) on which first prediction is made
# NUM_VALIDATION = 10  # number of prediction time windows
N_PT = 50  # numer of parameter values in each dimension
PARAM_EXTENT = (1., 90., 50., 2000.)  # tmin, tmax, dmin, dmax
NCPU = False  # use maximum number of CPUs for parallel processing

if __name__ == "__main__":

    # START_DAY_NUMBER = int(sys.argv[1])
    # NUM_VALIDATION = int(sys.argv[2])

    max_time_window = 24
    aoristic_method = 'start'
    num_validation = 60  # number of prediction time windows
    start_day_number = 180  # number of days (after start date) on which first prediction is made

    # load crime data
    end_date = START_DATE + datetime.timedelta(days=start_day_number + num_validation + 1)

    # load crime data
    obj = BirminghamCrimeLoader(aoristic_method=aoristic_method, max_time_window=max_time_window)
    data, t0, cid = obj.get_data(start_date=START_DATE,
                                 end_date=end_date)

    opt = optimisation.PlanarFixedBandwidth(data, data_index=cid, initial_cutoff=start_day_number,
                                            parallel=NCPU, kde_class=models.FixedBandwidthLinearSpaceExponentialTimeKde)
    opt.set_logger(verbose=True)
    opt.set_parameter_grid(N_PT, *PARAM_EXTENT)
    opt.run(num_validation)

    tt, dd = zip(opt.grid)
    tt = tt[0]
    dd = dd[0]

    with open(
        "planar_linearexponentialkde_start_day_%d_%d_iterations_start_max24hr.dill" % (start_day_number, num_validation),
        'w'
    ) as f:
        dill.dump({
            'tt': tt,
            'dd': dd,
            'll': opt.res_arr
        }, f)
