__author__ = 'gabriel'
from kde import optimisation, models
from database.birmingham.loader import load_network, BirminghamCrimeLoader
import dill
import numpy as np
import datetime


START_DATE = datetime.datetime(2013, 7, 1)  # first date for which data are required
START_DAY_NUMBER = 180  # number of days (after start date) on which first prediction is made
NUM_VALIDATION = 60  # number of prediction time windows
N_PT = 50  # numer of parameter values in each dimension
PARAM_EXTENT = (1., 90., 50., 2000.)  # tmin, tmax, dmin, dmax
NCPU = 4  # use maximum number of CPUs for parallel processing

if __name__ == "__main__":

    # load crime data
    end_date = START_DATE + datetime.timedelta(days=START_DAY_NUMBER + NUM_VALIDATION + 1)

    # load crime data
    obj = BirminghamCrimeLoader()
    data, t0, cid = obj.get_data(start_date=START_DATE,
                                 end_date=end_date)

    opt = optimisation.PlanarFixedBandwidth(data, data_index=cid, initial_cutoff=START_DAY_NUMBER,
                                            parallel=NCPU, kde_class=models.FixedBandwidthLinearSpaceExponentialTimeKde)
    opt.set_logger(verbose=True)
    opt.set_parameter_grid(N_PT, *PARAM_EXTENT)
    opt.run(NUM_VALIDATION)

    tt, dd = zip(opt.grid)
    tt = tt[0]
    dd = dd[0]

    with open("planar_linearexponentialkde_start_2013_07_01_60_days.dill", 'w') as f:
        dill.dump({
            'tt': tt,
            'dd': dd,
            'll': opt.res_arr
        }, f)
