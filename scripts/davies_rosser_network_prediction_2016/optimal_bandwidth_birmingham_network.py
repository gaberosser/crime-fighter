__author__ = 'gabriel'
from kde import optimisation
from database.birmingham.loader import load_network, BirminghamCrimeLoader
import dill
import numpy as np
from network.point import NetPointArray, NetTimePointArray
from data.models import NetworkSpaceTimeData, SpaceTimeDataArray
import datetime
import sys

START_DATE = datetime.datetime(2013, 7, 1)  # first date for which data are required
N_PT = 50  # numer of parameter values in each dimension
PARAM_EXTENT = (1., 90., 50., 2000.)  # tmin, tmax, dmin, dmax


if __name__ == "__main__":

    if len(sys.argv) == 3:
        start_day_number = int(sys.argv[1])
        num_validation = int(sys.argv[2])
        print "Using command line args. Start day: %d, num_val: %d" % (start_day_number, num_validation)
    else:
        num_validation = 60  # number of prediction time windows
        # num_validation = 1  # number of prediction time windows
        start_day_number = 180  # number of days (after start date) on which first prediction is made
        # start_day_number = 2  # number of days (after start date) on which first prediction is made

    max_time_window = 24
    # max_time_window = None
    aoristic_method = 'start'


    end_date = START_DATE + datetime.timedelta(days=start_day_number + num_validation + 1)

    # load crime data
    obj = BirminghamCrimeLoader(aoristic_method=aoristic_method, max_time_window=max_time_window)
    data, t0, cid = obj.get_data(start_date=START_DATE,
                                 end_date=end_date)

    # load network
    net = load_network()

    # snap
    snapped_data, failed = NetTimePointArray.from_cartesian(net, data, return_failure_idx=True)

    opt = optimisation.NetworkFixedBandwidth(snapped_data, initial_cutoff=start_day_number)
    opt.set_logger(verbose=True)
    opt.set_parameter_grid(N_PT, *PARAM_EXTENT)

    opt.run(num_validation)
    filestem = "network_linearexponentialkde_start_day_%d_%d_iterations_%s" % (start_day_number, num_validation, aoristic_method)
    if max_time_window:
        filestem += "_max%dhr" % max_time_window
    filename = filestem + ".dill"

    tt, dd = zip(opt.grid)
    tt = tt[0]
    dd = dd[0]

    with open(filename, 'w') as f:
        dill.dump({
            'tt': tt,
            'dd': dd,
            'll': opt.res_arr
        }, f)