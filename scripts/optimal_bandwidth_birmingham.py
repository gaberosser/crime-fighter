__author__ = 'gabriel'
from kde import optimisation
from database.birmingham.loader import load_network, BirminghamCrimeLoader
import pickle
import numpy as np
from data.models import NetworkSpaceTimeData, SpaceTimeDataArray
import datetime


if __name__ == "__main__":

    npt = 100
    num_validation = 60  # number of prediction time windows
    param_extent = (1., 90., 50., 2000.)
    start_date = datetime.datetime(2013, 7, 1)  # first date for which data are required
    start_day_number = 180  # number of days (after start date) on which first prediction is made
    end_date = start_date + datetime.timedelta(days=start_day_number + num_validation + 1)

    # load crime data
    obj = BirminghamCrimeLoader()
    data, t0, cid = obj.get_data(start_date=start_date,
                                 end_date=end_date)

    # load network
    net = load_network()

    # snap
    snapped_data, failed = NetworkSpaceTimeData.from_cartesian(net, data, return_failure_idx=True)
    opt = optimisation.NetworkFixedBandwidth(snapped_data, initial_cutoff=start_day_number)
    opt.set_logger(verbose=True)
    opt.set_parameter_grid(npt, *param_extent)

    opt.run(num_validation)