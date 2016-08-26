__author__ = 'gabriel'
from kde import optimisation
from database.birmingham.loader import BirminghamCrimeLoader
from network import itn
import dill
import numpy as np
from data.models import NetworkSpaceTimeData, SpaceTimeDataArray
import datetime
import sys

START_DATE = datetime.datetime(2013, 7, 1)  # first date for which data are required


if __name__ == "__main__":
    start_day_number = 180  # number of days (after start date) on which first prediction is made
    num_validation = 60  # number of prediction time windows

    max_time_window = 24
    # max_time_window = None
    aoristic_method = 'start'

    end_date = START_DATE + datetime.timedelta(days=start_day_number + num_validation + 1)

    # load crime data
    obj = BirminghamCrimeLoader(aoristic_method=aoristic_method, max_time_window=max_time_window)
    data, t0, cid = obj.get_data(start_date=START_DATE,
                                 end_date=end_date)

    # load network
    NET_FILE = '/home/gabriel/data/birmingham/street_network/birmingham_street_net.pickle'
    net = itn.ITNStreetNet.from_pickle(NET_FILE)
    # net = load_network()

    # snap
    snapped_data, failed = NetworkSpaceTimeData.from_cartesian(net, data, return_failure_idx=True)

    post_snap_xy = snapped_data.space.to_cartesian().data
    pre_snap_xy = data[:, 1:]