__author__ = 'gabriel'
from database import models
import numpy as np
import cad
import collections
import datetime
import pytz

# s_max = 500  # metres
# t_max = 90  # days
#
# camden = cad.get_camden_region()
# contract = camden.buffer(-s_max)
#
# target_data, t0, cid = cad.get_crimes_by_type(nicl_type=None)
# source_data, tmp2, cid_contract = cad.get_crimes_by_type(nicl_type=None, spatial_domain=contract)
#
# n_full = cid.size
# map_contract = np.array([np.any(cid_contract == cid[i]) for i in range(n_full)])
# idx_contract = np.arange(n_full)[map_contract]
# idx_buffer = np.arange(n_full)[~map_contract]


def month_iterator(start_date, end_date):
    try:
        start_date.tzinfo
    except AttributeError:
        kwargs = {}
    else:
        kwargs = {'tzinfo': pytz.utc}

    this_type = type(start_date)
    sd = start_date
    while sd < end_date:
        next_year = sd.year
        next_month = sd.month + 1
        if sd.month == 12:
            next_year += 1
            next_month = 1
        ed = this_type(next_year, next_month, 1, **kwargs)
        if ed <= end_date:
            yield (sd, ed)
        else:
            yield (sd, end_date+datetime.timedelta(days=1))
        sd = ed


def week_iterator(start_date, end_date):

    sd = start_date
    while sd < end_date:
        ed = sd + datetime.timedelta(days=7)
        if ed <= end_date:
            yield (sd, ed)
        else:
            yield (sd, end_date+datetime.timedelta(days=1))
        sd = ed


def daily_iterator(start_date, end_date, days=1):
    sd = start_date
    while sd < end_date:
        ed = sd + datetime.timedelta(days=days)
        if ed <= end_date:
            yield (sd, ed)
        else:
            yield (sd, end_date+datetime.timedelta(days=1))
        sd = ed


class Aggregate(object):
    def __init__(self, data, start_date=None, end_date=None, *args, **kwargs):
        self._start_date = start_date
        self._end_date = end_date
        self.raw_data = data
        # strip out timezone if supplied
        if isinstance(self.raw_data[0, 0], datetime.datetime):
            new_dt = [t.replace(tzinfo=None) for t in self.raw_data[:, 0]]
            self.raw_data[:, 0] = new_dt

    @property
    def start_date(self):
        if self._start_date:
            return self._start_date
        return self.raw_data[:, 0].min()

    @property
    def end_date(self):
        if self._end_date:
            return self._end_date
        return self.raw_data[:, 0].max()

    def aggregate(self):
        raise NotImplementedError()


class TemporalAggregate(Aggregate):
    def __init__(self, data, start_date=None, end_date=None):
        super(TemporalAggregate, self).__init__(data, start_date=start_date, end_date=end_date)
        self.data = self.bucket_data()

    @property
    def bucket_dict(self):
        raise NotImplementedError

    def bucket_data(self):
        res = collections.OrderedDict()
        for k, func in self.bucket_dict.items():
            res[k] = [x for x in self.raw_data if func(x)]
        return res


class DailyAggregate(TemporalAggregate):
    @staticmethod
    def create_bucket_fun(sd, ed):
        """ Interesting! If we just generate the lambda function inline in bucket_dict, the scope is updated each time,
         so the parameters (sd, ed) are also updated.  In essence the final function created is run every time.
         Instead use a function factory to capture the closure correctly. """
        return lambda x: sd <= x[0] < ed

    @property
    def bucket_dict(self):
        gen_day = daily_iterator(self.start_date, self.end_date)
        return collections.OrderedDict(
            [(sd, self.create_bucket_fun(sd, ed)) for sd, ed in gen_day]
        )